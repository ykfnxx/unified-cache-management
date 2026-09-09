/**
 * MIT License
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include <algorithm>
#include <atomic>
#include <chrono>
#include <list>
#include <mutex>
#include <numeric>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include "kv_block.h"
#include "logger/logger.h"
#include "metrics_api.h"
#include "transfer_queue.h"
#include "ucmstore_v1.h"

namespace UC::OnEvictCacheStore {

class OnEvictCacheStore final : public StoreV1 {
    using BlockId = Detail::BlockId;
    using Clock = std::chrono::steady_clock;
    using LruList = std::list<BlockId>;

    struct LruEntry {
        LruList::iterator position;
        Clock::time_point lastAccessTime;
    };

    struct RadixNode {
        BlockId block;
        BlockId parent{};
        bool hasParent{false};
        bool resident{false};
        size_t residentChildCount{0};
        uint64_t lastAccessSeq{0};
        Clock::time_point lastAccessTime;
    };

    using ProtectedBlocks = std::unordered_set<BlockId, Detail::BlockIdHasher>;
    using LeafKey = std::pair<uint64_t, BlockId>;

public:
    ~OnEvictCacheStore() override
    {
        if (transfers_) {
            // Unregister host memory on the transfer thread's device context,
            // after all previously submitted transfers have completed.
            auto cleanup = transfers_->Submit({NextId(),
                                               [this](Trans::Stream&) {
                                                   payloads_.clear();
                                                   return Status::OK();
                                               },
                                               Status::OK()});
            if (cleanup) { transfers_->Wait(cleanup.Value()); }
            transfers_.reset();
        }
    }

    Status Setup(const Detail::Dictionary& config) override
    {
        config.Get("store_backend", backend_);
        config.GetNumber("block_size", blockSize_);
        realKV_ = config.Contains("fake_res_cap");
        config.GetNumber(realKV_ ? "fake_res_cap" : "on_evict_cache_capacity_gb", capacityGb_);
        if (realKV_) { policy_ = "radix_lru"; }
        config.Get("on_evict_cache_policy", policy_);
        config.GetNumber("on_evict_cache_dump_max_idle_s", dumpMaxIdleSeconds_);

        if (!realKV_ && backend_ == nullptr) {
            return Status::InvalidParam("invalid store backend");
        }
        if (capacityGb_ == 0) { return Status::InvalidParam("invalid cache capacity"); }
        if (policy_ != "lru" && policy_ != "radix_lru") {
            return Status::InvalidParam("unsupported eviction policy");
        }
        if (realKV_) {
            config.Get("unique_id", uniqueId_);
            config.GetNumber("device_id", deviceId_);
            if (uniqueId_.empty()) { return Status::InvalidParam("unique_id is required"); }
            // Scheduler instances only query published shared-memory blocks.
            if (deviceId_ < 0) {
                UC_INFO(
                    "OnEvict setup role=scheduler unique_id={} policy={} fake_res_cap_gib={} "
                    "idle_s={}",
                    uniqueId_, policy_, capacityGb_, dumpMaxIdleSeconds_);
                return Status::OK();
            }
            config.GetNumber("shard_size", shardSize_);
            size_t tensorSize = 0;
            config.GetNumber("tensor_size", tensorSize);
            if (tensorSize != 0 && shardSize_ != 0) {
                tensorSizes_.assign(shardSize_ / tensorSize, tensorSize);
            } else {
                config.GetNumbers("tensor_size_list", tensorSizes_);
            }
            if (shardSize_ == 0 || blockSize_ % shardSize_ != 0 || tensorSizes_.empty() ||
                std::accumulate(tensorSizes_.begin(), tensorSizes_.end(), size_t{0}) > shardSize_) {
                return Status::InvalidParam("invalid KV block/shard/tensor sizes");
            }
        }
        if (blockSize_ == 0) { return Status::InvalidParam("invalid block size"); }

        capacityBlocks_ = (capacityGb_ << 30) / blockSize_;
        if (capacityBlocks_ == 0) { return Status::InvalidParam("cache capacity is too small"); }
        if (realKV_) {
            size_t timeoutMs = 30000;
            config.GetNumber("timeout_ms", timeoutMs);
            auto transfers = std::make_unique<TransferQueue>();
            auto status = transfers->Setup(deviceId_, timeoutMs);
            if (status.Failure()) { return status; }
            transfers_ = std::move(transfers);
        }
        UC_INFO(
            "OnEvict setup real_kv={} device={} unique_id={} policy={} capacity_gib={} "
            "capacity_blocks={} block_bytes={} shard_bytes={} idle_s={}",
            realKV_, deviceId_, uniqueId_, policy_, capacityGb_, capacityBlocks_, blockSize_,
            shardSize_, dumpMaxIdleSeconds_);
        return Status::OK();
    }

    std::string Readme() const override { return "OnEvictCacheStore"; }

    Expected<std::vector<uint8_t>> Lookup(const BlockId* blocks, size_t num) override
    { return LookupAt(blocks, num, Clock::now()); }

    Expected<ssize_t> LookupOnPrefix(const BlockId* blocks, size_t num) override
    {
        auto looked = Lookup(blocks, num);
        if (!looked) { return looked.Error(); }
        const auto& founds = looked.Value();
        for (size_t i = 0; i < founds.size(); ++i) {
            if (!founds[i]) { return static_cast<ssize_t>(i) - 1; }
        }
        return static_cast<ssize_t>(num) - 1;
    }

    Expected<ssize_t> LookupOnPrefix(const BlockId* blocks, size_t num,
                                     uint64_t logicalTimeNs) override
    {
        auto looked = LookupAt(blocks, num, LogicalTime(logicalTimeNs));
        if (!looked) { return looked.Error(); }
        const auto& founds = looked.Value();
        for (size_t i = 0; i < founds.size(); ++i) {
            if (!founds[i]) { return static_cast<ssize_t>(i) - 1; }
        }
        return static_cast<ssize_t>(num) - 1;
    }

    Expected<ssize_t> LookupOnReverse(const BlockId* blocks, size_t num) override
    {
        auto looked = Lookup(blocks, num);
        if (!looked) { return looked.Error(); }
        const auto& founds = looked.Value();
        for (size_t i = founds.size(); i > 0; --i) {
            if (founds[i - 1]) { return static_cast<ssize_t>(i - 1); }
        }
        return static_cast<ssize_t>(-1);
    }

    void Prefetch(const BlockId*, size_t) override {}

    Status ObserveRequest(const BlockId* blocks, size_t num) override
    {
        if (policy_ == "lru") { return Status::OK(); }
        std::lock_guard<std::mutex> lock(mutex_);
        ObserveRequestLocked(blocks, num, Clock::now());
        return Status::OK();
    }

    Expected<Detail::TaskHandle> Load(Detail::TaskDesc task) override
    {
        if (!realKV_) { return NextId(); }
        const auto shards = task.size();
        return transfers_->Submit({NextId(),
                                   [this, task = std::move(task)](Trans::Stream& stream) mutable {
                                       return LoadKV(task, stream);
                                   },
                                   Status::OK(), "load", shards, shards * shardSize_});
    }

    Expected<Detail::TaskHandle> Dump(Detail::TaskDesc task) override
    {
        if (policy_ == "radix_lru") {
            return Status::InvalidParam("radix_lru requires request context");
        }
        if (realKV_) { return SubmitDump(std::move(task), {}, Clock::now()); }
        std::lock_guard<std::mutex> lock(mutex_);
        auto status = DumpLru(task, Clock::now());
        if (status.Failure()) { return status; }
        return NextId();
    }

    Expected<Detail::TaskHandle> Dump(Detail::TaskDesc task,
                                      const Detail::RequestAwareDumpContext& context) override
    {
        if (realKV_) { return SubmitDump(std::move(task), context, Clock::now()); }
        std::lock_guard<std::mutex> lock(mutex_);
        return DumpWithContext(task, context, Clock::now());
    }

    Expected<Detail::TaskHandle> Dump(Detail::TaskDesc task,
                                      const Detail::RequestAwareDumpContext& context,
                                      uint64_t logicalTimeNs) override
    {
        if (realKV_) { return SubmitDump(std::move(task), context, LogicalTime(logicalTimeNs)); }
        std::lock_guard<std::mutex> lock(mutex_);
        return DumpWithContext(task, context, LogicalTime(logicalTimeNs));
    }

    Expected<bool> Check(Detail::TaskHandle task) override
    { return realKV_ ? transfers_->Check(task) : Expected<bool>{true}; }

    Status Wait(Detail::TaskHandle task) override
    { return realKV_ ? transfers_->Wait(task) : Status::OK(); }

private:
    friend class OnEvictCacheStoreTestPeer;

    std::string BlockName(const BlockId& block) const
    {
        static constexpr char hex[] = "0123456789abcdef";
        std::string name = "/uc_on_evict_" + uniqueId_ + "_";
        for (auto byte : block) {
            auto value = std::to_integer<unsigned>(byte);
            name += hex[value >> 4];
            name += hex[value & 15];
        }
        return name;
    }

    Expected<Detail::TaskHandle> SubmitDump(Detail::TaskDesc task,
                                            Detail::RequestAwareDumpContext context,
                                            Clock::time_point now)
    {
        const auto shards = task.size();
        return transfers_->Submit(
            {NextId(),
             [this, task = std::move(task), context = std::move(context),
              now](Trans::Stream& stream) mutable { return DumpKV(task, context, now, stream); },
             Status::OK(), "dump", shards, shards * shardSize_});
    }

    Status LoadKV(Detail::TaskDesc& task, Trans::Stream& stream)
    {
        std::vector<std::unique_ptr<KVBlock>> buffers;
        auto status = Status::OK();
        for (auto& shard : task) {
            auto block = std::make_unique<KVBlock>();
            status = block->Open(BlockName(shard.owner), blockSize_);
            if (status.Failure()) {
                UC_ERROR("OnEvict load open/register failed block={} shard={} status={}",
                         BlockName(shard.owner), shard.index, status);
                break;
            }
            auto* host = block->Shard(shard.index, shardSize_);
            buffers.push_back(std::move(block));
            status = stream.HostToDeviceAsync(host, shard.addrs.data(), tensorSizes_);
            if (status.Failure()) {
                UC_ERROR("OnEvict H2D submit failed block={} shard={} status={}",
                         BlockName(shard.owner), shard.index, status);
                break;
            }
        }
        // Mappings pin the payload until every submitted H2D has finished,
        // including when a later shard misses or a transfer submission fails.
        auto synced = stream.Synchronized();
        if (synced.Failure()) { UC_ERROR("OnEvict H2D sync failed status={}", synced); }
        return status.Failure() ? status : synced;
    }

    Status DumpKV(Detail::TaskDesc& task, const Detail::RequestAwareDumpContext& context,
                  Clock::time_point now, Trans::Stream& stream)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (task.prerequisiteHandle != 0) {
            auto status = stream.WaitEvent(Trans::Event{task.prerequisiteHandle});
            if (status.Failure()) {
                UC_ERROR("OnEvict dump prerequisite wait failed event={} status={}",
                         task.prerequisiteHandle, status);
                return status;
            }
        }
        auto status = Status::OK();
        size_t copiedShards = 0;
        size_t publishedBlocks = 0;
        for (auto& shard : task) {
            auto& block = payloads_[shard.owner];
            if (!block) {
                block = std::make_unique<KVBlock>();
                status = block->Create(BlockName(shard.owner), blockSize_, blockSize_ / shardSize_);
                if (status.Failure()) {
                    UC_ERROR(
                        "OnEvict dump allocate/register failed block={} block_bytes={} status={}",
                        BlockName(shard.owner), blockSize_, status);
                    payloads_.erase(shard.owner);
                    break;
                }
            }
            if (block->published || block->shardsReady[shard.index]) { continue; }
            status = stream.DeviceToHostAsync(shard.addrs.data(),
                                              block->Shard(shard.index, shardSize_), tensorSizes_);
            if (status.Failure()) {
                UC_ERROR("OnEvict D2H submit failed block={} shard={} status={}",
                         BlockName(shard.owner), shard.index, status);
                break;
            }
            ++copiedShards;
        }
        auto synced = stream.Synchronized();
        if (status.Failure()) { return status; }
        if (synced.Failure()) {
            UC_ERROR("OnEvict D2H sync failed status={}", synced);
            return synced;
        }
        for (const auto& shard : task) {
            payloads_.at(shard.owner)->shardsReady[shard.index] = true;
            // Refresh the whole batch before admission can evict an older entry.
            auto resident = entries_.find(shard.owner);
            if (resident != entries_.end()) { TouchLru(resident, now); }
        }

        ProtectedBlocks protectedBlocks;
        for (const auto& request : context) {
            protectedBlocks.insert(request.requestBlocks.begin(), request.requestBlocks.end());
            ObserveRequestLocked(request.requestBlocks.data(), request.requestBlocks.size(), now);
        }
        for (const auto& shard : task) {
            auto& block = *payloads_.at(shard.owner);
            if (block.published) {
                auto resident = entries_.find(shard.owner);
                if (resident != entries_.end()) { TouchLru(resident, now); }
                continue;  // Includes Dumped: do not promote it.
            }
            if (!std::all_of(block.shardsReady.begin(), block.shardsReady.end(),
                             [](bool ready) { return ready; })) {
                continue;
            }
            if (policy_ == "radix_lru") {
                status = AdmitRadix(shard.owner, protectedBlocks, now);
            } else {
                Detail::TaskDesc admission{
                    Detail::Shard{shard.owner, 0, {}}
                };
                status = DumpLru(admission, now);
            }
            if (status.Failure()) { return status; }
            status = block.Publish(BlockName(shard.owner));
            if (status.Failure()) {
                UC_ERROR("OnEvict publish failed block={} status={}", BlockName(shard.owner),
                         status);
                return status;
            }
            ++publishedBlocks;
        }
        UC_DEBUG(
            "OnEvict dump shards={} copied_shards={} copied_payload_bytes={} published_blocks={} "
            "resident_blocks={} capacity_blocks={} retained_payload_blocks={}",
            task.size(), copiedShards,
            copiedShards * std::accumulate(tensorSizes_.begin(), tensorSizes_.end(), size_t{0}),
            publishedBlocks, policy_ == "lru" ? entries_.size() : residentBlocks_, capacityBlocks_,
            payloads_.size());
        return Status::OK();
    }

    using EntryIterator = std::unordered_map<BlockId, LruEntry, Detail::BlockIdHasher>::iterator;

    static Clock::time_point LogicalTime(uint64_t logicalTimeNs)
    {
        return Clock::time_point{
            std::chrono::duration_cast<Clock::duration>(std::chrono::nanoseconds(logicalTimeNs))};
    }

    Expected<std::vector<uint8_t>> LookupAt(const BlockId* blocks, size_t num,
                                            Clock::time_point now)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (realKV_) {
            std::vector<uint8_t> founds;
            const auto seq = ++accessSeq_;
            for (size_t i = 0; i < num; ++i) {
                auto found = KVBlock::Exists(BlockName(blocks[i]));
                if (!found) {
                    UC_ERROR("OnEvict lookup failed block={} status={}", BlockName(blocks[i]),
                             found.Error());
                    return found.Error();
                }
                founds.push_back(found.Value());
                auto node = radixNodes_.find(blocks[i]);
                if (node != radixNodes_.end() && node->second.resident) {
                    TouchRadix(node->second, seq, now);
                }
                auto entry = entries_.find(blocks[i]);
                if (entry != entries_.end()) { TouchLru(entry, now); }
            }
            if (Logger::isEnabledFor(Logger::Level::DEBUG)) {
                const auto hits = std::count(founds.begin(), founds.end(), uint8_t{1});
                UC_DEBUG("OnEvict lookup blocks={} hits={} misses={}", num, hits, num - hits);
            }
            return founds;
        }
        std::vector<uint8_t> founds(num, false);
        std::vector<BlockId> backendBlocks;
        std::vector<size_t> backendIndexes;
        const auto accessSeq = ++accessSeq_;
        for (size_t i = 0; i < num; ++i) {
            if (policy_ == "lru") {
                auto it = entries_.find(blocks[i]);
                if (it != entries_.end()) {
                    founds[i] = true;
                    TouchLru(it, now);
                    continue;
                }
            } else {
                auto it = radixNodes_.find(blocks[i]);
                if (it != radixNodes_.end() && it->second.resident) {
                    founds[i] = true;
                    TouchRadix(it->second, accessSeq, now);
                    continue;
                }
            }
            backendBlocks.push_back(blocks[i]);
            backendIndexes.push_back(i);
        }
        if (!backendBlocks.empty()) {
            auto backendFounds = backend_->Lookup(backendBlocks.data(), backendBlocks.size());
            if (!backendFounds) { return backendFounds.Error(); }
            for (size_t i = 0; i < backendIndexes.size(); ++i) {
                founds[backendIndexes[i]] = backendFounds.Value()[i];
            }
        }
        return founds;
    }

    Expected<Detail::TaskHandle> DumpWithContext(const Detail::TaskDesc& task,
                                                 const Detail::RequestAwareDumpContext& context,
                                                 Clock::time_point now)
    {
        if (policy_ == "lru") {
            auto status = DumpLru(task, now);
            if (status.Failure()) { return status; }
            return NextId();
        }

        ProtectedBlocks protectedBlocks;
        for (const auto& request : context) {
            protectedBlocks.insert(request.requestBlocks.begin(), request.requestBlocks.end());
            ObserveRequestLocked(request.requestBlocks.data(), request.requestBlocks.size(), now);
        }
        for (const auto& request : context) {
            for (const auto& block : request.dumpBlocks) {
                auto status = AdmitRadix(block, protectedBlocks, now);
                if (status.Failure()) { return status; }
            }
        }
        return NextId();
    }

    void TouchLru(EntryIterator it, Clock::time_point now)
    {
        lru_.splice(lru_.begin(), lru_, it->second.position);
        it->second.lastAccessTime = now;
    }

    Status DumpLru(const Detail::TaskDesc& task, Clock::time_point now)
    {
        for (const auto& shard : task) {
            auto it = entries_.find(shard.owner);
            if (it != entries_.end()) {
                TouchLru(it, now);
                continue;
            }
            if (entries_.size() == capacityBlocks_) {
                auto status = EvictLru(now);
                if (status.Failure()) { return status; }
            }
            lru_.push_front(shard.owner);
            entries_[shard.owner] = LruEntry{lru_.begin(), now};
        }
        return Status::OK();
    }

    void ObserveRequestLocked(const BlockId* blocks, size_t num, Clock::time_point now)
    {
        const auto accessSeq = ++accessSeq_;
        for (size_t i = 0; i < num; ++i) {
            auto inserted = radixNodes_.emplace(blocks[i], RadixNode{});
            auto& node = inserted.first->second;
            if (inserted.second) {
                node.block = blocks[i];
                if (i != 0) {
                    node.parent = blocks[i - 1];
                    node.hasParent = true;
                }
            }
            if (node.resident) { TouchRadix(node, accessSeq, now); }
        }
    }

    void TouchRadix(RadixNode& node, uint64_t accessSeq, Clock::time_point now)
    {
        if (node.residentChildCount == 0) {
            leafLru_.erase(LeafKey{node.lastAccessSeq, node.block});
        }
        node.lastAccessSeq = accessSeq;
        node.lastAccessTime = now;
        if (node.residentChildCount == 0) {
            leafLru_.insert(LeafKey{node.lastAccessSeq, node.block});
        }
    }

    Status AdmitRadix(const BlockId& block, const ProtectedBlocks& protectedBlocks,
                      Clock::time_point now)
    {
        auto it = radixNodes_.find(block);
        if (it == radixNodes_.end()) { return Status::InvalidParam("block is not in radix tree"); }
        if (it->second.resident) {
            TouchRadix(it->second, ++accessSeq_, now);
            return Status::OK();
        }
        if (residentBlocks_ == capacityBlocks_) {
            auto status = EvictRadixPath(protectedBlocks, now);
            if (status.Failure()) { return status; }
        }

        auto& node = it->second;
        node.resident = true;
        node.lastAccessSeq = ++accessSeq_;
        node.lastAccessTime = now;
        ++residentBlocks_;
        if (node.hasParent) {
            auto& parent = radixNodes_.find(node.parent)->second;
            if (parent.resident && parent.residentChildCount == 0) {
                leafLru_.erase(LeafKey{parent.lastAccessSeq, parent.block});
            }
            ++parent.residentChildCount;
        }
        if (node.residentChildCount == 0) {
            leafLru_.insert(LeafKey{node.lastAccessSeq, node.block});
        }
        return Status::OK();
    }

    bool Expired(Clock::time_point lastAccessTime, Clock::time_point now) const
    {
        return dumpMaxIdleSeconds_ != 0 &&
               now - lastAccessTime >= std::chrono::seconds(dumpMaxIdleSeconds_);
    }

    Status DumpToBackend(const BlockId& block)
    {
        if (realKV_) {
            Metrics::UpdateStats(NAME_TO_METRIC_ID("on_evict_backend_write_requests_total"), 1.0);
            Metrics::UpdateStats(NAME_TO_METRIC_ID("on_evict_backend_write_bytes_total"),
                                 static_cast<double>(blockSize_));
            return Status::OK();
        }
        Detail::TaskDesc backendTask{
            Detail::Shard{block, 0, {}}
        };
        backendTask.brief = "OnEvictCache2Backend";
        auto submitted = backend_->Dump(std::move(backendTask));
        if (!submitted) { return submitted.Error(); }
        return backend_->Wait(submitted.Value());
    }

    void RecordEviction(size_t blocks, size_t discarded)
    {
        UC_DEBUG(
            "OnEvict eviction policy={} evicted_blocks={} dumped_blocks={} dropped_blocks={} "
            "resident_blocks={} capacity_blocks={}",
            policy_, blocks, blocks - discarded, discarded,
            policy_ == "lru" ? entries_.size() : residentBlocks_, capacityBlocks_);
        Metrics::UpdateStats(NAME_TO_METRIC_ID("on_evict_eviction_paths_total"), 1.0);
        Metrics::UpdateStats(NAME_TO_METRIC_ID("on_evict_evicted_blocks_total"),
                             static_cast<double>(blocks));
        Metrics::UpdateStats(NAME_TO_METRIC_ID("on_evict_discarded_blocks_total"),
                             static_cast<double>(discarded));
        Metrics::UpdateStats(NAME_TO_METRIC_ID("on_evict_discarded_bytes_total"),
                             static_cast<double>(discarded * blockSize_));
    }

    Status EvictLru(Clock::time_point now)
    {
        const auto victim = lru_.back();
        const auto it = entries_.find(victim);
        const auto discarded = Expired(it->second.lastAccessTime, now);
        if (!discarded) {
            auto status = DumpToBackend(victim);
            if (status.Failure()) { return status; }
        }
        if (realKV_ && discarded) { payloads_.erase(victim); }
        entries_.erase(it);
        lru_.pop_back();
        RecordEviction(1, discarded ? 1 : 0);
        return Status::OK();
    }

    Status EvictRadixPath(const ProtectedBlocks& protectedBlocks, Clock::time_point now)
    {
        auto leaf = leafLru_.begin();
        while (leaf != leafLru_.end() && protectedBlocks.count(leaf->second) != 0) { ++leaf; }
        if (leaf == leafLru_.end()) {
            UC_ERROR(
                "OnEvict NoSpace resident_blocks={} capacity_blocks={} leaves={} "
                "protected_blocks={}",
                residentBlocks_, capacityBlocks_, leafLru_.size(), protectedBlocks.size());
            return Status::NoSpace();
        }

        const auto startSeq = leaf->first;
        std::vector<BlockId> victims;
        auto current = leaf->second;
        while (true) {
            victims.push_back(current);
            const auto& node = radixNodes_.find(current)->second;
            if (!node.hasParent) { break; }
            const auto& parent = radixNodes_.find(node.parent)->second;
            if (!parent.resident || parent.residentChildCount >= 2 ||
                protectedBlocks.count(parent.block) != 0 || parent.lastAccessSeq > startSeq) {
                break;
            }
            current = parent.block;
        }

        size_t discarded = 0;
        for (const auto& victim : victims) {
            const auto& node = radixNodes_.find(victim)->second;
            if (Expired(node.lastAccessTime, now)) {
                ++discarded;
                continue;
            }
            auto status = DumpToBackend(victim);
            if (status.Failure()) { return status; }
        }

        for (const auto& victim : victims) {
            auto& node = radixNodes_.find(victim)->second;
            leafLru_.erase(LeafKey{node.lastAccessSeq, node.block});
            node.resident = false;
            if (realKV_ && Expired(node.lastAccessTime, now)) { payloads_.erase(victim); }
            --residentBlocks_;
            if (node.hasParent) { --radixNodes_.find(node.parent)->second.residentChildCount; }
        }
        for (const auto& victim : victims) {
            const auto& node = radixNodes_.find(victim)->second;
            if (!node.hasParent) { continue; }
            auto& parent = radixNodes_.find(node.parent)->second;
            if (parent.resident && parent.residentChildCount == 0) {
                leafLru_.insert(LeafKey{parent.lastAccessSeq, parent.block});
            }
        }
        RecordEviction(victims.size(), discarded);
        return Status::OK();
    }

    static Detail::TaskHandle NextId() noexcept
    {
        static std::atomic<Detail::TaskHandle> id{1};
        return id.fetch_add(1, std::memory_order_relaxed);
    }

    bool realKV_{false};
    int32_t deviceId_{-1};
    size_t shardSize_{0};
    std::vector<size_t> tensorSizes_;
    std::string uniqueId_;
    std::unordered_map<BlockId, std::unique_ptr<KVBlock>, Detail::BlockIdHasher> payloads_;
    std::unique_ptr<TransferQueue> transfers_;
    StoreV1* backend_{nullptr};
    size_t blockSize_{0};
    size_t capacityGb_{0};
    size_t capacityBlocks_{0};
    size_t dumpMaxIdleSeconds_{0};
    std::string policy_{"lru"};
    std::mutex mutex_;
    LruList lru_;
    std::unordered_map<BlockId, LruEntry, Detail::BlockIdHasher> entries_;
    uint64_t accessSeq_{0};
    size_t residentBlocks_{0};
    std::unordered_map<BlockId, RadixNode, Detail::BlockIdHasher> radixNodes_;
    std::set<LeafKey> leafLru_;
};

}  // namespace UC::OnEvictCacheStore

extern "C" UC::StoreV1* MakeOnEvictCacheStore()
{ return new UC::OnEvictCacheStore::OnEvictCacheStore(); }
