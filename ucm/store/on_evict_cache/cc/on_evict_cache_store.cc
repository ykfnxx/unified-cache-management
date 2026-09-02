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
#include <atomic>
#include <chrono>
#include <list>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include "metrics_api.h"
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
    Status Setup(const Detail::Dictionary& config) override
    {
        config.Get("store_backend", backend_);
        config.GetNumber("block_size", blockSize_);
        config.GetNumber("on_evict_cache_capacity_gb", capacityGb_);
        config.Get("on_evict_cache_policy", policy_);
        config.GetNumber("on_evict_cache_dump_max_idle_s", dumpMaxIdleSeconds_);

        if (backend_ == nullptr) { return Status::InvalidParam("invalid store backend"); }
        if (blockSize_ == 0) { return Status::InvalidParam("invalid block size"); }
        if (capacityGb_ == 0) { return Status::InvalidParam("invalid cache capacity"); }
        if (policy_ != "lru" && policy_ != "radix_lru") {
            return Status::InvalidParam("unsupported eviction policy");
        }

        capacityBlocks_ = (capacityGb_ << 30) / blockSize_;
        if (capacityBlocks_ == 0) { return Status::InvalidParam("cache capacity is too small"); }
        return Status::OK();
    }

    std::string Readme() const override { return "OnEvictCacheStore"; }

    Expected<std::vector<uint8_t>> Lookup(const BlockId* blocks, size_t num) override
    {
        return LookupAt(blocks, num, Clock::now());
    }

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

    Expected<Detail::TaskHandle> Load(Detail::TaskDesc) override { return NextId(); }

    Expected<Detail::TaskHandle> Dump(Detail::TaskDesc task) override
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (policy_ == "radix_lru") {
            return Status::InvalidParam("radix_lru requires request context");
        }
        auto status = DumpLru(task, Clock::now());
        if (status.Failure()) { return status; }
        return NextId();
    }

    Expected<Detail::TaskHandle> Dump(Detail::TaskDesc task,
                                      const Detail::RequestAwareDumpContext& context) override
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return DumpWithContext(task, context, Clock::now());
    }

    Expected<Detail::TaskHandle> Dump(Detail::TaskDesc task,
                                      const Detail::RequestAwareDumpContext& context,
                                      uint64_t logicalTimeNs) override
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return DumpWithContext(task, context, LogicalTime(logicalTimeNs));
    }

    Expected<bool> Check(Detail::TaskHandle) override { return true; }

    Status Wait(Detail::TaskHandle) override { return Status::OK(); }

private:
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

    Expected<Detail::TaskHandle> DumpWithContext(
        const Detail::TaskDesc& task, const Detail::RequestAwareDumpContext& context,
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
        entries_.erase(it);
        lru_.pop_back();
        RecordEviction(1, discarded ? 1 : 0);
        return Status::OK();
    }

    Status EvictRadixPath(const ProtectedBlocks& protectedBlocks, Clock::time_point now)
    {
        auto leaf = leafLru_.begin();
        while (leaf != leafLru_.end() && protectedBlocks.count(leaf->second) != 0) { ++leaf; }
        if (leaf == leafLru_.end()) { return Status::NoSpace(); }

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
