/**
 * MIT License
 *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
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
 * */
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <future>
#include <limits>
#include <numeric>
#include <thread>
#include <unordered_map>
#include "buffer_pool.h"
#include "context_index.h"
#include "copy_stream.h"
#include "metrics_api.h"
#include "shared_metadata.h"
#include "template/spsc_ring_queue.h"
#include "trans/cuda/gdr/gdr_config.h"
#include "ucmstore_v1.h"

namespace UC::Context {
class ContextStore final : public StoreV1 {
    using Clock = std::chrono::steady_clock;
    static double Milliseconds(Clock::duration duration)
    { return std::chrono::duration<double, std::milli>(duration).count(); }
    static constexpr size_t none = std::numeric_limits<size_t>::max();
    struct Entry {
        size_t memory = none, references = 0;
        std::vector<bool> shards;
        bool ready = false, busy = false;
    };
    struct Job {
        Detail::TaskDesc desc;
        std::promise<Status> result;
        uint64_t batch = 0;
        bool localComplete = false;
        std::vector<Key> held;
        Clock::time_point h2dDone;
        std::atomic<int32_t> transferFailure{0};
        std::chrono::steady_clock::time_point prepareStarted;
        std::chrono::steady_clock::time_point submitted = std::chrono::steady_clock::now();
    };
    struct Queue {
        std::mutex mutex;
        std::condition_variable wake;
        std::deque<std::shared_ptr<Job>> jobs;
        bool stop = false;
        std::thread thread;
    };
    struct ReadItem {
        std::shared_ptr<Job> job;
        Key key{};
        size_t slot = none;
        size_t row = 0;
        bool ownsReference = false;
        bool backend = false, terminal = false;
        Status status = Status::OK();
    };
    struct ReadQueue {
        SpscRingQueue<ReadItem> items;
        std::atomic_bool stop{false};
        std::thread thread;
    } readQueue_;
    std::mutex mutex_;
    ContextIndex index_;
    std::unordered_map<Key, Entry, Detail::BlockIdHasher> entries_;
    // Only the most recent observation is retained per live request; connector retires it.
    struct Observation {
        uint64_t sequence;
        std::vector<Key> path;
    };
    std::map<std::string, Observation> observed_;
    std::unordered_map<Key, size_t, Detail::BlockIdHasher> contextReferences_;
    BufferPool memory_;
    StoreV1* backend_ = nullptr;
    size_t tpRank_ = 0, tpSize_ = 1;
    SharedMetadata metadata_;
    std::vector<std::unique_ptr<SharedMetadata>> rankMetadata_;
    Queue load_, dump_, completion_;
    std::mutex tasksMutex_;
    std::map<size_t, std::shared_future<Status>> tasks_;
    size_t nextTask_ = 0;
    int deviceId_ = -1;
    size_t blockBytes_ = 0, shardBytes_ = 0, shardCount_ = 0, memoryCount_ = 0;
    size_t runningDepth_ = 524288, localRankSize_ = 8;
    size_t streams_ = 4, queueDepth_ = 8192, limit_ = 64, timeoutMs_ = 30000;
    std::vector<size_t> tensorSizes_;
    double alpha_ = .01;
    int64_t retention_ = -1;
    uint64_t now_ = 0;
    bool useGdr_ = false, aggregation_ = false, sdma_ = false;
    bool sharedMla_ = false, reader_ = false;
    uint64_t layout_ = 1469598103934665603ULL;
    std::unique_ptr<Trans::GdrKVBufferConfig> registrations_;
    std::map<std::string, uint64_t> stats_;

public:
    ~ContextStore() override
    {
        metadata_.Deactivate();
        Stop(load_);
        if (readQueue_.thread.joinable()) {
            if (!readQueue_.stop.load()) { readQueue_.items.Push(ReadItem{}); }
            readQueue_.thread.join();
        }
        Stop(completion_);
        Stop(dump_);
        metadata_.Close();
    }
    std::string Readme() const override { return "ContextStore"; }
    Status Setup(const Detail::Dictionary& config) override;
    Expected<std::vector<uint8_t>> Lookup(const Key* keys, size_t n) override
    {
        if (rankMetadata_.empty()) { return LookupRank(metadata_, keys, n, tpRank_); }
        std::vector<uint8_t> result(n, 1);
        for (size_t r = 0; r < rankMetadata_.size(); ++r) {
            auto found = LookupRank(*rankMetadata_[r], keys, n, r);
            if (!found) { return found.Error(); }
            for (size_t i = 0; i < n; ++i) { result[i] &= found.Value()[i]; }
        }
        return result;
    }
    Expected<ssize_t> LookupOnPrefix(const Key* keys, size_t n) override
    {
        auto found = Lookup(keys, n);
        if (!found) { return found.Error(); }
        for (size_t i = 0; i < n; ++i) {
            if (!found.Value()[i]) { return ssize_t(i) - 1; }
        }
        return ssize_t(n) - 1;
    }
    Expected<ssize_t> LookupOnReverse(const Key* keys, size_t n) override
    {
        auto found = Lookup(keys, n);
        if (!found) { return found.Error(); }
        for (size_t i = n; i; --i) {
            if (found.Value()[i - 1]) { return ssize_t(i) - 1; }
        }
        return -1;
    }
    void Prefetch(const Key*, size_t) override {}
    Expected<size_t> Dump(Detail::TaskDesc task) override
    {
        if (reader_) { return Status::Unsupported(); }
        return Submit(dump_, std::move(task));
    }
    Expected<size_t> Load(Detail::TaskDesc task) override { return Submit(load_, std::move(task)); }
    Expected<bool> Check(size_t task) override
    {
        std::lock_guard<std::mutex> lock(tasksMutex_);
        auto it = tasks_.find(task);
        if (it == tasks_.end()) { return Status::NotFound(); }
        return it->second.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
    }
    Status Wait(size_t task) override
    {
        std::shared_future<Status> result;
        {
            std::lock_guard<std::mutex> lock(tasksMutex_);
            auto it = tasks_.find(task);
            if (it == tasks_.end()) { return Status::NotFound(); }
            result = it->second;
            tasks_.erase(it);
        }
        bool timeout =
            result.wait_for(std::chrono::milliseconds(timeoutMs_)) != std::future_status::ready;
        // A deadline does not revoke device access to the caller's tensor addresses.
        auto status = result.get();
        if (status.Failure()) {
            UC_ERROR("Context load/dump wait failed: rank={}, task={}, status={}", tpRank_, task,
                     status);
            return status;
        }
        return timeout ? Status{Status::Timeout().Underlying(),
                                fmt::format(
                                    "context task wait timeout: rank={}, task={}, timeout_ms={}",
                                    tpRank_, task, timeoutMs_)}
                       : status;
    }
    Status ObserveRequest(const std::string& id, uint64_t observation, uint64_t time,
                          const std::vector<Key>& blocks) override
    {
        if (deviceId_ < 0) { return Status::Unsupported(); }
        if (reader_) { return Status::OK(); }
        auto started = Clock::now();
        std::lock_guard<std::mutex> lock(mutex_);
        auto previous = observed_.find(id);
        if (!blocks.empty() && previous != observed_.end() &&
            previous->second.sequence >= observation) {
            return Status::OK();
        }
        if (!blocks.empty() && time < now_) {
            return Status::InvalidParam("context timestamps must be nondecreasing");
        }
        if (!blocks.empty()) {
            auto status = index_.Observe(blocks, time);
            if (status.Failure()) { return status; }
            for (const auto& key : blocks) { ++contextReferences_[key]; }
            now_ = time;
        }
        std::vector<Key> old;
        if (previous != observed_.end()) {
            old = std::move(previous->second.path);
            observed_.erase(previous);
            for (const auto& key : old) { ReleaseContext(key); }
        }
        if (!blocks.empty()) { observed_[id] = Observation{observation, blocks}; }
        for (auto it = old.rbegin(); it != old.rend(); ++it) { Prune(*it); }
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("context_observe_duration_ms"),
                                 Milliseconds(Clock::now() - started));
        return Status::OK();
    }
    std::map<std::string, uint64_t> ContextStats() override
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto result = stats_;
        if (!reader_) { result["memory_blocks"] = memoryCount_ - memory_.FreeCount(); }
        result["topology_nodes"] = index_.Size();
        return result;
    }

private:
    Status Start(Queue& queue, bool dump);
    static void Stop(Queue& queue)
    {
        {
            std::lock_guard<std::mutex> lock(queue.mutex);
            queue.stop = true;
        }
        queue.wake.notify_all();
        if (queue.thread.joinable()) { queue.thread.join(); }
    }
    Expected<size_t> Submit(Queue& queue, Detail::TaskDesc task);
    Status DumpTask(CopyStream& stream, Detail::TaskDesc& task);
    Status StartReadTransfer();
    void PushRead(ReadItem item);
    void FinishRead(const std::shared_ptr<Job>& job, Status status);
    void PrepareLoad(const std::shared_ptr<Job>& job);
    void Complete(const std::shared_ptr<Job>& job, Status status, bool dump,
                  std::chrono::steady_clock::time_point started);
    Status Evict(std::unique_lock<std::mutex>& lock, const std::set<Key>& protectedKeys);
    Status Publish(const Key& id, const Entry& entry)
    { return metadata_.Publish(id, entry.ready ? 1 : 0, entry.memory); }
    Key BackendKey(Key key, size_t rank) const
    {
        // Keep the canonical prefix hash in the policy; namespace only backend I/O.
        // MLA has a single writer and one backend identity for the whole TP group.
        if (!sharedMla_) {
            for (size_t i = 0; i < sizeof(uint64_t); ++i) {
                key[8 + i] ^= std::byte((uint64_t(rank) >> (8 * i)) & 255);
            }
        }
        return key;
    }
    Expected<std::vector<uint8_t>> LookupRank(SharedMetadata& metadata, const Key* keys, size_t n,
                                              size_t rank)
    {
        auto found = metadata.Lookup(keys, n);
        if (!found) { return found.Error(); }
        std::vector<Key> missing;
        std::vector<size_t> offsets;
        for (size_t i = 0; i < n; ++i) {
            if (!found.Value()[i]) {
                missing.push_back(BackendKey(keys[i], rank));
                offsets.push_back(i);
            }
        }
        if (!missing.empty()) {
            auto stored = backend_->Lookup(missing.data(), missing.size());
            if (!stored) { return stored.Error(); }
            for (size_t i = 0; i < offsets.size(); ++i) {
                found.Value()[offsets[i]] = stored.Value()[i];
            }
        }
        return found;
    }
    void ReleaseContext(const Key& key)
    {
        auto it = contextReferences_.find(key);
        if (--it->second == 0) { contextReferences_.erase(it); }
    }
    void Prune(const Key& id)
    {
        index_.Prune(
            id, [this](const Key& k) { return entries_.count(k) || contextReferences_.count(k); });
    }
};

Status ContextStore::Setup(const Detail::Dictionary& c)
{
    c.Get("store_backend", backend_);
    if (!backend_) { return Status::InvalidParam("ContextStore requires a backend"); }
    std::string name;
    c.Get("unique_id", name);
    if (name.empty() || name.find_first_not_of(
                            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-") !=
                            std::string::npos) {
        return Status::InvalidParam("invalid context unique_id");
    }
    c.Get("share_buffer_enable", sharedMla_);
    c.GetNumber("device_id", deviceId_);
    if (deviceId_ < -1) { return Status::InvalidParam("invalid context device_id"); }
    // Each rank owns its payload. Watchers intersect only availability metadata.
    size_t tpSize = 1, tpRank = 0;
    c.GetNumber("context_tp_size", tpSize);
    c.GetNumber("context_tp_rank", tpRank);
    if (!tpSize || tpRank >= tpSize) {
        return Status::InvalidParam("invalid context TP rank/size");
    }
    if (sharedMla_ && tpSize > 64) {
        return Status::InvalidParam("ContextStore MLA supports at most 64 TP ranks");
    }
    tpRank_ = tpRank;
    tpSize_ = tpSize;
    reader_ = sharedMla_ && tpRank != 0 && deviceId_ >= 0;
    if (sharedMla_) { name += "_mla"; }
    if (deviceId_ == -1) {
        if (sharedMla_) { return metadata_.Setup(name, false); }
        for (size_t rank = 0; rank < tpSize; ++rank) {
            auto watcher = std::make_unique<SharedMetadata>();
            auto status = watcher->Setup(name + "_tp" + std::to_string(rank), false);
            if (status.Failure()) { return status; }
            rankMetadata_.push_back(std::move(watcher));
        }
        return Status::OK();
    }
    if (!sharedMla_) { name += "_tp" + std::to_string(tpRank); }
    c.GetNumber("block_size", blockBytes_);
    c.GetNumber("shard_size", shardBytes_);
    c.GetNumbers("tensor_size_list", tensorSizes_);
    size_t tensorSize = 0;
    c.GetNumber("tensor_size", tensorSize);
    if (tensorSizes_.empty() && tensorSize && shardBytes_ % tensorSize == 0) {
        tensorSizes_.assign(shardBytes_ / tensorSize, tensorSize);
    }
    if (!blockBytes_ || !shardBytes_ || blockBytes_ % shardBytes_ || tensorSizes_.empty() ||
        std::any_of(tensorSizes_.begin(), tensorSizes_.end(), [](size_t s) { return !s; }) ||
        std::accumulate(tensorSizes_.begin(), tensorSizes_.end(), size_t(0)) > shardBytes_) {
        return Status::InvalidParam("invalid context block/shard/tensor layout");
    }
    shardCount_ = blockBytes_ / shardBytes_;
    auto capacity = [&c](const std::string& prefix) -> size_t {
        ssize_t bytes = 0, gb = 0;
        c.GetNumber(prefix + "_capacity_bytes", bytes);
        c.GetNumber(prefix + "_capacity_gb", gb);
        if (bytes < 0 || gb < 0 || (bytes && gb) || uint64_t(gb) > (uint64_t(1) << 32)) {
            return 0;
        }
        return bytes ? size_t(bytes) : size_t(gb) << 30;
    };
    memoryCount_ = capacity("context_memory") / blockBytes_;
    c.Get("context_alpha", alpha_);
    c.GetNumber("context_retention_ns", retention_);
    c.GetNumber("context_max_eviction_blocks", limit_);
    c.GetNumber("cache_stream_number", streams_);
    c.GetNumber("running_queue_depth", runningDepth_);
    c.GetNumber("local_rank_size", localRankSize_);
    c.GetNumber("waiting_queue_depth", queueDepth_);
    c.GetNumber("timeout_ms", timeoutMs_);
    c.Get("use_gdr", useGdr_);
    c.Get("cache_io_aggregation", aggregation_);
    c.Get("cache_sdma_direct", sdma_);
    if (!memoryCount_ || !std::isfinite(alpha_) || alpha_ <= 0 || alpha_ > 1 || !limit_ ||
        !streams_ || streams_ > 32 || runningDepth_ < 2 || !localRankSize_ || !queueDepth_ ||
        !timeoutMs_ || retention_ < -1 || (aggregation_ && sdma_) ||
        (useGdr_ && (aggregation_ || sdma_))) {
        return Status::InvalidParam("invalid context capacity/policy/transfer configuration");
    }
    std::vector<uintptr_t> addresses;
    std::vector<size_t> sizes;
    c.GetNumbers("gpu_kv_buffer_addrs", addresses);
    c.GetNumbers("gpu_kv_buffer_sizes", sizes);
    auto status = Trans::GdrKVBufferConfig::Validate(addresses, sizes);
    if (status.Failure()) { return status; }
    if (!addresses.empty()) {
        registrations_ = std::make_unique<Trans::GdrKVBufferConfig>();
        status = registrations_->Register(addresses, sizes);
        if (status.Failure()) { return status; }
    }
    // Store a layout signature so peers cannot interpret different tensor layouts
    // or capacity settings as the same shared block offsets.
    for (size_t value : {blockBytes_, shardBytes_, memoryCount_}) {
        layout_ = (layout_ ^ uint64_t(value)) * 1099511628211ULL;
    }
    for (auto value : tensorSizes_) { layout_ = (layout_ ^ uint64_t(value)) * 1099511628211ULL; }
    if (sharedMla_) {
        status = memory_.SetupShared(name + "_memory", deviceId_, memoryCount_, blockBytes_,
                                     !reader_, sdma_);
    } else {
        status = memory_.Setup(deviceId_, memoryCount_, blockBytes_, sdma_);
    }
    if (status.Failure()) { return status; }
    status = metadata_.Setup(name, !reader_, 2 * memoryCount_ + 1, layout_,
                             sharedMla_ ? tpSize_ : 1, sharedMla_ && tpSize_ > 1 ? queueDepth_ : 0);
    if (status.Failure()) { return status; }
    status = StartReadTransfer();
    if (status.Failure()) { return status; }
    status = Start(load_, false);
    if (status.Failure()) { return status; }
    return reader_ ? Status::OK() : Start(dump_, true);
}
Status ContextStore::Start(Queue& queue, bool dump)
{
    std::promise<Status> started;
    auto ready = started.get_future();
    queue.thread = std::thread([this, &queue, dump, start = std::move(started)]() mutable {
        CopyStream stream;
        auto status = Status::OK();
        if (dump) {
            status = aggregation_ ? stream.SetupIoAggregation(deviceId_, useGdr_)
                     : sdma_      ? stream.SetupSdmaDirect(deviceId_, useGdr_)
                                  : stream.Setup(deviceId_, streams_, useGdr_);
        }
        start.set_value(status);
        if (status.Failure()) { return; }
        for (;;) {
            std::shared_ptr<Job> job;
            {
                std::unique_lock<std::mutex> lock(queue.mutex);
                queue.wake.wait(lock, [&] { return queue.stop || !queue.jobs.empty(); });
                if (queue.jobs.empty()) { return; }
                job = std::move(queue.jobs.front());
                queue.jobs.pop_front();
            }
            if (!dump) {
                PrepareLoad(job);
                continue;
            }
            auto started = std::chrono::steady_clock::now();
            auto result = DumpTask(stream, job->desc);
            Complete(job, result, true, started);
        }
    });
    return ready.get();
}
void ContextStore::Complete(const std::shared_ptr<Job>& job, Status status, bool dump,
                            std::chrono::steady_clock::time_point started)
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stats_[dump ? "dump_ns" : "load_ns"] +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() -
                                                                 started)
                .count();
        stats_["queue_wait_ns"] +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(started - job->submitted).count();
        ++stats_[dump ? "dump_tasks" : "load_tasks"];
        if (status.Failure()) { ++stats_["failed_tasks"]; }
        if (!reader_) {
            for (const auto& shard : job->desc) { ReleaseContext(shard.owner); }
            for (auto it = job->desc.rbegin(); it != job->desc.rend(); ++it) { Prune(it->owner); }
        }
    }
    job->result.set_value(status);
}
void ContextStore::PushRead(ReadItem item) { readQueue_.items.Push(std::move(item)); }
Status ContextStore::StartReadTransfer()
{
    readQueue_.items.Setup(runningDepth_);
    if (sharedMla_ && tpSize_ > 1 && !reader_) {
        completion_.thread = std::thread([this] {
            for (;;) {
                std::shared_ptr<Job> job;
                {
                    std::unique_lock<std::mutex> lock(completion_.mutex);
                    completion_.wake.wait(
                        lock, [&] { return completion_.stop || !completion_.jobs.empty(); });
                    if (completion_.jobs.empty()) { return; }
                    job = std::move(completion_.jobs.front());
                    completion_.jobs.pop_front();
                    completion_.wake.notify_all();
                }
                auto result = metadata_.WaitReaders(job->batch);
                UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("context_load_mla_completion_wait_ms"),
                                         Milliseconds(Clock::now() - job->h2dDone));
                if (result.Failure()) {
                    UC_ERROR("Context MLA deferred release failed: rank={}, batch={}, status={}",
                             tpRank_, job->batch, result);
                }
                metadata_.FailLoad(job->batch, result);
                FinishRead(job, result);
            }
        });
    }
    std::promise<Status> started;
    auto ready = started.get_future();
    readQueue_.thread = std::thread([this, start = std::move(started)]() mutable {
        CopyStream stream;
        auto status = aggregation_ ? stream.SetupIoAggregation(deviceId_, useGdr_)
                      : sdma_      ? stream.SetupSdmaDirect(deviceId_, useGdr_)
                                   : stream.Setup(deviceId_, streams_, useGdr_);
        start.set_value(status);
        if (status.Failure()) {
            readQueue_.stop.store(true);
            return;
        }
        std::vector<Key> held;
        size_t memoryShards = 0, backendShards = 0;
        Status result = Status::OK();
        readQueue_.items.ConsumerLoop(readQueue_.stop, [&](ReadItem&& item) {
            if (!item.job) {
                readQueue_.stop.store(true);
                return;
            }
            auto& job = item.job;
            if (!item.terminal) {
                bool first = memoryShards + backendShards == 0;
                if (result.Success()) {
                    auto& shard = job->desc[item.row];
                    auto* host = static_cast<char*>(memory_.CopyAddress(item.slot)) +
                                 shard.index * shardBytes_;
                    result = stream.HostToDeviceAsync(host, shard.addrs.data(), tensorSizes_);
                    if (first && result.Success()) {
                        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("context_load_first_h2d_ms"),
                                                 Milliseconds(Clock::now() - job->submitted));
                    }
                }
                if (result.Failure()) {
                    job->transferFailure.store(result.Underlying());
                    if (job->batch) { metadata_.FailLoad(job->batch, result); }
                }
                ++(item.backend ? backendShards : memoryShards);
                if (item.ownsReference) { held.push_back(item.key); }
                return;
            }
            // A failed prepare still sends a terminal marker: drain and release all earlier work.
            if (result.Success()) { result = item.status; }
            auto syncStart = Clock::now();
            auto synced = stream.Synchronize();
            UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("context_load_h2d_sync_ms"),
                                     Milliseconds(Clock::now() - syncStart));
            if (synced.Failure()) { result = synced; }
            job->held = std::move(held);
            if (result.Success()) {
                std::lock_guard<std::mutex> lock(mutex_);
                const auto payload =
                    std::accumulate(tensorSizes_.begin(), tensorSizes_.end(), size_t(0));
                stats_["h2d_bytes"] += payload * job->desc.size();
                stats_["backend_h2d_shards"] += backendShards;
                stats_["memory_load_shards"] += memoryShards;
            }
            if (job->batch) { metadata_.FailLoad(job->batch, result); }
            if (job->batch && !reader_ && result.Success()) {
                job->h2dDone = Clock::now();
                job->localComplete = true;
                Complete(job, result, false, job->prepareStarted);
                // Protect slots until readers finish, without blocking the next layer's H2D.
                std::unique_lock<std::mutex> lock(completion_.mutex);
                completion_.wake.wait(lock, [&] { return completion_.jobs.size() < queueDepth_; });
                completion_.jobs.push_back(job);
                completion_.wake.notify_all();
            } else {
                FinishRead(job, result);
            }
            held.clear();
            memoryShards = backendShards = 0;
            result = Status::OK();
        });
    });
    return ready.get();
}
void ContextStore::FinishRead(const std::shared_ptr<Job>& job, Status status)
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& key : job->held) {
            if (reader_) {
                metadata_.Release(key);
            } else {
                --entries_.at(key).references;
            }
        }
    }
    if (job->batch) { metadata_.EndLoad(job->batch, tpRank_); }
    if (!job->localComplete) { Complete(job, status, false, job->prepareStarted); }
}
Expected<size_t> ContextStore::Submit(Queue& queue, Detail::TaskDesc task)
{
    if (deviceId_ < 0) { return Status::Unsupported(); }
    if (task.empty()) { return Status::InvalidParam("empty context transfer"); }
    for (const auto& shard : task) {
        if (shard.index >= shardCount_ || shard.addrs.size() != tensorSizes_.size() ||
            std::any_of(shard.addrs.begin(), shard.addrs.end(), [](void* p) { return !p; })) {
            return Status::InvalidParam("invalid context transfer shard");
        }
    }
    const bool load = &queue == &load_;
    // Distinct blocks cannot exceed shard rows; avoid per-layer set construction.
    if (load && task.size() > memoryCount_) {
        std::set<Key> blocks;
        for (const auto& shard : task) { blocks.insert(shard.owner); }
        if (blocks.size() > memoryCount_) {
            return Status{Status::NoSpace().Underlying(),
                          "load requires " + std::to_string(blocks.size()) +
                              " blocks; Memory holds " + std::to_string(memoryCount_)};
        }
    }
    auto job = std::make_shared<Job>();
    job->desc = std::move(task);
    if (load && sharedMla_ && tpSize_ > 1) {
        // The signature excludes device addresses and includes every (block, shard).
        uint64_t a = 1469598103934665603ULL, b = 1099511628211ULL;
        for (const auto& shard : job->desc) {
            for (auto byte : shard.owner) {
                a = (a ^ uint8_t(byte)) * 1099511628211ULL;
                b = (b ^ uint8_t(byte)) * 0x9e3779b185ebca87ULL;
            }
            for (size_t i = 0; i < 8; ++i) {
                a = (a ^ ((uint64_t(shard.index) >> (i * 8)) & 255)) * 1099511628211ULL;
                b = (b ^ ((uint64_t(shard.index) >> (i * 8)) & 255)) * 0x9e3779b185ebca87ULL;
            }
        }
        Key signature{};
        for (size_t i = 0; i < 8; ++i) {
            signature[i] = std::byte(a >> (i * 8));
            signature[8 + i] = std::byte(b >> (i * 8));
        }
        auto batch = metadata_.BeginLoad(signature, tpRank_, timeoutMs_);
        if (!batch) { return batch.Error(); }
        job->batch = batch.Value();
    }
    auto reject = [&](Status status) -> Expected<size_t> {
        if (job->batch) {
            metadata_.FailLoad(job->batch, status);
            metadata_.EndLoad(job->batch, tpRank_);
        }
        return status;
    };
    std::lock_guard<std::mutex> lock(queue.mutex);
    if (queue.stop) { return reject(Status::Error("context store stopped")); }
    if (queue.jobs.size() >= queueDepth_) { return reject(Status::NoSpace()); }
    if (!reader_) {
        std::lock_guard<std::mutex> guard(mutex_);
        for (const auto& shard : job->desc) {
            if (!index_.Contains(shard.owner)) {
                return reject(Status::InvalidParam("ObserveRequest required before transfer"));
            }
        }
        for (const auto& shard : job->desc) { ++contextReferences_[shard.owner]; }
    }
    size_t id;
    {
        std::lock_guard<std::mutex> guard(tasksMutex_);
        id = ++nextTask_;
        tasks_[id] = job->result.get_future().share();
    }
    queue.jobs.push_back(std::move(job));
    queue.wake.notify_one();
    return id;
}
Status ContextStore::Evict(std::unique_lock<std::mutex>& lock, const std::set<Key>& protectedKeys)
{
    auto decisionStart = std::chrono::steady_clock::now();
    auto victim =
        index_.Select(alpha_ * memoryCount_, limit_, [this, &protectedKeys](const Key& k) {
            const auto& e = entries_.at(k);
            return e.ready && !e.busy && e.references == 0 && !protectedKeys.count(k) &&
                   (!sharedMla_ || metadata_.Evictable(k));
        });
    stats_["decision_ns"] += std::chrono::duration_cast<std::chrono::nanoseconds>(
                                 std::chrono::steady_clock::now() - decisionStart)
                                 .count();
    ++stats_["eviction_decisions"];
    if (victim.blocks.empty()) {
        ++stats_["no_space"];
        return Status::NoSpace();
    }
    bool drop = retention_ >= 0 && now_ - victim.lastAccess > uint64_t(retention_);
    if (sharedMla_ && !metadata_.ReserveEviction(victim.blocks)) {
        ++stats_["no_space"];
        return Status::NoSpace();
    }
    for (const auto& k : victim.blocks) { entries_.at(k).busy = true; }
    for (const auto& k : victim.blocks) {
        auto& e = entries_.at(k);
        if (!drop) {
            auto backendKey = BackendKey(k, tpRank_);
            lock.unlock();
            auto exists = backend_->Lookup(&backendKey, 1);
            auto status = exists ? Status::OK() : exists.Error();
            bool written = false;
            if (exists && !exists.Value()[0]) {
                Detail::TaskDesc desc;
                for (size_t i = 0; i < shardCount_; ++i) {
                    desc.push_back(Detail::Shard{
                        backendKey,
                        i,
                        {static_cast<char*>(memory_.Data(e.memory)) + i * shardBytes_}});
                }
                auto task = backend_->Dump(std::move(desc));
                status = task ? backend_->Wait(task.Value()) : task.Error();
                written = status.Success();
            }
            lock.lock();
            if (status.Failure()) {
                // Keep every remaining victim readable and release MLA reservations.
                for (const auto& pending : victim.blocks) {
                    auto it = entries_.find(pending);
                    if (it != entries_.end() && it->second.busy) {
                        it->second.busy = false;
                        Publish(pending, it->second);
                    }
                }
                return status;
            }
            if (written) {
                ++stats_["backend_dump_blocks"];
                stats_["backend_dump_bytes"] += blockBytes_;
            } else {
                ++stats_["backend_dump_skipped_blocks"];
            }
        }
        e.ready = false;
        auto status = Publish(k, e);
        if (status.Failure()) { return status; }
        memory_.Release(e.memory);
        e.memory = none;
        e.shards.clear();
        e.busy = false;
        index_.Remove(k);
        ++stats_[drop ? "drop_blocks" : "dump_blocks"];
        ++stats_["evicted_blocks"];
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("context_evict_blocks_total"), 1.0);
        if (drop) {
            UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("context_drop_blocks_total"), 1.0);
        } else {
            UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("context_dump_blocks_total"), 1.0);
        }
        entries_.erase(k);
    }
    for (auto it = victim.blocks.rbegin(); it != victim.blocks.rend(); ++it) { Prune(*it); }
    return Status::OK();
}
void ContextStore::PrepareLoad(const std::shared_ptr<Job>& job)
{
    job->prepareStarted = Clock::now();
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("context_load_prepare_queue_wait_ms"),
                             Milliseconds(job->prepareStarted - job->submitted));
    double mlaWait = 0;
    std::vector<size_t> order;
    order.reserve(job->desc.size());
    for (size_t r = 0; r < localRankSize_; ++r) {
        size_t slice = (size_t(deviceId_) + r) % localRankSize_;
        for (size_t i = slice; i < job->desc.size(); i += localRankSize_) { order.push_back(i); }
    }
    std::vector<std::pair<Key, size_t>> rows;
    std::set<Key> protectedKeys;
    for (auto i : order) {
        if (protectedKeys.insert(job->desc[i].owner).second) {
            rows.emplace_back(job->desc[i].owner, i);
        }
    }
    std::unordered_map<Key, ReadItem, Detail::BlockIdHasher> prepared;
    size_t nextRow = 0;
    auto publish = [&](ReadItem item) {
        prepared.emplace(item.key, std::move(item));
        while (nextRow < order.size()) {
            auto row = order[nextRow];
            auto found = prepared.find(job->desc[row].owner);
            if (found == prepared.end()) { break; }
            auto shard = found->second;
            shard.row = row;
            found->second.ownsReference = false;
            PushRead(std::move(shard));
            ++nextRow;
        }
    };
    auto status = Status::OK();
    std::map<Key, bool> backendReady;
    struct Pending {
        ReadItem item;
        bool allocated;
        size_t fetched;
    };
    std::vector<Pending> pending;
    Detail::TaskDesc fetch;
    // Small batches amortize backend calls without waiting for the whole request.
    constexpr size_t admissionBatch = 32;
    pending.reserve(admissionBatch);
    auto flush = [&]() {
        if (pending.empty()) { return; }
        auto task = backend_->Load(std::move(fetch));
        auto loaded = task ? backend_->Wait(task.Value()) : task.Error();
        for (auto& p : pending) {
            auto result = loaded;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                auto& e = entries_.at(p.item.key);
                e.busy = false;
                if (result.Success()) {
                    e.ready = true;
                    result = Publish(p.item.key, e);
                }
                if (result.Success()) {
                    e.shards.assign(shardCount_, true);
                    ++stats_["backend_load_blocks"];
                    stats_["backend_load_shards"] += p.fetched;
                } else {
                    e.ready = false;
                    --e.references;
                    if (p.allocated) {
                        memory_.Release(e.memory);
                        index_.Remove(p.item.key);
                        entries_.erase(p.item.key);
                    }
                }
            }
            if (result.Success()) {
                publish(std::move(p.item));
            } else if (status.Success()) {
                status = result;
            }
        }
        pending.clear();
        fetch = Detail::TaskDesc{};
    };
    for (auto row = rows.begin(); row != rows.end(); ++row) {
        auto& [key, requested] = *row;
        if (status.Failure()) { break; }
        if (auto failure = job->transferFailure.load()) {
            status = Status{failure, "context H2D submission failed"};
            break;
        }
        if (reader_) {
            auto waitStart = Clock::now();
            auto location = metadata_.WaitAcquire(key, layout_, job->batch);
            mlaWait += Milliseconds(Clock::now() - waitStart);
            if (!location) {
                status = location.Error();
                break;
            }
            publish(ReadItem{job, key, location.Value().slot, requested, true});
            continue;
        }
        std::unique_lock<std::mutex> lock(mutex_);
        auto it = entries_.find(key);
        if ((it == entries_.end() || !it->second.ready) && !backendReady.count(key)) {
            // Only build the backend lookup on the first miss. Resident layers avoid
            // a second scan of entries_; remaining keys are checked in one backend call.
            lock.unlock();
            std::vector<Key> backendKeys;
            for (auto next = row; next != rows.end(); ++next) {
                backendKeys.push_back(BackendKey(next->first, tpRank_));
            }
            auto found = backend_->Lookup(backendKeys.data(), backendKeys.size());
            if (!found) {
                status = found.Error();
                break;
            }
            size_t i = 0;
            for (auto next = row; next != rows.end(); ++next) {
                backendReady.emplace(next->first, found.Value()[i++]);
            }
            lock.lock();
            it = entries_.find(key);
        }
        if (it == entries_.end() || !it->second.ready) {
            const bool allocate = it == entries_.end();
            if (!allocate && (it->second.busy || it->second.references)) {
                status = Status::NotFound();
                break;
            }
            auto backendKey = BackendKey(key, tpRank_);
            if (!backendReady.at(key)) {
                // Fake membership may have appeared after the batched lookup,
                // when another task evicted this previously resident block.
                auto found = backend_->Lookup(&backendKey, 1);
                if (!found || !found.Value()[0]) {
                    status = found ? Status::NotFound() : found.Error();
                    break;
                }
            }
            if (allocate && memory_.Empty()) {
                status = Evict(lock, protectedKeys);
                if (status.Failure()) { break; }
            }
            auto& e = entries_[key];
            if (allocate) {
                e.memory = memory_.Allocate();
                e.shards.assign(shardCount_, false);
                index_.Insert(key);
            }
            e.busy = true;
            ++e.references;
            if (pending.empty()) {
                fetch.reserve(std::min(admissionBatch, rows.size()) * shardCount_);
            }
            auto before = fetch.size();
            for (size_t i = 0; i < shardCount_; ++i) {
                if (!e.shards[i]) {
                    fetch.push_back(
                        {backendKey,
                         i,
                         {static_cast<char*>(memory_.Data(e.memory)) + i * shardBytes_}});
                }
            }
            pending.push_back({
                ReadItem{job, key, e.memory, requested, true, true},
                allocate,
                fetch.size() - before
            });
            stats_["memory_peak_blocks"] = std::max(stats_["memory_peak_blocks"],
                                                    uint64_t(memoryCount_ - memory_.FreeCount()));
            lock.unlock();
            if (pending.size() == admissionBatch) { flush(); }
        } else {
            if (it->second.busy) {
                status = Status::NotFound();
                break;
            }
            ++it->second.references;
            auto slot = it->second.memory;
            lock.unlock();
            publish(ReadItem{job, key, slot, requested, true});
        }
    }
    // Even a later preparation error must complete and release earlier reservations.
    flush();
    for (auto& [key, item] : prepared) {
        if (!item.ownsReference) { continue; }
        // A preceding failed block may have prevented this ready block from submitting.
        std::lock_guard<std::mutex> lock(mutex_);
        if (reader_) {
            metadata_.Release(key);
        } else {
            --entries_.at(key).references;
        }
    }
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("context_load_prepare_duration_ms"),
                             Milliseconds(Clock::now() - job->prepareStarted));
    if (reader_) {
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("context_load_mla_ready_wait_ms"), mlaWait);
    }
    if (job->batch) { metadata_.FailLoad(job->batch, status); }
    ReadItem terminal;
    terminal.job = job;
    terminal.terminal = true;
    terminal.status = status;
    PushRead(std::move(terminal));
}
Status ContextStore::DumpTask(CopyStream& stream, Detail::TaskDesc& task)
{
    struct Held {
        Key key;
        size_t slot;
    };
    std::map<Key, Held> held;
    std::vector<size_t> copied;
    std::set<Key> protectedKeys;
    for (const auto& shard : task) { protectedKeys.insert(shard.owner); }
    auto status = Status::OK();
    std::unique_lock<std::mutex> lock(mutex_);
    // Validate topology before any capacity-changing operation.
    for (const auto& shard : task) {
        if (!index_.Contains(shard.owner)) {
            return Status::InvalidParam("ObserveRequest required before transfer");
        }
    }
    for (const auto& shard : task) {
        if (held.count(shard.owner)) { continue; }
        auto it = entries_.find(shard.owner);
        if (it == entries_.end()) {
            if (memory_.Empty()) {
                status = Evict(lock, protectedKeys);
                if (status.Failure()) { break; }
            }
            auto& entry = entries_[shard.owner];
            entry.memory = memory_.Allocate();
            stats_["memory_peak_blocks"] = std::max(stats_["memory_peak_blocks"],
                                                    uint64_t(memoryCount_ - memory_.FreeCount()));
            entry.shards.assign(shardCount_, false);
            index_.Insert(shard.owner);
            it = entries_.find(shard.owner);
        }
        if (it == entries_.end() || it->second.busy) {
            status = Status::NotFound();
            break;
        }
        auto& e = it->second;
        ++e.references;
        held.emplace(shard.owner, Held{shard.owner, e.memory});
    }
    if (status.Success()) {
        lock.unlock();
        if (task.prerequisiteHandle) {
            status = stream.WaitEvent(Trans::Event{task.prerequisiteHandle});
        }
        std::set<std::pair<Key, size_t>> submitted;
        for (size_t i = 0; status.Success() && i < task.size(); ++i) {
            auto& shard = task[i];
            lock.lock();
            bool ready = entries_.at(shard.owner).shards[shard.index];
            lock.unlock();
            if (ready || !submitted.insert({shard.owner, shard.index}).second) { continue; }
            const auto& h = held.at(shard.owner);
            auto* host =
                static_cast<char*>(memory_.CopyAddress(h.slot)) + shard.index * shardBytes_;
            status = stream.DeviceToHostAsync(shard.addrs.data(), host, tensorSizes_);
            if (status.Success()) { copied.push_back(i); }
        }
        // Drain even on a submission error before releasing any addresses.
        auto synced = stream.Synchronize();
        if (synced.Failure()) { status = synced; }
        lock.lock();
    }
    const size_t payload = std::accumulate(tensorSizes_.begin(), tensorSizes_.end(), size_t(0));
    if (status.Success()) {
        for (auto i : copied) {
            const auto& shard = task[i];
            entries_.at(shard.owner).shards[shard.index] = true;
            stats_["d2h_bytes"] += payload;
        }
    }
    for (const auto& pair : held) {
        const auto& key = pair.first;
        auto& e = entries_.at(key);
        --e.references;
        if (status.Success()) {
            e.ready = std::all_of(e.shards.begin(), e.shards.end(), [](bool b) { return b; });
            auto published = Publish(key, e);
            if (published.Failure()) { status = published; }
        }
        if (status.Failure() && !e.ready) {
            memory_.Release(e.memory);
            e.memory = none;
            e.shards.clear();
            index_.Remove(key);
            entries_.erase(key);
        }
    }
    stats_["memory_peak_blocks"] =
        std::max(stats_["memory_peak_blocks"], uint64_t(memoryCount_ - memory_.FreeCount()));
    return status;
}
}  // namespace UC::Context
extern "C" UC::StoreV1* MakeContextStore() { return new UC::Context::ContextStore; }
