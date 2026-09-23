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
#include <cmath>
#include <numeric>
#include "context_index.h"
#include "trans/cuda/gdr/gdr_config.h"
#include "trans_manager.h"

namespace UC::Context {
class ContextStore : public StoreV1 {
    Config config_;
    TransBuffer buffer_;
    std::vector<std::unique_ptr<TransBuffer>> watchers_;
    std::unique_ptr<Trans::GdrKVBufferConfig> registrations_;
    TransManager transfers_;

public:
    Status Setup(const Detail::Dictionary& c) override
    {
        auto& p = config_;
        c.Get("store_backend", p.storeBackend);
        c.Get("unique_id", p.uniqueId);
        c.GetNumber("device_id", p.deviceId);
        c.Get("share_buffer_enable", p.shareBufferEnable);
        c.GetNumber("context_tp_size", p.tpSize);
        c.GetNumber("context_tp_rank", p.tpRank);
        if (!p.storeBackend || p.uniqueId.empty() ||
            p.uniqueId.find_first_not_of(
                "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-") !=
                std::string::npos ||
            !p.tpSize || p.tpRank >= p.tpSize || p.deviceId < -1) {
            return Status::InvalidParam("invalid context identity/backend");
        }
        if (p.deviceId < 0) {
            for (size_t rank = 0; rank < (p.shareBufferEnable ? 1 : p.tpSize); ++rank) {
                auto watcher = std::make_unique<TransBuffer>();
                auto cfg = p;
                cfg.tpRank = rank;
                auto s = watcher->Setup(cfg);
                if (s.Failure()) { return s; }
                watchers_.push_back(std::move(watcher));
            }
            return Status::OK();
        }
        c.GetNumber("block_size", p.blockSize);
        c.GetNumber("shard_size", p.shardSize);
        c.GetNumbers("tensor_size_list", p.tensorSizes);
        size_t tensor = 0;
        c.GetNumber("tensor_size", tensor);
        if (p.tensorSizes.empty() && tensor && p.shardSize % tensor == 0) {
            p.tensorSizes.assign(p.shardSize / tensor, tensor);
        }
        ssize_t bytes = 0, gb = 0;
        c.GetNumber("context_memory_capacity_bytes", bytes);
        c.GetNumber("context_memory_capacity_gb", gb);
        if (bytes < 0 || gb < 0 || (bytes && gb) || uint64_t(gb) > (uint64_t(1) << 32)) {
            return Status::InvalidParam("invalid context capacity");
        }
        p.bufferCapacity = bytes ? size_t(bytes) : size_t(gb) << 30;
        c.Get("context_alpha", p.alpha);
        c.GetNumber("context_retention_ns", p.retention);
        c.GetNumber("context_max_eviction_blocks", p.evictionLimit);
        c.GetNumber("cache_stream_number", p.streamNumber);
        c.GetNumber("waiting_queue_depth", p.waitingQueueDepth);
        c.GetNumber("running_queue_depth", p.runningQueueDepth);
        c.GetNumber("local_rank_size", p.localRankSize);
        c.GetNumber("timeout_ms", p.timeoutMs);
        c.Get("cache_sdma_direct", p.cacheSdmaDirect);
        c.Get("cache_io_aggregation", p.cacheIOAggregation);
        c.Get("use_gdr", p.useGdr);
        c.GetNumbers("gpu_kv_buffer_addrs", p.gpuKvBufferAddrs);
        c.GetNumbers("gpu_kv_buffer_sizes", p.gpuKvBufferSizes);
        c.Get("cpu_affinity_cores", p.cpuAffinityCores);
        if (!p.blockSize || !p.shardSize || p.blockSize % p.shardSize ||
            p.bufferCapacity < p.blockSize || p.tensorSizes.empty() ||
            std::any_of(
                p.tensorSizes.begin(), p.tensorSizes.end(), [](size_t n) { return !n; }) ||
            std::accumulate(p.tensorSizes.begin(), p.tensorSizes.end(), size_t(0)) > p.shardSize) {
            return Status::InvalidParam("invalid context block/shard/tensor layout");
        }
        if (!std::isfinite(p.alpha) || p.alpha <= 0 || p.alpha > 1 || p.retention < -1 ||
            !p.evictionLimit || !p.timeoutMs || !p.localRankSize || p.streamNumber < 1 ||
            p.streamNumber > 32 || p.waitingQueueDepth < 2 || p.runningQueueDepth < 2 ||
            (p.cacheSdmaDirect && p.cacheIOAggregation) ||
            (p.useGdr && (p.cacheSdmaDirect || p.cacheIOAggregation))) {
            return Status::InvalidParam("invalid context policy/transfer configuration");
        }
        auto s = Trans::GdrKVBufferConfig::Validate(p.gpuKvBufferAddrs, p.gpuKvBufferSizes);
        if (s.Failure()) { return s; }
        if (!p.gpuKvBufferAddrs.empty()) {
            registrations_ = std::make_unique<Trans::GdrKVBufferConfig>();
            s = registrations_->Register(p.gpuKvBufferAddrs, p.gpuKvBufferSizes);
            if (s.Failure()) { return s; }
        }
        s = buffer_.Setup(p);
        if (s.Failure()) { return s; }
        return transfers_.Setup(p, &buffer_);
    }
    std::string Readme() const override { return "ContextStore"; }
    Expected<std::vector<uint8_t>> Lookup(const Detail::BlockId* keys, size_t n) override
    {
        if (config_.deviceId >= 0) { return buffer_.Lookup(keys, n); }
        std::vector<uint8_t> result(n, 1);
        for (auto& watcher : watchers_) {
            auto found = watcher->Lookup(keys, n);
            if (!found) { return found.Error(); }
            for (size_t i = 0; i < n; ++i) { result[i] &= found.Value()[i]; }
        }
        return result;
    }
    Expected<ssize_t> LookupOnPrefix(const Detail::BlockId* keys, size_t n) override
    {
        auto result = Lookup(keys, n);
        if (!result) { return result.Error(); }
        size_t i = 0;
        while (i < n && result.Value()[i]) { ++i; }
        return ssize_t(i) - 1;
    }
    Expected<ssize_t> LookupOnReverse(const Detail::BlockId* keys, size_t n) override
    {
        auto result = Lookup(keys, n);
        if (!result) { return result.Error(); }
        for (size_t i = n; i > 0; --i) {
            if (result.Value()[i - 1]) { return ssize_t(i - 1); }
        }
        return ssize_t(-1);
    }
    void Prefetch(const Detail::BlockId* keys, size_t n) override
    { config_.storeBackend->Prefetch(keys, n); }
    Status ObserveRequest(const std::string& id, uint64_t observation, uint64_t time,
                          const std::vector<Key>& path) override
    {
        return config_.deviceId < 0 ? Status::Unsupported()
                                    : buffer_.Observe(id, observation, time, path);
    }
    Expected<Detail::TaskHandle> Load(Detail::TaskDesc task) override
    { return Submit(TransTask::Type::LOAD, std::move(task)); }
    Expected<Detail::TaskHandle> Dump(Detail::TaskDesc task) override
    {
        if (config_.shareBufferEnable && config_.tpRank != 0) { return Status::Unsupported(); }
        return Submit(TransTask::Type::DUMP, std::move(task));
    }
    Expected<bool> Check(Detail::TaskHandle task) override { return transfers_.Check(task); }
    Status Wait(Detail::TaskHandle task) override { return transfers_.Wait(task); }
    std::map<std::string, uint64_t> ContextStats() override { return buffer_.Stats(); }

private:
    Expected<Detail::TaskHandle> Submit(TransTask::Type type, Detail::TaskDesc task)
    {
        if (config_.deviceId < 0) { return Status::Unsupported(); }
        auto s = buffer_.BeginTask(task);
        if (s.Failure()) { return s; }
        auto result = transfers_.Submit({type, std::move(task)});
        if (!result) { buffer_.EndTask(); }
        return result;
    }
};
}  // namespace UC::Context
extern "C" UC::StoreV1* MakeContextStore() { return new UC::Context::ContextStore(); }
