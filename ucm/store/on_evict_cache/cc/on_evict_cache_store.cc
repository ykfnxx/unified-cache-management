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
#include <list>
#include <mutex>
#include <string>
#include <unordered_map>
#include "ucmstore_v1.h"

namespace UC::OnEvictCacheStore {

class OnEvictCacheStore final : public StoreV1 {
    using BlockId = Detail::BlockId;
    using LruList = std::list<BlockId>;

public:
    Status Setup(const Detail::Dictionary& config) override
    {
        config.Get("store_backend", backend_);
        config.GetNumber("block_size", blockSize_);
        config.GetNumber("on_evict_cache_capacity_gb", capacityGb_);
        config.Get("on_evict_cache_policy", policy_);

        if (backend_ == nullptr) { return Status::InvalidParam("invalid store backend"); }
        if (blockSize_ == 0) { return Status::InvalidParam("invalid block size"); }
        if (capacityGb_ == 0) { return Status::InvalidParam("invalid cache capacity"); }
        if (policy_ != "lru") { return Status::InvalidParam("unsupported eviction policy"); }

        capacityBlocks_ = (capacityGb_ << 30) / blockSize_;
        if (capacityBlocks_ == 0) { return Status::InvalidParam("cache capacity is too small"); }
        return Status::OK();
    }

    std::string Readme() const override { return "OnEvictCacheStore"; }

    Expected<std::vector<uint8_t>> Lookup(const BlockId* blocks, size_t num) override
    {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<uint8_t> founds(num, false);
        std::vector<BlockId> backendBlocks;
        std::vector<size_t> backendIndexes;
        for (size_t i = 0; i < num; ++i) {
            auto it = entries_.find(blocks[i]);
            if (it != entries_.end()) {
                founds[i] = true;
                Touch(it);
            } else {
                backendBlocks.push_back(blocks[i]);
                backendIndexes.push_back(i);
            }
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

    Expected<Detail::TaskHandle> Load(Detail::TaskDesc) override { return NextId(); }

    Expected<Detail::TaskHandle> Dump(Detail::TaskDesc task) override
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& shard : task) {
            auto it = entries_.find(shard.owner);
            if (it != entries_.end()) {
                Touch(it);
                continue;
            }
            if (entries_.size() == capacityBlocks_) {
                auto status = EvictOne();
                if (status.Failure()) { return status; }
            }
            lru_.push_front(shard.owner);
            entries_[shard.owner] = lru_.begin();
        }
        return NextId();
    }

    Expected<bool> Check(Detail::TaskHandle) override { return true; }

    Status Wait(Detail::TaskHandle) override { return Status::OK(); }

private:
    using EntryIterator =
        std::unordered_map<BlockId, LruList::iterator, Detail::BlockIdHasher>::iterator;

    void Touch(EntryIterator it) { lru_.splice(lru_.begin(), lru_, it->second); }

    Status EvictOne()
    {
        const auto victim = lru_.back();
        Detail::TaskDesc backendTask{
            Detail::Shard{victim, 0, {}}
        };
        backendTask.brief = "OnEvictCache2Backend";
        auto submitted = backend_->Dump(std::move(backendTask));
        if (!submitted) { return submitted.Error(); }
        auto status = backend_->Wait(submitted.Value());
        if (status.Failure()) { return status; }
        entries_.erase(victim);
        lru_.pop_back();
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
    std::string policy_{"lru"};
    std::mutex mutex_;
    LruList lru_;
    std::unordered_map<BlockId, LruList::iterator, Detail::BlockIdHasher> entries_;
};

}  // namespace UC::OnEvictCacheStore

extern "C" UC::StoreV1* MakeOnEvictCacheStore()
{ return new UC::OnEvictCacheStore::OnEvictCacheStore(); }
