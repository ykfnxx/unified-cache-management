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
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <gtest/gtest.h>
#include <memory>
#include <mutex>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include "load_queue.h"
#include "trans/device.h"
#include "trans_buffer.h"
#include "ucmstore_v1.h"

extern "C" UC::StoreV1* MakeMemoryStore();

namespace {
UC::Detail::BlockId MakeBlockId(uint8_t seed)
{
    UC::Detail::BlockId block{};
    for (size_t i = 0; i < block.size(); ++i) {
        block[i] = std::byte{static_cast<uint8_t>(seed + i)};
    }
    return block;
}

class BlockingLoadBackend : public UC::StoreV1 {
    struct PendingTask {
        UC::Detail::TaskDesc desc;
    };

public:
    explicit BlockingLoadBackend(std::array<std::byte, 8> payload) : payload_(payload) {}

    UC::Status Setup(const UC::Detail::Dictionary&) override { return UC::Status::OK(); }
    std::string Readme() const override { return "BlockingLoadBackend"; }
    UC::Expected<std::vector<uint8_t>> Lookup(const UC::Detail::BlockId*, size_t num) override
    {
        return std::vector<uint8_t>(num, false);
    }
    UC::Expected<ssize_t> LookupOnPrefix(const UC::Detail::BlockId*, size_t) override { return -1; }
    void Prefetch(const UC::Detail::BlockId*, size_t) override {}
    UC::Expected<UC::Detail::TaskHandle> Load(UC::Detail::TaskDesc task) override
    {
        auto id = nextId_.fetch_add(1, std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> guard(mutex_);
            pending_.emplace(id, PendingTask{std::move(task)});
        }
        loadCalls_.fetch_add(1, std::memory_order_relaxed);
        cv_.notify_all();
        return id;
    }
    UC::Expected<UC::Detail::TaskHandle> Dump(UC::Detail::TaskDesc) override
    {
        return UC::Status::Unsupported();
    }
    UC::Expected<bool> Check(UC::Detail::TaskHandle) override { return true; }
    UC::Status Wait(UC::Detail::TaskHandle taskId) override
    {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return releaseWait_; });
        auto iter = pending_.find(taskId);
        if (iter == pending_.end()) { return UC::Status::NotFound(); }
        auto& task = iter->second.desc;
        if (task.empty() || task.front().addrs.size() != 1 || !task.front().addrs.front()) {
            return UC::Status::InvalidParam("invalid backend task");
        }
        std::memcpy(task.front().addrs.front(), payload_.data(), payload_.size());
        pending_.erase(iter);
        return UC::Status::OK();
    }

    bool WaitUntilLoads(size_t expected, size_t timeoutMs)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        return cv_.wait_for(lock, std::chrono::milliseconds(timeoutMs), [this, expected] {
            return loadCalls_.load(std::memory_order_relaxed) >= expected;
        });
    }

    void ReleaseWaiters()
    {
        {
            std::lock_guard<std::mutex> guard(mutex_);
            releaseWait_ = true;
        }
        cv_.notify_all();
    }

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    std::unordered_map<UC::Detail::TaskHandle, PendingTask> pending_;
    std::atomic<UC::Detail::TaskHandle> nextId_{1};
    std::atomic<size_t> loadCalls_{0};
    bool releaseWait_{false};
    std::array<std::byte, 8> payload_{};
};
}  // namespace

TEST(UCMTokenLayerApiTest, DefaultStoreV1TokenLayerMethodsAreUnsupported)
{
    class MinimalStore : public UC::StoreV1 {
    public:
        UC::Status Setup(const UC::Detail::Dictionary&) override { return UC::Status::OK(); }
        std::string Readme() const override { return "MinimalStore"; }
        UC::Expected<std::vector<uint8_t>> Lookup(const UC::Detail::BlockId*, size_t num) override
        {
            return std::vector<uint8_t>(num, false);
        }
        UC::Expected<ssize_t> LookupOnPrefix(const UC::Detail::BlockId*, size_t) override
        {
            return -1;
        }
        void Prefetch(const UC::Detail::BlockId*, size_t) override {}
        UC::Expected<UC::Detail::TaskHandle> Load(UC::Detail::TaskDesc) override { return 1; }
        UC::Expected<UC::Detail::TaskHandle> Dump(UC::Detail::TaskDesc) override { return 2; }
        UC::Expected<bool> Check(UC::Detail::TaskHandle) override { return true; }
        UC::Status Wait(UC::Detail::TaskHandle) override { return UC::Status::OK(); }
    };

    MinimalStore store;
    UC::Detail::TokenLayerTaskDesc task;
    auto lookup = store.LookupTokens(task);
    auto load = store.LoadTokens(task);
    auto dump = store.DumpTokens(task);
    EXPECT_FALSE(lookup.HasValue());
    EXPECT_FALSE(load.HasValue());
    EXPECT_FALSE(dump.HasValue());
}

TEST(UCMMemoryStoreTest, TokenDumpLoadRoundTripUsesTensorType)
{
    auto store = std::unique_ptr<UC::StoreV1>(MakeMemoryStore());
    UC::Detail::Dictionary config;
    config.Set<UC::StoreV1*>("store_backend", nullptr);
    config.SetNumber("device_id", 0);
    config.SetNumber("shard_size", 16);
    config.SetNumber("block_size", 16);
    config.SetNumber("memory_token_chunk_size", 2);
    config.SetNumber("memory_buffer_capacity_gb", 1);
    config.Set("memory_required_tensor_types", std::vector<ssize_t>{0, 1});
    config.Set("memory_tensor_size_by_type_0", std::vector<ssize_t>{4});
    config.Set("memory_tensor_size_by_type_1", std::vector<ssize_t>{4});
    ASSERT_EQ(store->Setup(config), UC::Status::OK());

    auto block = MakeBlockId(7);
    std::array<std::byte, 4> src{std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4}};
    std::array<std::byte, 4> dst{};
    UC::Detail::TokenLayerTaskDesc dump;
    dump.push_back({block, 0, 0, 0, {src.data()}});
    auto dumpTask = store->DumpTokens(dump);
    ASSERT_TRUE(dumpTask.HasValue());
    ASSERT_EQ(store->Wait(dumpTask.Value()), UC::Status::OK());

    UC::Detail::TokenLayerTaskDesc query;
    query.push_back({block, 0, 0, 0, {dst.data()}});
    auto lookup = store->LookupTokens(query);
    ASSERT_TRUE(lookup.HasValue());
    ASSERT_EQ(lookup.Value(), std::vector<uint8_t>{1});

    auto loadTask = store->LoadTokens(query);
    ASSERT_TRUE(loadTask.HasValue());
    ASSERT_EQ(store->Wait(loadTask.Value()), UC::Status::OK());
    EXPECT_EQ(std::memcmp(src.data(), dst.data(), src.size()), 0);
}

TEST(UCMMemoryStoreStructureTest, TransferUsesPublicTransStreamApi)
{
    UC::Trans::Device device;
    ASSERT_EQ(device.Setup(0), UC::Status::OK());

    std::array<std::byte, 4> src{std::byte{9}, std::byte{8}, std::byte{7}, std::byte{6}};
    std::array<std::byte, 4> dst{};
    auto stream = device.MakeSharedStream();
    ASSERT_NE(stream, nullptr);
    ASSERT_EQ(stream->HostToDeviceAsync(src.data(), dst.data(), src.size()), UC::Status::OK());
    ASSERT_EQ(stream->Synchronized(), UC::Status::OK());
    EXPECT_EQ(std::memcmp(src.data(), dst.data(), src.size()), 0);
}

TEST(UCMMemoryStoreStructureTest, LoadQueueKeepsBackendLoadAsyncAcrossStages)
{
    using namespace UC::MemoryStore;

    BlockingLoadBackend backend{
        {std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4}, std::byte{5}, std::byte{6},
         std::byte{7}, std::byte{8}}
    };
    Config config;
    config.storeBackend = &backend;
    config.deviceId = 0;
    config.shardSize = 8;
    config.blockSize = 8;
    config.tensorSizes = {8};
    config.memoryTokenChunkSize = 1;
    config.memoryBufferCapacity = 1ULL << 20;
    config.requiredTensorTypes = {0};
    config.tensorSizesByType[0] = {8};
    config.tokensPerBlock = 1;

    TransBuffer buffer;
    ASSERT_EQ(buffer.Setup(config), UC::Status::OK());
    UC::HashSet<UC::Detail::TaskHandle> failureSet;
    LoadQueue loadQ;
    ASSERT_EQ(loadQ.Setup(config, &failureSet, &buffer), UC::Status::OK());

    std::array<std::byte, 8> dst1{};
    std::array<std::byte, 8> dst2{};
    UC::Detail::TaskDesc desc1{
        {MakeBlockId(11), 0, {dst1.data()}}
    };
    UC::Detail::TaskDesc desc2{
        {MakeBlockId(22), 0, {dst2.data()}}
    };
    auto task1 = std::make_shared<TransTask>(TransTask::Type::LOAD, desc1);
    auto task2 = std::make_shared<TransTask>(TransTask::Type::LOAD, desc2);
    auto waiter1 = std::make_shared<UC::Latch>();
    auto waiter2 = std::make_shared<UC::Latch>();

    loadQ.Submit(task1, waiter1);
    loadQ.Submit(task2, waiter2);

    ASSERT_TRUE(backend.WaitUntilLoads(2, 2000));
    backend.ReleaseWaiters();

    ASSERT_TRUE(waiter1->WaitForDuration(2000));
    ASSERT_TRUE(waiter2->WaitForDuration(2000));
    ASSERT_FALSE(failureSet.Contains(task1->id));
    ASSERT_FALSE(failureSet.Contains(task2->id));

    std::array<std::byte, 8> expected{std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4},
                                      std::byte{5}, std::byte{6}, std::byte{7}, std::byte{8}};
    EXPECT_EQ(std::memcmp(dst1.data(), expected.data(), expected.size()), 0);
    EXPECT_EQ(std::memcmp(dst2.data(), expected.data(), expected.size()), 0);
}

TEST(UCMMemoryStoreStructureTest, TransBufferSetupDoesNotTakeConfigByValue)
{
    using SetupSignature = decltype(&UC::MemoryStore::TransBuffer::Setup);
    using ExpectedSignature =
        UC::Status (UC::MemoryStore::TransBuffer::*)(const UC::MemoryStore::Config&);
    EXPECT_TRUE((std::is_same_v<SetupSignature, ExpectedSignature>));
}
