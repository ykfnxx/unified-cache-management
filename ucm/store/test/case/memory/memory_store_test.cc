/**
 * MIT License
 *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
 * */
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <memory>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <gtest/gtest.h>
#include "dump_queue.h"
#include "load_queue.h"
#include "memory_store.h"
#include "trans/device.h"
#include "trans_buffer.h"
#include "ucmstore_v1.h"

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

    std::string Readme() const override { return "BlockingLoadBackend"; }
    UC::Expected<std::vector<uint8_t>> Lookup(const UC::Detail::BlockId*, size_t num) override
    {
        return std::vector<uint8_t>(num, false);
    }
    UC::Expected<ssize_t> LookupOnPrefix(const UC::Detail::BlockId*, size_t) override
    {
        return -1;
    }
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

class BlockingDumpBackend : public UC::StoreV1 {
    struct PendingTask {
        UC::Detail::TaskDesc desc;
    };

public:
    std::string Readme() const override { return "BlockingDumpBackend"; }
    UC::Expected<std::vector<uint8_t>> Lookup(const UC::Detail::BlockId*, size_t num) override
    {
        return std::vector<uint8_t>(num, false);
    }
    UC::Expected<ssize_t> LookupOnPrefix(const UC::Detail::BlockId*, size_t) override
    {
        return -1;
    }
    void Prefetch(const UC::Detail::BlockId*, size_t) override {}
    UC::Expected<UC::Detail::TaskHandle> Load(UC::Detail::TaskDesc) override
    {
        return UC::Status::Unsupported();
    }
    UC::Expected<UC::Detail::TaskHandle> Dump(UC::Detail::TaskDesc task) override
    {
        auto id = nextId_.fetch_add(1, std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> guard(mutex_);
            pending_.emplace(id, PendingTask{std::move(task)});
        }
        dumpCalls_.fetch_add(1, std::memory_order_relaxed);
        cv_.notify_all();
        return id;
    }
    UC::Expected<bool> Check(UC::Detail::TaskHandle) override { return true; }
    UC::Status Wait(UC::Detail::TaskHandle taskId) override
    {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return releaseWait_; });
        auto iter = pending_.find(taskId);
        if (iter == pending_.end()) { return UC::Status::NotFound(); }
        pending_.erase(iter);
        return UC::Status::OK();
    }

    bool WaitUntilDumps(size_t expected, size_t timeoutMs)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        return cv_.wait_for(lock, std::chrono::milliseconds(timeoutMs), [this, expected] {
            return dumpCalls_.load(std::memory_order_relaxed) >= expected;
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
    std::atomic<size_t> dumpCalls_{0};
    bool releaseWait_{false};
};
}  // namespace

TEST(UCMTokenLayerApiTest, DefaultStoreV1TokenLayerMethodsAreUnsupported)
{
    class MinimalStore : public UC::StoreV1 {
    public:
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
    auto store = std::make_unique<UC::MemoryStore::MemoryStore>();
    UC::MemoryStore::Config config;
    config.storeBackend = 0;
    config.deviceId = 0;
    config.shardSize = 16;
    config.blockSize = 16;
    config.tensorSizeList = {8, 8};
    config.memoryTokenChunkSize = 2;
    config.memoryBufferCapacity = 1ULL << 20;
    config.requiredTensorTypes = {0, 1};
    config.tensorSizesByType = {{0, {4}}, {1, {4}}};
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

TEST(UCMMemoryStoreStructureTest, DumpTransferUsesPublicDeviceToHostApi)
{
    UC::Trans::Device device;
    ASSERT_EQ(device.Setup(0), UC::Status::OK());

    std::array<std::byte, 4> src{std::byte{9}, std::byte{8}, std::byte{7}, std::byte{6}};
    std::array<std::byte, 4> dst{};
    auto stream = device.MakeStream();
    ASSERT_NE(stream, nullptr);
    ASSERT_EQ(stream->DeviceToHostAsync(src.data(), dst.data(), src.size()), UC::Status::OK());
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
    config.storeBackend = reinterpret_cast<uintptr_t>(&backend);
    config.deviceId = 0;
    config.shardSize = 8;
    config.blockSize = 8;
    config.tensorSizeList = {8};
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
    UC::Detail::TaskDesc desc1{{MakeBlockId(11), 0, {dst1.data()}}};
    UC::Detail::TaskDesc desc2{{MakeBlockId(22), 0, {dst2.data()}}};
    auto task1 = std::make_shared<TransTask>(TransTask::Type::LOAD, desc1);
    auto task2 = std::make_shared<TransTask>(TransTask::Type::LOAD, desc2);
    auto waiter1 = std::make_shared<UC::Latch>();
    auto waiter2 = std::make_shared<UC::Latch>();

    loadQ.Submit(task1, waiter1);
    loadQ.Submit(task2, waiter2);

    ASSERT_TRUE(backend.WaitUntilLoads(2, 2000));
    backend.ReleaseWaiters();

    ASSERT_TRUE(waiter1->WaitFor(2000));
    ASSERT_TRUE(waiter2->WaitFor(2000));
    ASSERT_FALSE(failureSet.Contains(task1->id));
    ASSERT_FALSE(failureSet.Contains(task2->id));

    std::array<std::byte, 8> expected{std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4},
                                      std::byte{5}, std::byte{6}, std::byte{7}, std::byte{8}};
    EXPECT_EQ(std::memcmp(dst1.data(), expected.data(), expected.size()), 0);
    EXPECT_EQ(std::memcmp(dst2.data(), expected.data(), expected.size()), 0);
}

TEST(UCMMemoryStoreStructureTest, DumpQueueKeepsBackendDumpAsyncAcrossStages)
{
    using namespace UC::MemoryStore;

    BlockingDumpBackend backend;
    Config config;
    config.storeBackend = reinterpret_cast<uintptr_t>(&backend);
    config.deviceId = 0;
    config.shardSize = 8;
    config.blockSize = 8;
    config.tensorSizeList = {8};
    config.memoryTokenChunkSize = 1;
    config.memoryBufferCapacity = 1ULL << 20;
    config.requiredTensorTypes = {0};
    config.tensorSizesByType[0] = {8};
    config.tokensPerBlock = 1;

    TransBuffer buffer;
    ASSERT_EQ(buffer.Setup(config), UC::Status::OK());
    UC::HashSet<UC::Detail::TaskHandle> failureSet;
    DumpQueue dumpQ;
    ASSERT_EQ(dumpQ.Setup(config, &failureSet, &buffer), UC::Status::OK());

    std::array<std::byte, 8> src1{std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4},
                                  std::byte{5}, std::byte{6}, std::byte{7}, std::byte{8}};
    std::array<std::byte, 8> src2{std::byte{9}, std::byte{10}, std::byte{11}, std::byte{12},
                                  std::byte{13}, std::byte{14}, std::byte{15}, std::byte{16}};
    UC::Detail::TaskDesc desc1{{MakeBlockId(31), 0, {src1.data()}}};
    UC::Detail::TaskDesc desc2{{MakeBlockId(32), 0, {src2.data()}}};
    auto task1 = std::make_shared<TransTask>(TransTask::Type::DUMP, desc1);
    auto task2 = std::make_shared<TransTask>(TransTask::Type::DUMP, desc2);
    auto waiter1 = std::make_shared<UC::Latch>();
    auto waiter2 = std::make_shared<UC::Latch>();

    dumpQ.Submit(task1, waiter1);
    dumpQ.Submit(task2, waiter2);

    ASSERT_TRUE(backend.WaitUntilDumps(2, 2000));
    backend.ReleaseWaiters();

    ASSERT_TRUE(waiter1->WaitFor(2000));
    ASSERT_TRUE(waiter2->WaitFor(2000));
    ASSERT_FALSE(failureSet.Contains(task1->id));
    ASSERT_FALSE(failureSet.Contains(task2->id));
}

TEST(UCMMemoryStoreStructureTest, DumpTokenQueueKeepsBackendDumpAsyncAcrossStages)
{
    using namespace UC::MemoryStore;

    BlockingDumpBackend backend;
    Config config;
    config.storeBackend = reinterpret_cast<uintptr_t>(&backend);
    config.deviceId = 0;
    config.shardSize = 8;
    config.blockSize = 8;
    config.tensorSizeList = {8};
    config.memoryTokenChunkSize = 1;
    config.memoryBufferCapacity = 1ULL << 20;
    config.requiredTensorTypes = {0, 1};
    config.tensorSizesByType[0] = {4};
    config.tensorSizesByType[1] = {4};
    config.tokensPerBlock = 1;

    TransBuffer buffer;
    ASSERT_EQ(buffer.Setup(config), UC::Status::OK());
    UC::HashSet<UC::Detail::TaskHandle> failureSet;
    DumpQueue dumpQ;
    ASSERT_EQ(dumpQ.Setup(config, &failureSet, &buffer), UC::Status::OK());

    std::array<std::byte, 4> src1a{std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4}};
    std::array<std::byte, 4> src1b{std::byte{5}, std::byte{6}, std::byte{7}, std::byte{8}};
    std::array<std::byte, 4> src2a{std::byte{9}, std::byte{10}, std::byte{11}, std::byte{12}};
    std::array<std::byte, 4> src2b{std::byte{13}, std::byte{14}, std::byte{15}, std::byte{16}};
    UC::Detail::TokenLayerTaskDesc desc1;
    desc1.push_back({MakeBlockId(41), 0, 0, 0, {src1a.data()}});
    desc1.push_back({MakeBlockId(41), 0, 0, 1, {src1b.data()}});
    UC::Detail::TokenLayerTaskDesc desc2;
    desc2.push_back({MakeBlockId(42), 0, 0, 0, {src2a.data()}});
    desc2.push_back({MakeBlockId(42), 0, 0, 1, {src2b.data()}});

    auto task1 = std::make_shared<TransTask>(TransTask::Type::DUMP_TOKENS, desc1);
    auto task2 = std::make_shared<TransTask>(TransTask::Type::DUMP_TOKENS, desc2);
    auto waiter1 = std::make_shared<UC::Latch>();
    auto waiter2 = std::make_shared<UC::Latch>();

    dumpQ.Submit(task1, waiter1);
    dumpQ.Submit(task2, waiter2);

    ASSERT_TRUE(backend.WaitUntilDumps(2, 2000));
    backend.ReleaseWaiters();

    ASSERT_TRUE(waiter1->WaitFor(2000));
    ASSERT_TRUE(waiter2->WaitFor(2000));
    ASSERT_FALSE(failureSet.Contains(task1->id));
    ASSERT_FALSE(failureSet.Contains(task2->id));
}

TEST(UCMMemoryStoreStructureTest, TransBufferSetupDoesNotTakeConfigByValue)
{
    using SetupSignature = decltype(&UC::MemoryStore::TransBuffer::Setup);
    using ExpectedSignature =
        UC::Status (UC::MemoryStore::TransBuffer::*)(const UC::MemoryStore::Config&);
    EXPECT_TRUE((std::is_same_v<SetupSignature, ExpectedSignature>));
}
