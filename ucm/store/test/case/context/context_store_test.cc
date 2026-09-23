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
#include <cstring>
#include <future>
#include <gtest/gtest.h>
#include <map>
#include <mutex>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include "load_queue.h"
#include "metrics_api.h"
#include "trans_buffer.h"
extern "C" UC::StoreV1* MakeContextStore();
namespace {
using namespace UC;
using Key = Detail::BlockId;
Key Id(size_t n)
{
    Key k{};
    std::memcpy(k.data(), &n, sizeof(n));
    return k;
}
class Backend : public StoreV1 {
public:
    std::map<std::pair<Key, size_t>, std::array<char, 64>> data;
    size_t writes = 0;
    bool fail = false;
    Status Setup(const Detail::Dictionary&) override { return Status::OK(); }
    std::string Readme() const override { return "TestBackend"; }
    Expected<std::vector<uint8_t>> Lookup(const Key* ids, size_t n) override
    {
        std::vector<uint8_t> found(n);
        for (size_t i = 0; i < n; ++i) { found[i] = data.count({ids[i], 0}); }
        return found;
    }
    Expected<ssize_t> LookupOnPrefix(const Key*, size_t) override { return -1; }
    Expected<ssize_t> LookupOnReverse(const Key*, size_t) override { return -1; }
    void Prefetch(const Key*, size_t) override {}
    Expected<size_t> Load(Detail::TaskDesc task) override
    {
        for (const auto& s : task) {
            auto it = data.find({s.owner, s.index});
            if (it == data.end()) { return Status::NotFound(); }
            std::memcpy(s.addrs[0], it->second.data(), 64);
        }
        return size_t(1);
    }
    Expected<size_t> Dump(Detail::TaskDesc task) override
    {
        if (fail) { return Status::Error("writeback failure"); }
        for (const auto& s : task) {
            std::memcpy(data[{s.owner, s.index}].data(), s.addrs[0], 64);
            ++writes;
        }
        return size_t(1);
    }
    Expected<bool> Check(size_t) override { return true; }
    Status Wait(size_t) override { return Status::OK(); }
};
class ContextBufferTest : public ::testing::Test {
protected:
    Backend backend;
    Context::Config cfg;
    Context::TransBuffer buffer;
    void Open(size_t slots = 2, int64_t retention = -1, bool shared = false)
    {
        static size_t serial = 0;
        cfg.uniqueId = "clock_test_" + std::to_string(getpid()) + "_" + std::to_string(++serial);
        cfg.deviceId = 0;
        cfg.storeBackend = &backend;
        cfg.shardSize = 64;
        cfg.blockSize = 128;
        cfg.tensorSizes = {64};
        cfg.bufferCapacity = slots * 64;
        cfg.loadExclusiveBufferNumber = 0;
        cfg.shareBufferEnable = shared;
        cfg.ioDirect = false;
        cfg.retentionNs = retention;
        ASSERT_TRUE(buffer.Setup(cfg).Success());
    }
    void Save(size_t key, size_t layer = 0, bool persisted = false)
    {
        auto h = buffer.Get(Id(key), layer);
        ASSERT_TRUE(h) << h.Error().ToString();
        std::memset(h.Value().Data(), int(key), 64);
        h.Value().MarkReady(persisted);
    }
};
TEST_F(ContextBufferTest, OnlyEvictionWritesAndUsesVictimPayload)
{
    Open();
    Save(1);
    Save(2);
    EXPECT_EQ(backend.writes, 0);
    Save(3);
    EXPECT_EQ(backend.writes, 1);
    ASSERT_EQ(backend.data.size(), 1);
    for (char byte : backend.data.begin()->second) { EXPECT_EQ(byte, 1); }
}
TEST_F(ContextBufferTest, EvictsOneShardNotWholeBlock)
{
    Open();
    Save(1, 0);
    Save(1, 1);
    Save(2, 0);
    EXPECT_FALSE(buffer.Exist(Id(1), 0));
    EXPECT_TRUE(buffer.Exist(Id(1), 1));
    EXPECT_EQ(backend.writes, 1);
    EXPECT_EQ(backend.data.count({Id(1), 0}), 1);
}
TEST_F(ContextBufferTest, ExpiredShardDropsWithoutWriting)
{
    Open(1, 0);
    Save(1);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    Save(2);
    EXPECT_EQ(backend.writes, 0);
    EXPECT_FALSE(buffer.Exist(Id(1), 0));
}
TEST_F(ContextBufferTest, ReadRefreshesRetention)
{
    Open(1, 50000000);
    Save(1);
    std::this_thread::sleep_for(std::chrono::milliseconds(60));
    {
        auto h = buffer.Get(Id(1), 0);
        ASSERT_TRUE(h);
        EXPECT_TRUE(h.Value().Ready());
    }
    Save(2);
    EXPECT_EQ(backend.writes, 1);
}
TEST_F(ContextBufferTest, LoadedCleanShardDoesNotWriteAgain)
{
    Open(1);
    Save(1, 0, true);
    Save(2);
    EXPECT_EQ(backend.writes, 0);
}
TEST_F(ContextBufferTest, HeldShardCannotBeReused)
{
    Open();
    Save(1);
    Save(2);
    auto pinned = buffer.Get(Id(1), 0);
    ASSERT_TRUE(pinned);
    Save(3);
    EXPECT_TRUE(buffer.Exist(Id(1), 0));
    EXPECT_FALSE(buffer.Exist(Id(2), 0));
    EXPECT_EQ(*static_cast<char*>(pinned.Value().Data()), 1);
}
TEST_F(ContextBufferTest, WriteFailurePreservesMappingAndPayload)
{
    Open(1);
    Save(1);
    backend.fail = true;
    auto failed = buffer.Get(Id(2), 0);
    EXPECT_FALSE(failed);
    EXPECT_TRUE(buffer.Exist(Id(1), 0));
    {
        auto h = buffer.Get(Id(1), 0);
        ASSERT_TRUE(h);
        EXPECT_TRUE(h.Value().Ready());
        EXPECT_EQ(*static_cast<char*>(h.Value().Data()), 1);
    }
    backend.fail = false;
    Save(2);
    EXPECT_EQ(backend.writes, 1);
}
TEST_F(ContextBufferTest, FailedAndPreallocatedSlotsAreNotWritten)
{
    Open(1);
    buffer.Prealloc(Id(1), 0);
    {
        auto h = buffer.Get(Id(1), 0);
        ASSERT_TRUE(h);
        h.Value().MarkFailed(Status::Error());
    }
    Save(2);
    EXPECT_EQ(backend.writes, 0);
}
TEST_F(ContextBufferTest, SharedReadersSeeOnePayload)
{
    Open(4, -1, true);
    Save(1);
    std::vector<std::unique_ptr<Context::TransBuffer>> readers;
    for (int rank = 1; rank < 16; ++rank) {
        auto reader = std::make_unique<Context::TransBuffer>();
        cfg.deviceId = rank;
        ASSERT_TRUE(reader->Setup(cfg).Success());
        readers.push_back(std::move(reader));
    }
    std::atomic<int> errors{0};
    std::vector<std::thread> threads;
    for (auto& reader : readers) {
        auto* ptr = reader.get();
        threads.emplace_back([&, ptr] {
            for (int i = 0; i < 1000; ++i) {
                auto h = ptr->Get(Id(1), 0);
                if (!h || !h.Value().Ready() || *static_cast<char*>(h.Value().Data()) != 1) {
                    ++errors;
                }
            }
        });
    }
    for (auto& t : threads) { t.join(); }
    EXPECT_EQ(errors, 0);
    EXPECT_EQ(backend.writes, 0);
}
TEST_F(ContextBufferTest, SharedShardReadableFromAnotherProcess)
{
    Open(4, -1, true);
    Save(7, 1);
    auto child = fork();
    ASSERT_GE(child, 0);
    if (child == 0) {
        Context::TransBuffer reader;
        cfg.deviceId = 1;
        if (reader.Setup(cfg).Failure()) { _exit(2); }
        auto h = reader.Get(Id(7), 1);
        _exit(h && h.Value().Ready() && *static_cast<char*>(h.Value().Data()) == 7 ? 0 : 3);
    }
    int status;
    ASSERT_EQ(waitpid(child, &status, 0), child);
    ASSERT_TRUE(WIFEXITED(status));
    EXPECT_EQ(WEXITSTATUS(status), 0);
}
TEST_F(ContextBufferTest, MetricsCountActualWrites)
{
    Metrics::SetUp();
    for (auto name : {"context_evict_shards_total", "context_writeback_shards_total",
                      "context_drop_shards_total"}) {
        Metrics::CreateStats(name, "counter");
    }
    Metrics::GetAllStatsAndClear();
    Open(1);
    Save(1);
    Save(2);
    auto counters = std::get<0>(Metrics::GetAllStatsAndClear());
    EXPECT_EQ(counters["context_evict_shards_total"], 1);
    EXPECT_EQ(counters["context_writeback_shards_total"], 1);
    EXPECT_EQ(counters["context_drop_shards_total"], 0);
}
TEST_F(ContextBufferTest, MetricsCountDropsWithoutWrites)
{
    Metrics::SetUp();
    for (auto name : {"context_evict_shards_total", "context_writeback_shards_total",
                      "context_drop_shards_total"}) {
        Metrics::CreateStats(name, "counter");
    }
    Metrics::GetAllStatsAndClear();
    Open(1, 0);
    Save(1);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    Save(2);
    auto counters = std::get<0>(Metrics::GetAllStatsAndClear());
    EXPECT_EQ(counters["context_evict_shards_total"], 1);
    EXPECT_EQ(counters["context_writeback_shards_total"], 0);
    EXPECT_EQ(counters["context_drop_shards_total"], 1);
}
TEST_F(ContextBufferTest, QueuedWritebackFailureDrainsEarlierCopies)
{
    Open();
    Save(1);
    Save(2);
    backend.fail = true;
    cfg.localRankSize = 1;
    cfg.waitingQueueDepth = 8;
    cfg.runningQueueDepth = 8;
    HashSet<Detail::TaskHandle> failures;
    Context::LoadQueue queue;
    ASSERT_TRUE(queue.Setup(cfg, &failures, &buffer).Success());
    std::array<char, 64> first{}, second{};
    auto task = std::make_shared<Context::TransTask>(
        Context::TransTask::Type::LOAD,
        Detail::TaskDesc{
            {Id(1), 0, {first.data()} },
            {Id(3), 0, {second.data()}}
    });
    auto waiter = std::make_shared<Latch>();
    queue.Submit(task, waiter);
    ASSERT_TRUE(waiter->WaitFor(5000));
    EXPECT_TRUE(failures.Contains(task->id));
    EXPECT_TRUE(buffer.Exist(Id(2), 0));
    for (char byte : first) { EXPECT_EQ(byte, 1); }
    backend.fail = false;
    Save(4);  // The completed task must no longer pin its copied shard.
}

TEST(ContextStoreTest, QueuedSaveAndLayerLoadsPreserveThreeComponents)
{
    Backend backend;
    Detail::Dictionary cfg;
    cfg.Set("unique_id", "clock_store_" + std::to_string(getpid()));
    cfg.Set<StoreV1*>("store_backend", &backend);
    cfg.SetNumber("device_id", 0);
    cfg.Set("share_buffer_enable", false);
    cfg.Set("io_direct", false);
    cfg.SetNumber("context_memory_capacity_gb", 1);
    cfg.SetNumber("cache_load_exclusive_buffer_number", 0);
    cfg.SetNumber("shard_size", 1 << 20);
    cfg.SetNumber("block_size", 3 << 20);
    cfg.Set("tensor_size_list", std::vector<ssize_t>{32, 16, 16});
    std::unique_ptr<StoreV1> store(MakeContextStore());
    ASSERT_TRUE(store->Setup(cfg).Success());
    std::array<char, 64> input, output{};
    input.fill(41);
    for (size_t layer = 0; layer < 3; ++layer) {
        auto task = store->Dump({
            {Id(1), layer, {input.data(), input.data() + 32, input.data() + 48}}
        });
        ASSERT_TRUE(task);
        ASSERT_TRUE(store->Wait(task.Value()).Success());
        auto load = store->Load({
            {Id(1), layer, {output.data(), output.data() + 32, output.data() + 48}}
        });
        ASSERT_TRUE(load);
        ASSERT_TRUE(store->Wait(load.Value()).Success());
        EXPECT_EQ(output, input);
    }
    EXPECT_EQ(backend.writes, 0);
}
TEST(ContextStoreTest, OriginalBytesCapacityControlsEviction)
{
    Backend backend;
    Detail::Dictionary cfg;
    cfg.Set("unique_id", "clock_bytes_" + std::to_string(getpid()));
    cfg.Set<StoreV1*>("store_backend", &backend);
    cfg.SetNumber("device_id", 0);
    cfg.Set("share_buffer_enable", false);
    cfg.Set("io_direct", false);
    cfg.SetNumber("context_memory_capacity_bytes", 1024 * 64);
    cfg.SetNumber("cache_load_exclusive_buffer_number", 0);
    cfg.SetNumber("shard_size", 64);
    cfg.SetNumber("block_size", 64);
    cfg.SetNumber("tensor_size", 64);
    cfg.SetNumber("waiting_queue_depth", 8);
    cfg.SetNumber("running_queue_depth", 8);
    std::unique_ptr<StoreV1> store(MakeContextStore());
    ASSERT_TRUE(store->Setup(cfg).Success());
    std::array<char, 64> input{};
    for (size_t i = 0; i < 1024; ++i) {
        auto task = store->Dump({
            {Id(i), 0, {input.data()}}
        });
        ASSERT_TRUE(task);
        ASSERT_TRUE(store->Wait(task.Value()).Success());
    }
    EXPECT_EQ(backend.writes, 0);
    auto task = store->Dump({
        {Id(1024), 0, {input.data()}}
    });
    ASSERT_TRUE(task);
    ASSERT_TRUE(store->Wait(task.Value()).Success());
    EXPECT_EQ(backend.writes, 1);
}
}  // namespace
