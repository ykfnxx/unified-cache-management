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
#include <cstring>
#include <future>
#include <gtest/gtest.h>
#include <mutex>
#include <numeric>
#include <random>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include "context_index.h"
#include "metrics_api.h"
#include "shared_metadata.h"
#include "ucmstore_v1.h"
extern "C" UC::StoreV1* MakeContextStore();
namespace {
using namespace UC;
using Key = Detail::BlockId;
Key Id(unsigned n)
{
    Key k{};
    k[0] = std::byte(n);
    return k;
}
// A byte-preserving test backend validates addresses, admission and failure handling.
// Production Fake is exercised separately by the native pipeline tests.
class TestBackend : public StoreV1 {
public:
    std::mutex mutex;
    std::map<std::pair<Key, size_t>, std::array<unsigned char, 64>> data;
    bool failDump = false, failLoad = false;
    size_t lookups = 0, loads = 0;
    std::function<void()> beforeLoad;
    Status Setup(const Detail::Dictionary&) override { return Status::OK(); }
    std::string Readme() const override { return "TestBackend"; }
    Expected<std::vector<uint8_t>> Lookup(const Key* keys, size_t n) override
    {
        std::lock_guard<std::mutex> lock(mutex);
        ++lookups;
        std::vector<uint8_t> result;
        for (size_t i = 0; i < n; ++i) { result.push_back(data.count({keys[i], 0})); }
        return result;
    }
    Expected<ssize_t> LookupOnPrefix(const Key*, size_t) override { return -1; }
    Expected<ssize_t> LookupOnReverse(const Key*, size_t) override { return -1; }
    Expected<size_t> Dump(Detail::TaskDesc desc) override
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (failDump) { return Status::Error("injected dump failure"); }
        for (const auto& shard : desc) {
            std::memcpy(data[{shard.owner, shard.index}].data(), shard.addrs[0], 64);
        }
        return size_t(1);
    }
    Expected<size_t> Load(Detail::TaskDesc desc) override
    {
        if (beforeLoad) { beforeLoad(); }
        std::lock_guard<std::mutex> lock(mutex);
        ++loads;
        if (failLoad) { return Status::Error("injected load failure"); }
        for (const auto& shard : desc) {
            auto it = data.find({shard.owner, shard.index});
            if (it == data.end()) { return Status::NotFound(); }
            std::memcpy(shard.addrs[0], it->second.data(), 64);
        }
        return size_t(1);
    }
    void Prefetch(const Key*, size_t) override {}
    Expected<bool> Check(size_t) override { return true; }
    Status Wait(size_t) override { return Status::OK(); }
};
class ContextStoreTest : public ::testing::Test {
protected:
    TestBackend backend;
    std::unique_ptr<StoreV1> store, watcher;
    std::string name;
    uint64_t clock = 0;
    void Open(size_t memory = 2, int64_t retention = -1, size_t shards = 1, bool shared = false)
    {
        static unsigned next = 0;
        name = "test_" + std::to_string(getpid()) + "_" + std::to_string(++next);
        Detail::Dictionary config;
        config.Set("unique_id", name);
        config.Set<StoreV1*>("store_backend", &backend);
        config.Set("share_buffer_enable", shared);
        watcher.reset(MakeContextStore());
        ASSERT_TRUE(watcher->Setup(config).Success());
        config.SetNumber("device_id", 0);
        config.SetNumber("block_size", 64 * shards);
        config.SetNumber("shard_size", 64);
        config.SetNumber("tensor_size", 64);
        config.SetNumber("context_memory_capacity_bytes", memory * 64 * shards);
        config.SetNumber("context_retention_ns", retention);
        config.SetNumber("context_max_eviction_blocks", 1);
        config.SetNumber("cache_stream_number", 2);
        store.reset(MakeContextStore());
        ASSERT_TRUE(store->Setup(config).Success());
    }
    void Observe(std::initializer_list<unsigned> keys, uint64_t time = 0)
    {
        std::vector<Key> path;
        for (auto key : keys) { path.push_back(Id(key)); }
        ++clock;
        ASSERT_TRUE(store->ObserveRequest(std::to_string(clock), clock, time ? time : clock, path)
                        .Success());
    }
    Status Dump(unsigned key, size_t shard = 0, unsigned char value = 71)
    {
        std::array<unsigned char, 64> input;
        input.fill(value);
        Detail::TaskDesc desc{
            Detail::Shard{Id(key), shard, {input.data()}}
        };
        auto task = store->Dump(std::move(desc));
        return task ? store->Wait(task.Value()) : task.Error();
    }
    bool Found(unsigned key)
    {
        auto k = Id(key);
        auto found = watcher->Lookup(&k, 1);
        EXPECT_TRUE(bool(found));
        return found && found.Value()[0];
    }
    Status Load(unsigned key, size_t shard = 0, unsigned char value = 71)
    {
        std::array<unsigned char, 64> output{};
        auto task = store->Load(Detail::TaskDesc{
            Detail::Shard{Id(key), shard, {output.data()}}
        });
        if (!task) { return task.Error(); }
        auto status = store->Wait(task.Value());
        if (status.Success()) {
            for (auto v : output) { EXPECT_EQ(v, value); }
        }
        return status;
    }
};
TEST_F(ContextStoreTest, BatchedReadmissionRollsBackAndRetries)
{
    Open(70, -1, 2);
    std::vector<Key> path;
    std::vector<std::array<unsigned char, 64>> output(70);
    Detail::TaskDesc desc;
    for (unsigned i = 1; i <= 70; ++i) {
        path.push_back(Id(i));
        for (size_t shard = 0; shard < 2; ++shard) { backend.data[{Id(i), shard}].fill(i); }
        desc.push_back({Id(i), 0, {output[i - 1].data()}});
    }
    ASSERT_TRUE(store->ObserveRequest("batch", 1, 1, path).Success());
    backend.failLoad = true;
    auto failed = store->Load(desc);
    ASSERT_TRUE(failed);
    EXPECT_TRUE(store->Wait(failed.Value()).Failure());
    EXPECT_EQ(store->ContextStats()["memory_blocks"], 0);
    EXPECT_EQ(backend.loads, 1);
    backend.failLoad = false;
    backend.loads = backend.lookups = 0;
    backend.beforeLoad = [&] {
        if (backend.loads == 1) { backend.failLoad = true; }
    };
    auto partial = store->Load(desc);
    ASSERT_TRUE(partial);
    EXPECT_TRUE(store->Wait(partial.Value()).Failure());
    EXPECT_EQ(store->ContextStats()["memory_blocks"], 32);
    EXPECT_EQ(backend.loads, 2);
    for (size_t i = 0; i < 32; ++i) { EXPECT_EQ(output[i][0], i + 1); }
    backend.beforeLoad = nullptr;
    backend.failLoad = false;
    backend.loads = backend.lookups = 0;
    auto task = store->Load(desc);
    ASSERT_TRUE(task);
    ASSERT_TRUE(store->Wait(task.Value()).Success());
    EXPECT_EQ(backend.lookups, 1);
    EXPECT_EQ(backend.loads, 2);
    EXPECT_EQ(store->ContextStats()["backend_load_shards"], 140);
    for (size_t i = 0; i < output.size(); ++i) {
        EXPECT_TRUE(std::all_of(output[i].begin(), output[i].end(),
                                [i](unsigned char b) { return b == i + 1; }));
    }
}

TEST_F(ContextStoreTest, MetricsCountCompletedBlockEvictions)
{
    Metrics::SetUp();
    for (const auto* name :
         {"context_evict_blocks_total", "context_dump_blocks_total", "context_drop_blocks_total"}) {
        Metrics::CreateStats(name, "counter");
    }
    auto check = [](double evict, double dump, double drop) {
        auto counters = std::get<0>(Metrics::GetAllStatsAndClear());
        EXPECT_EQ(counters["context_evict_blocks_total"], evict);
        EXPECT_EQ(counters["context_dump_blocks_total"], dump);
        EXPECT_EQ(counters["context_drop_blocks_total"], drop);
        EXPECT_EQ(evict, dump + drop);
    };
    Open(1, 5, 2);
    Observe({1}, 1);
    ASSERT_TRUE(Dump(1, 0).Success());
    ASSERT_TRUE(Dump(1, 1).Success());
    check(0, 0, 0);  // Device saves are not policy Dump evictions.
    Observe({2}, 2);
    backend.failDump = true;
    EXPECT_TRUE(Dump(2).Failure());
    check(0, 0, 0);  // A failed backend write retains the victim.
    backend.failDump = false;
    ASSERT_TRUE(Dump(2, 0).Success());
    ASSERT_TRUE(Dump(2, 1).Success());
    check(1, 1, 0);  // Both layers form one logical block.
    ASSERT_TRUE(Load(1).Success());
    check(1, 1, 0);  // Admission can also trigger an eviction.
    Observe({3}, 3);
    ASSERT_TRUE(Dump(3, 0).Success());
    ASSERT_TRUE(Dump(3, 1).Success());
    EXPECT_EQ(store->ContextStats()["backend_dump_skipped_blocks"], 1);
    check(1, 1, 0);  // Backend-already-present still counts as policy Dump.
    Observe({4}, 20);
    ASSERT_TRUE(Dump(4).Success());
    check(1, 0, 1);
    check(0, 0, 0);  // Collection drains deltas, without counting them twice.
}

TEST_F(ContextStoreTest, H2dStartsBeforeLaterBlockAdmissionCompletes)
{
    Metrics::SetUp();
    Metrics::CreateStats("context_load_first_h2d_ms", "histogram", {1, 1000});
    Open(2);
    Observe({1});
    ASSERT_TRUE(Dump(1).Success());
    Observe({2});
    backend.data[{Id(2), 0}].fill(83);
    std::promise<void> entered, unblock;
    auto enteredFuture = entered.get_future();
    auto released = unblock.get_future().share();
    backend.beforeLoad = [&] {
        entered.set_value();
        released.wait();
    };
    std::array<unsigned char, 64> first{}, second{};
    auto task = store->Load(Detail::TaskDesc{
        Detail::Shard{Id(1), 0, {first.data()} },
        Detail::Shard{Id(2), 0, {second.data()}}
    });
    ASSERT_TRUE(bool(task));
    auto preparing = enteredFuture.wait_for(std::chrono::seconds(2));
    bool submitted = false;
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (preparing == std::future_status::ready && !submitted &&
           std::chrono::steady_clock::now() < deadline) {
        auto histograms = std::get<2>(Metrics::GetAllStatsAndClear());
        auto it = histograms.find("context_load_first_h2d_ms");
        submitted = it != histograms.end() &&
                    std::accumulate(it->second.bucketCounts.begin(), it->second.bucketCounts.end(),
                                    uint64_t(0)) > 0;
        if (!submitted) { std::this_thread::yield(); }
    }
    // Always unblock before assertions so a regression cannot hang store destruction.
    unblock.set_value();
    EXPECT_EQ(preparing, std::future_status::ready);
    EXPECT_TRUE(submitted);
    ASSERT_TRUE(store->Wait(task.Value()).Success());
    for (auto value : first) { EXPECT_EQ(value, 71); }
    for (auto value : second) { EXPECT_EQ(value, 83); }
}
TEST_F(ContextStoreTest, PartialPrepareFailureDrainsEarlierCopies)
{
    Open(2);
    Observe({1});
    ASSERT_TRUE(Dump(1).Success());
    Observe({2});
    std::array<unsigned char, 64> first{}, second{};
    auto task = store->Load(Detail::TaskDesc{
        Detail::Shard{Id(1), 0, {first.data()} },
        Detail::Shard{Id(2), 0, {second.data()}}
    });
    ASSERT_TRUE(bool(task));
    EXPECT_EQ(store->Wait(task.Value()), Status::NotFound());
    for (auto value : first) { EXPECT_EQ(value, 71); }
    for (auto value : second) { EXPECT_EQ(value, 0); }
    // No leaked local read references after the failed task.
    ASSERT_TRUE(Dump(2).Success());
    Observe({3});
    ASSERT_TRUE(Dump(3).Success());
}
TEST_F(ContextStoreTest, OversizedLoadFailsBeforeEviction)
{
    Open(1);
    Observe({1});
    ASSERT_TRUE(Dump(1).Success());
    Observe({2});
    std::array<unsigned char, 64> first{}, second{};
    auto task = store->Load(Detail::TaskDesc{
        Detail::Shard{Id(1), 0, {first.data()} },
        Detail::Shard{Id(2), 0, {second.data()}}
    });
    ASSERT_FALSE(bool(task));
    EXPECT_EQ(task.Error(), Status::NoSpace());
    EXPECT_NE(task.Error().ToString().find("load requires 2 blocks; Memory holds 1"),
              std::string::npos);
    EXPECT_EQ(store->ContextStats()["evicted_blocks"], 0);
    EXPECT_TRUE(Found(1));
}
TEST_F(ContextStoreTest, MlaReaderCompletionDoesNotBlockNextLayerH2d)
{
    Metrics::SetUp();
    Metrics::CreateStats("context_load_h2d_sync_ms", "histogram", {1000000});
    Detail::Dictionary config;
    config.Set("unique_id", "layers_" + std::to_string(getpid()));
    config.Set<StoreV1*>("store_backend", &backend);
    config.Set("share_buffer_enable", true);
    config.SetNumber("context_tp_size", 2);
    config.SetNumber("block_size", 128);
    config.SetNumber("shard_size", 64);
    config.SetNumber("tensor_size", 64);
    config.SetNumber("context_memory_capacity_bytes", 128);
    config.SetNumber("timeout_ms", 5000);
    config.SetNumber("device_id", 0);
    config.SetNumber("context_tp_rank", 0);
    store.reset(MakeContextStore());
    ASSERT_TRUE(store->Setup(config).Success());
    config.SetNumber("device_id", 1);
    config.SetNumber("context_tp_rank", 1);
    std::unique_ptr<StoreV1> reader(MakeContextStore());
    ASSERT_TRUE(reader->Setup(config).Success());
    Observe({1});
    ASSERT_TRUE(Dump(1, 0, 11).Success());
    ASSERT_TRUE(Dump(1, 1, 22).Success());
    Metrics::GetAllStatsAndClear();
    std::array<std::array<unsigned char, 64>, 2> ownerOut{}, readerOut{};
    std::vector<size_t> ownerTasks;
    for (size_t layer = 0; layer < 2; ++layer) {
        auto task = store->Load({
            Detail::Shard{Id(1), layer, {ownerOut[layer].data()}}
        });
        ASSERT_TRUE(task);
        ownerTasks.push_back(task.Value());
    }
    uint64_t synced = 0;
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (synced < 2 && std::chrono::steady_clock::now() < deadline) {
        auto histograms = std::get<2>(Metrics::GetAllStatsAndClear());
        auto it = histograms.find("context_load_h2d_sync_ms");
        if (it != histograms.end()) {
            synced += std::accumulate(it->second.bucketCounts.begin(),
                                      it->second.bucketCounts.end(), uint64_t(0));
        }
        std::this_thread::yield();
    }
    // Join readers even on a regression, before asserting the overlap result.
    for (size_t layer = 0; layer < 2; ++layer) {
        auto task = reader->Load({
            Detail::Shard{Id(1), layer, {readerOut[layer].data()}}
        });
        ASSERT_TRUE(task);
        EXPECT_TRUE(reader->Wait(task.Value()).Success());
        EXPECT_TRUE(store->Wait(ownerTasks[layer]).Success());
        EXPECT_EQ(ownerOut[layer], readerOut[layer]);
    }
    EXPECT_EQ(synced, 2);
}

TEST_F(ContextStoreTest, SharedMlaLeasePreventsEviction)
{
    Open(1, -1, 1, true);
    Observe({1});
    ASSERT_TRUE(Dump(1).Success());
    Context::SharedMetadata peer;
    ASSERT_TRUE(peer.Setup(name + "_mla", false).Success());
    uint64_t layout = 1469598103934665603ULL;
    for (size_t value : {64, 64, 1, 64}) { layout = (layout ^ uint64_t(value)) * 1099511628211ULL; }
    EXPECT_FALSE(bool(peer.Acquire(Id(1), layout + 1)));
    auto lease = peer.Acquire(Id(1), layout);
    ASSERT_TRUE(bool(lease));
    EXPECT_EQ(lease.Value().slot, 0);
    Observe({2});
    EXPECT_EQ(Dump(2), Status::NoSpace());
    EXPECT_TRUE(Found(1));
    peer.Release(Id(1));
    ASSERT_TRUE(Dump(2).Success());
    lease = peer.Acquire(Id(1), layout);
    EXPECT_FALSE(bool(lease));
    EXPECT_EQ(store->ContextStats()["backend_dump_bytes"], 64);
    ASSERT_TRUE(Load(1).Success());
}

TEST_F(ContextStoreTest, SharedMlaEvictionReservationExcludesReaders)
{
    Context::SharedMetadata owner, peer;
    std::string id = "lease_" + std::to_string(getpid());
    ASSERT_TRUE(owner.Setup(id, true, 8, 123).Success());
    ASSERT_TRUE(peer.Setup(id, false).Success());
    ASSERT_TRUE(owner.Publish(Id(1), 1, 4).Success());
    ASSERT_TRUE(owner.Publish(Id(2), 1, 5).Success());
    ASSERT_TRUE(bool(peer.Acquire(Id(2), 123)));
    EXPECT_FALSE(owner.ReserveEviction({Id(1), Id(2)}));
    // Failed reservation is atomic: it must not leave the first entry busy.
    ASSERT_TRUE(bool(peer.Acquire(Id(1), 123)));
    peer.Release(Id(1));
    peer.Release(Id(2));
    ASSERT_TRUE(owner.ReserveEviction({Id(1), Id(2)}));
    EXPECT_FALSE(bool(peer.Acquire(Id(1), 123)));
    ASSERT_TRUE(owner.Publish(Id(1), 1, 6).Success());
    auto lease = peer.Acquire(Id(1), 123);
    ASSERT_TRUE(bool(lease));
    EXPECT_EQ(lease.Value().slot, 6);
    peer.Release(Id(1));
    owner.Deactivate();
    EXPECT_FALSE(bool(peer.Acquire(Id(1), 123)));
}

TEST_F(ContextStoreTest, WritesOnlyOnEvictionAndReadmitsBackendBytes)
{
    Open();
    Observe({1});
    ASSERT_TRUE(Dump(1).Success());
    Observe({2});
    ASSERT_TRUE(Dump(2).Success());
    EXPECT_EQ(store->ContextStats()["backend_dump_blocks"], 0);
    EXPECT_TRUE(Found(1));
    Observe({3});
    ASSERT_TRUE(Dump(3).Success());
    auto stats = store->ContextStats();
    EXPECT_EQ(stats["backend_dump_blocks"], 1);
    EXPECT_EQ(stats["memory_blocks"], 2);
    ASSERT_TRUE(Load(1).Success());
    EXPECT_EQ(store->ContextStats()["backend_load_blocks"], 1);
    EXPECT_EQ(store->ContextStats()["memory_blocks"], 2);
    // A repeated load hits admitted Memory and does not call the backend again.
    ASSERT_TRUE(Load(1).Success());
    EXPECT_EQ(store->ContextStats()["backend_load_blocks"], 1);
    Observe({4});
    ASSERT_TRUE(Dump(4).Success());
    Observe({5});
    ASSERT_TRUE(Dump(5).Success());
    EXPECT_EQ(store->ContextStats()["backend_dump_skipped_blocks"], 1);
}
TEST_F(ContextStoreTest, DropDoesNotWriteAndMissingLoadFails)
{
    Open(1, 5);
    Observe({1}, 1);
    ASSERT_TRUE(Dump(1).Success());
    Observe({2}, 10);
    ASSERT_TRUE(Dump(2).Success());
    EXPECT_FALSE(Found(1));
    EXPECT_TRUE(Found(2));
    EXPECT_EQ(store->ContextStats()["backend_dump_blocks"], 0);
    EXPECT_EQ(store->ContextStats()["drop_blocks"], 1);
    EXPECT_TRUE(Load(1).Failure());
}
TEST_F(ContextStoreTest, DropAfterReadmissionKeepsExistingBackendRecord)
{
    Open(1, 5);
    Observe({1}, 1);
    ASSERT_TRUE(Dump(1).Success());
    Observe({2}, 2);
    ASSERT_TRUE(Dump(2).Success());
    ASSERT_TRUE(Load(1).Success());
    EXPECT_EQ(store->ContextStats()["backend_dump_blocks"], 2);
    Observe({3}, 20);
    ASSERT_TRUE(Dump(3).Success());
    EXPECT_EQ(store->ContextStats()["drop_blocks"], 1);
    EXPECT_EQ(store->ContextStats()["backend_dump_blocks"], 2);
    EXPECT_TRUE(Found(1));
    EXPECT_TRUE(Found(2));
    EXPECT_TRUE(Found(3));
}
TEST_F(ContextStoreTest, WholeBlockReadinessAndNoSpace)
{
    Open(1, -1, 2);
    Observe({1});
    ASSERT_TRUE(Dump(1, 0).Success());
    EXPECT_FALSE(Found(1));
    EXPECT_TRUE(Load(1).Failure());
    Observe({2});
    EXPECT_EQ(Dump(2), Status::NoSpace());
    ASSERT_TRUE(Dump(1, 1, 83).Success());
    EXPECT_TRUE(Found(1));
    ASSERT_TRUE(Load(1, 0).Success());
    ASSERT_TRUE(Load(1, 1, 83).Success());
    ASSERT_TRUE(Dump(2).Success());
    EXPECT_EQ(store->ContextStats()["backend_dump_bytes"], 128);
    ASSERT_TRUE(Dump(2, 1).Success());
    ASSERT_TRUE(Load(1, 1, 83).Success());
}
TEST_F(ContextStoreTest, BackendReadmissionPreservesAlreadySavedLayers)
{
    Open(1, -1, 2);
    Observe({1});
    ASSERT_TRUE(Dump(1, 0).Success());
    ASSERT_TRUE(Dump(1, 1, 83).Success());
    Observe({2});
    ASSERT_TRUE(Dump(2, 0).Success());
    ASSERT_TRUE(Dump(2, 1).Success());
    // Partial device save of a key that also exists in the backend.
    Observe({1});
    ASSERT_TRUE(Dump(1, 0, 99).Success());
    ASSERT_TRUE(Load(1, 1, 83).Success());
    ASSERT_TRUE(Load(1, 0, 99).Success());
    EXPECT_EQ(store->ContextStats()["backend_load_shards"], 1);
}
TEST_F(ContextStoreTest, BackendFailurePreservesVictimAndReadmissionRetries)
{
    Open(1, -1, 1, true);
    Observe({1});
    ASSERT_TRUE(Dump(1).Success());
    Observe({2});
    backend.failDump = true;
    EXPECT_TRUE(Dump(2).Failure());
    EXPECT_TRUE(Found(1));
    ASSERT_TRUE(Load(1).Success());
    backend.failDump = false;
    ASSERT_TRUE(Dump(2).Success());
    backend.failLoad = true;
    EXPECT_TRUE(Load(1).Failure());
    EXPECT_EQ(store->ContextStats()["memory_blocks"], 0);
    backend.failLoad = false;
    ASSERT_TRUE(Load(1).Success());
    EXPECT_EQ(store->ContextStats()["memory_blocks"], 1);
}
TEST_F(ContextStoreTest, WatcherWorksInAnotherProcessAndOwnerCloses)
{
    Open();
    Observe({1});
    ASSERT_TRUE(Dump(1).Success());
    pid_t child = fork();
    ASSERT_GE(child, 0);
    if (child == 0) {
        Detail::Dictionary config;
        config.Set("unique_id", name);
        config.Set<StoreV1*>("store_backend", &backend);
        std::unique_ptr<StoreV1> peer(MakeContextStore());
        auto setup = peer->Setup(config);
        auto key = Id(1);
        auto hit = peer->Lookup(&key, 1);
        _exit(setup.Success() && hit && hit.Value()[0] ? 0 : 1);
    }
    int status;
    ASSERT_EQ(waitpid(child, &status, 0), child);
    ASSERT_TRUE(WIFEXITED(status));
    ASSERT_EQ(WEXITSTATUS(status), 0);
    EXPECT_TRUE(Found(1));
    store.reset();
    EXPECT_FALSE(Found(1));
}
TEST(ContextIndexTest, CandidateDepthBudgetAndBranchMerge)
{
    Context::ContextIndex tree;
    ASSERT_TRUE(tree.Observe({Id(1), Id(2), Id(3)}, 1).Success());
    tree.Insert(Id(1));
    tree.Insert(Id(2));
    tree.Insert(Id(3));
    ASSERT_TRUE(tree.Observe({Id(4)}, 2).Success());
    tree.Insert(Id(4));
    auto eligible = [](const Key&) { return true; };
    EXPECT_EQ(tree.Select(1, 1, eligible).blocks, std::vector<Key>{Id(3)});
    EXPECT_EQ(tree.Select(4, 1, eligible).blocks, std::vector<Key>{Id(4)});
    ASSERT_TRUE(tree.Observe({Id(1), Id(5)}, 3).Success());
    tree.Insert(Id(5));
    tree.Remove(Id(5));
    tree.Prune(Id(5), [](const Key&) { return false; });
    EXPECT_EQ(tree.Select(4, 2, eligible).blocks, std::vector<Key>{Id(4)});
    tree.Remove(Id(4));
    tree.Prune(Id(4), [](const Key&) { return false; });
    EXPECT_EQ(tree.Select(4, 2, eligible).blocks, (std::vector<Key>{Id(2), Id(3)}));
}
}  // namespace

namespace {
TEST_F(ContextStoreTest, MissingContextAndPartialFailureNeverPublishData)
{
    Open(1, -1, 2);
    EXPECT_TRUE(Dump(1).Failure());
    Observe({1});
    std::array<unsigned char, 64> input{};
    auto invalid = store->Dump(Detail::TaskDesc{
        Detail::Shard{Id(1), 2, {input.data()}}
    });
    EXPECT_FALSE(bool(invalid));
    EXPECT_FALSE(Found(1));
    ASSERT_TRUE(Dump(1, 0).Success());
    ASSERT_TRUE(Dump(1, 1).Success());
    EXPECT_TRUE(Found(1));
}
TEST_F(ContextStoreTest, RetiringRequestDoesNotInvalidateQueuedTransfer)
{
    Open();
    ASSERT_TRUE(store->ObserveRequest("request", 1, 1, {Id(1)}).Success());
    std::array<unsigned char, 64> input{};
    auto task = store->Dump(Detail::TaskDesc{
        Detail::Shard{Id(1), 0, {input.data()}}
    });
    ASSERT_TRUE(bool(task));
    ASSERT_TRUE(store->ObserveRequest("request", 0, 0, {}).Success());
    ASSERT_TRUE(store->Wait(task.Value()).Success());
    EXPECT_TRUE(Found(1));
    EXPECT_EQ(store->ContextStats()["topology_nodes"], 1);
    ASSERT_TRUE(store->ObserveRequest("ghost", 1, 2, {Id(9), Id(10)}).Success());
    ASSERT_TRUE(store->ObserveRequest("ghost", 0, 0, {}).Success());
    EXPECT_EQ(store->ContextStats()["topology_nodes"], 1);
}
TEST_F(ContextStoreTest, WatcherExitDoesNotRemoveOwnerAndLookupDoesNotPin)
{
    Open(1, 0);
    Observe({1}, 1);
    ASSERT_TRUE(Dump(1).Success());
    EXPECT_TRUE(Found(1));
    watcher.reset();
    Detail::Dictionary config;
    config.Set("unique_id", name);
    config.Set<StoreV1*>("store_backend", &backend);
    watcher.reset(MakeContextStore());
    ASSERT_TRUE(watcher->Setup(config).Success());
    EXPECT_TRUE(Found(1));
    Observe({2}, 2);
    ASSERT_TRUE(Dump(2).Success());
    EXPECT_TRUE(Load(1).Failure());
    EXPECT_FALSE(Found(1));
}
TEST(ContextIndexTest, ProtectedTailAndDeterministicTies)
{
    Context::ContextIndex tree;
    ASSERT_TRUE(tree.Observe({Id(1), Id(2)}, 1).Success());
    tree.Insert(Id(1));
    tree.Insert(Id(2));
    ASSERT_TRUE(tree.Observe({Id(3)}, 2).Success());
    tree.Insert(Id(3));
    auto victim = tree.Select(1, 2, [](const Key& id) { return id != Id(2); });
    EXPECT_EQ(victim.blocks, std::vector<Key>{Id(3)});
    auto empty = tree.Select(3, 2, [](const Key&) { return false; });
    EXPECT_TRUE(empty.blocks.empty());
    // A conflicting parent must not modify the tree.
    EXPECT_TRUE(tree.Observe({Id(4), Id(2)}, 3).Failure());
    EXPECT_FALSE(tree.Contains(Id(4)));
}
}  // namespace

namespace {
TEST(ContextIndexTest, IncrementalIndexMatchesTreeScan)
{
    struct Node {
        unsigned parent = 0;
        std::set<unsigned> children;
        bool resident = false;
        uint64_t sequence = 0;
        size_t depth = 1;
    };
    std::map<unsigned, Node> nodes;
    Context::ContextIndex index;
    std::mt19937 random(42);
    uint64_t sequence = 0;
    for (unsigned id = 1; id <= 80; ++id) {
        unsigned parent = random() % id;
        nodes[id].parent = parent;
        if (parent) {
            nodes[parent].children.insert(id);
            nodes[id].depth = nodes[parent].depth + 1;
        }
        std::vector<Key> path;
        for (unsigned n = id; n; n = nodes[n].parent) { path.push_back(Id(n)); }
        std::reverse(path.begin(), path.end());
        ASSERT_TRUE(index.Observe(path, id).Success());
        for (const auto& key : path) {
            nodes[std::to_integer<unsigned>(key[0])].sequence = ++sequence;
        }
        nodes[id].resident = true;
        index.Insert(Id(id));
    }
    for (unsigned step = 0; step < 500; ++step) {
        unsigned touched = 1 + random() % 80;
        std::vector<Key> path;
        for (unsigned n = touched; n; n = nodes[n].parent) { path.push_back(Id(n)); }
        std::reverse(path.begin(), path.end());
        ASSERT_TRUE(index.Observe(path, 81 + step).Success());
        for (const auto& key : path) {
            nodes[std::to_integer<unsigned>(key[0])].sequence = ++sequence;
        }
        if (!nodes[touched].resident) {
            index.Insert(Id(touched));
            nodes[touched].resident = true;
        }
        size_t limit = 1 + random() % 5;
        double budget = 1 + random() % 20;
        struct Candidate {
            unsigned endpoint;
            size_t count;
            uint64_t recent;
            std::vector<Key> blocks;
        };
        std::vector<Candidate> candidates;
        for (const auto& pair : nodes) {
            auto endpoint = pair.first;
            if (pair.second.children.size() == 1) { continue; }
            bool descendant = false;
            for (const auto& member : nodes) {
                if (!member.second.resident) { continue; }
                for (unsigned n = member.second.parent; n; n = nodes[n].parent) {
                    if (n == endpoint) { descendant = true; }
                }
            }
            if (descendant) { continue; }
            Candidate c{endpoint, 0, 0, {}};
            for (unsigned n = endpoint; n;) {
                if (nodes[n].resident) {
                    ++c.count;
                    c.recent = std::max(c.recent, nodes[n].sequence);
                    if (c.blocks.size() < limit) { c.blocks.push_back(Id(n)); }
                }
                auto parent = nodes[n].parent;
                if (!parent || nodes[parent].children.size() != 1) { break; }
                n = parent;
            }
            if (c.count) {
                std::reverse(c.blocks.begin(), c.blocks.end());
                candidates.push_back(c);
            }
        }
        std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
            return std::tie(a.recent, a.endpoint) < std::tie(b.recent, b.endpoint);
        });
        using Score = std::tuple<size_t, size_t, uint64_t, unsigned>;
        std::optional<Score> best;
        std::vector<Key> expected;
        size_t covered = 0;
        for (const auto& c : candidates) {
            Score score{nodes[c.endpoint].depth, c.count, c.recent, c.endpoint};
            if (!best || score < *best) {
                best = score;
                expected = c.blocks;
            }
            covered += c.count;
            if (covered >= budget) { break; }
        }
        auto actual = index.Select(budget, limit, [](const Key&) { return true; });
        ASSERT_EQ(actual.blocks, expected) << "step=" << step;
        for (const auto& key : actual.blocks) {
            nodes[std::to_integer<unsigned>(key[0])].resident = false;
            index.Remove(key);
        }
    }
}
}  // namespace

namespace {
TEST(ContextMetadataTest, ProbeChainsSurviveCollisionsAndChurn)
{
    Context::SharedMetadata metadata;
    ASSERT_TRUE(metadata.Setup("churn_" + std::to_string(getpid()), true, 7, 1).Success());
    std::vector<Key> collisions;
    for (unsigned i = 0; i < 256 && collisions.size() < 5; ++i) {
        if (Detail::BlockIdHasher{}(Id(i)) % 7 == 6) { collisions.push_back(Id(i)); }
    }
    ASSERT_EQ(collisions.size(), 5);
    for (size_t round = 0; round < 100; ++round) {
        for (size_t i = 0; i < collisions.size(); ++i) {
            ASSERT_TRUE(metadata.Publish(collisions[i], 1, i).Success());
        }
        // Pin the entry that deletion will shift across the table boundary.
        ASSERT_TRUE(bool(metadata.Acquire(collisions[4], 1)));
        ASSERT_TRUE(metadata.Publish(collisions[1], 0).Success());
        ASSERT_TRUE(metadata.Publish(collisions[0], 0).Success());
        auto found = metadata.Lookup(collisions.data(), collisions.size());
        ASSERT_TRUE(bool(found));
        EXPECT_EQ(found.Value(), (std::vector<uint8_t>{0, 0, 1, 1, 1}));
        EXPECT_FALSE(metadata.Evictable(collisions[4]));
        metadata.Release(collisions[4]);
        for (size_t i = 2; i < collisions.size(); ++i) {
            auto held = metadata.Acquire(collisions[i], 1);
            ASSERT_TRUE(bool(held));
            EXPECT_EQ(held.Value().slot, i);
            metadata.Release(collisions[i]);
            ASSERT_TRUE(metadata.Publish(collisions[i], 0).Success());
        }
    }
}
TEST(ContextMetadataTest, NotificationsFailuresAndBatchReuse)
{
    Context::SharedMetadata owner, peer;
    auto name = "notify_" + std::to_string(getpid());
    ASSERT_TRUE(owner.Setup(name, true, 7, 123, 2, 2).Success());
    ASSERT_TRUE(peer.Setup(name, false).Success());
    auto reader = peer.BeginLoad(Id(9), 1, 1000);
    auto writer = owner.BeginLoad(Id(9), 0, 1000);
    ASSERT_TRUE(bool(reader));
    ASSERT_TRUE(bool(writer));
    ASSERT_EQ(reader.Value(), writer.Value());
    auto waiting = std::async(std::launch::async,
                              [&] { return peer.WaitAcquire(Id(1), 123, reader.Value()); });
    ASSERT_TRUE(owner.Publish(Id(1), 1, 4).Success());
    auto location = waiting.get();
    ASSERT_TRUE(bool(location));
    EXPECT_EQ(location.Value().slot, 4);
    EXPECT_FALSE(owner.ReserveEviction({Id(1)}));
    peer.Release(Id(1));
    peer.EndLoad(reader.Value(), 1);
    EXPECT_TRUE(owner.WaitReaders(writer.Value()).Success());
    owner.EndLoad(writer.Value(), 0);
    writer = owner.BeginLoad(Id(9), 0, 1000);
    ASSERT_TRUE(bool(writer));
    owner.FailLoad(writer.Value(), Status::NoSpace());
    // A late reader sees the failed batch, not an old ready copy of this key.
    reader = peer.BeginLoad(Id(9), 1, 1000);
    ASSERT_TRUE(bool(reader));
    EXPECT_EQ(reader.Value(), writer.Value());
    auto failed = peer.WaitAcquire(Id(1), 123, reader.Value());
    ASSERT_FALSE(bool(failed));
    EXPECT_EQ(failed.Error(), Status::NoSpace());
    peer.EndLoad(reader.Value(), 1);
    owner.EndLoad(writer.Value(), 0);
    writer = owner.BeginLoad(Id(10), 0, 20);
    ASSERT_TRUE(bool(writer));
    EXPECT_EQ(owner.WaitReaders(writer.Value()), Status::Timeout());
    owner.FailLoad(writer.Value(), Status::Timeout());
    owner.EndLoad(writer.Value(), 0);
}
}  // namespace
