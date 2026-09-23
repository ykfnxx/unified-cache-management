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
#include <sys/mman.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include "context_index.h"
#include "metrics_api.h"
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
    std::function<void(const Detail::TaskDesc&)> beforeDump;
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
        if (beforeDump) { beforeDump(desc); }
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
    ASSERT_TRUE(Load(1, 0).Success());
    ASSERT_TRUE(Load(1, 1).Success());
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

TEST_F(ContextStoreTest, RepeatedObservationRefreshesRecencyAndRetiresReferences)
{
    Open(2, 10);
    ASSERT_TRUE(store->ObserveRequest("a", 1, 1, {Id(1)}).Success());
    ASSERT_TRUE(Dump(1).Success());
    ASSERT_TRUE(store->ObserveRequest("b", 1, 2, {Id(2)}).Success());
    ASSERT_TRUE(Dump(2).Success());
    ASSERT_TRUE(store->ObserveRequest("a", 2, 100, {Id(1)}).Success());
    ASSERT_TRUE(store->ObserveRequest("c", 1, 101, {Id(3)}).Success());
    ASSERT_TRUE(Dump(3).Success());
    EXPECT_TRUE(Found(1));
    EXPECT_FALSE(Found(2));
    EXPECT_EQ(store->ContextStats()["drop_blocks"], 1);
    ASSERT_TRUE(store->ObserveRequest("ghost", 1, 102, {Id(9), Id(10)}).Success());
    auto nodes = store->ContextStats()["topology_nodes"];
    for (size_t i = 2; i < 10; ++i) {
        ASSERT_TRUE(store->ObserveRequest("ghost", i, 102 + i, {Id(9), Id(10)}).Success());
    }
    ASSERT_TRUE(store->ObserveRequest("ghost", 0, 0, {}).Success());
    EXPECT_EQ(store->ContextStats()["topology_nodes"], nodes - 2);
}
TEST(ContextIndexTest, RejectsRepeatedExistingAndNewNodes)
{
    Context::ContextIndex tree;
    ASSERT_TRUE(tree.Observe({Id(1), Id(2)}, 1).Success());
    EXPECT_TRUE(tree.Observe({Id(1), Id(2), Id(1)}, 2).Failure());
    EXPECT_TRUE(tree.Observe({Id(3), Id(4), Id(3)}, 2).Failure());
    EXPECT_EQ(tree.Size(), 2);
    EXPECT_TRUE(tree.Observe({Id(1), Id(2)}, 3).Success());
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
    ASSERT_TRUE(Load(1).Success());
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

TEST_F(ContextStoreTest, DuplicateDumpsDoNotCompleteMissingLayers)
{
    Open(2, -1, 3);
    Observe({1});
    std::array<unsigned char, 64> input;
    input.fill(71);
    auto task = store->Dump(Detail::TaskDesc{
        Detail::Shard{Id(1), 0, {input.data()}},
        Detail::Shard{Id(1), 0, {input.data()}}
    });
    ASSERT_TRUE(task);
    ASSERT_TRUE(store->Wait(task.Value()).Success());
    EXPECT_FALSE(Found(1));
    ASSERT_TRUE(Dump(1, 0).Success());
    ASSERT_TRUE(Dump(1, 1).Success());
    ASSERT_TRUE(Dump(1, 1).Success());
    EXPECT_FALSE(Found(1));
    ASSERT_TRUE(Dump(1, 2).Success());
    EXPECT_TRUE(Found(1));
    EXPECT_EQ(store->ContextStats()["d2h_bytes"], 3 * 64);
    for (size_t layer = 0; layer < 3; ++layer) { ASSERT_TRUE(Load(1, layer).Success()); }
}

TEST_F(ContextStoreTest, ConcurrentObserveLoadAndEviction)
{
    Open(16, -1, 2);
    std::vector<Key> keys;
    for (unsigned key = 1; key <= 64; ++key) { keys.push_back(Id(key)); }
    ASSERT_TRUE(store->ObserveRequest("seed", 1, 1, keys).Success());
    for (unsigned key = 1; key <= 16; ++key) {
        for (size_t layer = 0; layer < 2; ++layer) {
            ASSERT_TRUE(Dump(key, layer, key * 3 + layer).Success());
        }
    }
    std::vector<Key> prefix(keys.begin(), keys.begin() + 8);
    auto observe = std::async(std::launch::async, [&] {
        for (uint64_t step = 2; step < 202; ++step) {
            EXPECT_TRUE(store->ObserveRequest("reads", step, step, prefix).Success());
            std::this_thread::yield();
        }
    });
    auto write = std::async(std::launch::async, [&] {
        for (unsigned key = 17; key <= 64; ++key) {
            for (size_t layer = 0; layer < 2; ++layer) {
                EXPECT_TRUE(Dump(key, layer, key * 3 + layer).Success());
            }
        }
    });
    for (size_t step = 0; step < 40; ++step) {
        std::array<std::array<unsigned char, 64>, 8> output{};
        Detail::TaskDesc desc;
        for (size_t i = 0; i < output.size(); ++i) {
            desc.push_back({Id(i + 1), step % 2, {output[i].data()}});
        }
        auto task = store->Load(std::move(desc));
        ASSERT_TRUE(task);
        ASSERT_TRUE(store->Wait(task.Value()).Success());
        for (size_t i = 0; i < output.size(); ++i) {
            EXPECT_TRUE(std::all_of(output[i].begin(), output[i].end(),
                                    [&](auto v) { return v == (i + 1) * 3 + step % 2; }));
        }
    }
    observe.get();
    write.get();
    EXPECT_EQ(store->ContextStats()["failed_tasks"], 0);
}

TEST_F(ContextStoreTest, NonzeroRankLoadsWithoutOwnerLoadAndOnlyRequestedLayer)
{
    Open(2, -1, 3, true);
    Observe({1});
    for (size_t layer = 0; layer < 3; ++layer) { backend.data[{Id(1), layer}].fill(10 + layer); }
    Detail::Dictionary config;
    config.Set("unique_id", name);
    config.Set<StoreV1*>("store_backend", &backend);
    config.Set("share_buffer_enable", true);
    config.SetNumber("context_tp_size", 16);
    config.SetNumber("context_tp_rank", 15);
    config.SetNumber("device_id", 15);
    config.SetNumber("block_size", 192);
    config.SetNumber("shard_size", 64);
    config.SetNumber("tensor_size", 64);
    config.SetNumber("context_memory_capacity_bytes", 384);
    config.SetNumber("local_rank_size", 16);
    std::unique_ptr<StoreV1> reader(MakeContextStore());
    ASSERT_TRUE(reader->Setup(config).Success());
    std::array<unsigned char, 64> out{};
    auto t = reader->Load({
        {Id(1), 0, {out.data()}}
    });
    ASSERT_TRUE(t);
    ASSERT_TRUE(reader->Wait(t.Value()).Success());
    EXPECT_EQ(backend.loads, 1);
    EXPECT_EQ(out[0], 10);
    EXPECT_EQ(reader->ContextStats()["backend_load_shards"], 1);
    EXPECT_EQ(store->ContextStats()["backend_load_shards"], 0);
    ASSERT_TRUE(Load(1, 0, 10).Success());
    EXPECT_EQ(backend.loads, 1);
    // Reading layer 0 does not pull in layers 1 and 2.
    for (size_t layer = 1; layer < 3; ++layer) {
        auto task = reader->Load({
            {Id(1), layer, {out.data()}}
        });
        ASSERT_TRUE(task);
        ASSERT_TRUE(reader->Wait(task.Value()).Success());
        EXPECT_EQ(out[0], 10 + layer);
    }
    EXPECT_EQ(backend.loads, 3);
}

TEST_F(ContextStoreTest, FailedFillRetriesAndDoesNotConsumeCapacityForever)
{
    Open(1);
    Observe({1, 2, 3});
    backend.data[{Id(1), 0}].fill(71);
    backend.failLoad = true;
    EXPECT_TRUE(Load(1).Failure());
    backend.failLoad = false;
    ASSERT_TRUE(Load(1).Success());
    ASSERT_TRUE(Dump(2).Success());
    backend.failLoad = true;
    EXPECT_TRUE(Load(1).Failure());
    backend.failLoad = false;
    ASSERT_TRUE(Dump(3).Success());
    ASSERT_TRUE(Load(3).Success());
}

TEST_F(ContextStoreTest, Tp16SingleFillPerShardAndIndependentCompletion)
{
    Open(4, -1, 2, true);
    Observe({1, 2, 3, 4});
    for (unsigned key = 1; key <= 4; ++key)
        for (size_t layer = 0; layer < 2; ++layer) {
            backend.data[{Id(key), layer}].fill(key * 10 + layer);
        }
    std::vector<std::unique_ptr<StoreV1>> peers;
    for (size_t rank = 1; rank < 16; ++rank) {
        Detail::Dictionary c;
        c.Set("unique_id", name);
        c.Set<StoreV1*>("store_backend", &backend);
        c.Set("share_buffer_enable", true);
        c.SetNumber("context_tp_rank", rank);
        c.SetNumber("context_tp_size", 16);
        c.SetNumber("device_id", rank);
        c.SetNumber("block_size", 128);
        c.SetNumber("shard_size", 64);
        c.SetNumber("tensor_size", 64);
        c.SetNumber("context_memory_capacity_bytes", 512);
        c.SetNumber("local_rank_size", 16);
        auto peer = std::unique_ptr<StoreV1>(MakeContextStore());
        ASSERT_TRUE(peer->Setup(c).Success());
        peers.push_back(std::move(peer));
    }
    std::promise<void> start;
    auto gate = start.get_future().share();
    std::vector<std::future<void>> jobs;
    for (size_t rank = 0; rank < 16; ++rank) {
        jobs.push_back(std::async(std::launch::async, [&, rank] {
            gate.wait();
            auto* worker = rank ? peers[rank - 1].get() : store.get();
            for (size_t layer = 0; layer < 2; ++layer) {
                std::array<std::array<unsigned char, 64>, 4> out{};
                Detail::TaskDesc desc;
                for (unsigned key = 1; key <= 4; ++key) {
                    desc.push_back({Id(key), layer, {out[key - 1].data()}});
                }
                auto t = worker->Load(std::move(desc));
                ASSERT_TRUE(t);
                ASSERT_TRUE(worker->Wait(t.Value()).Success());
                for (unsigned key = 1; key <= 4; ++key) {
                    EXPECT_TRUE(std::all_of(out[key - 1].begin(), out[key - 1].end(),
                                            [=](auto v) { return v == key * 10 + layer; }));
                }
            }
        }));
    }
    start.set_value();
    for (auto& job : jobs) { job.get(); }
    EXPECT_EQ(backend.loads, 8);
}
TEST_F(ContextStoreTest, LaterSubmitFailureDrainsEarlierBackendWrite)
{
    class DelayedBackend : public TestBackend {
    public:
        std::promise<void> entered, release;
        std::shared_future<void> gate = release.get_future().share();
        std::future<void> pending;
        Expected<size_t> Load(Detail::TaskDesc desc) override
        {
            if (desc[0].owner == Id(2)) { return Status::Error("later shard rejected"); }
            pending = std::async(std::launch::async, [this, desc = std::move(desc)] {
                entered.set_value();
                gate.wait();
                std::memset(desc[0].addrs[0], 71, 64);
            });
            return size_t(1);
        }
        Status Wait(size_t) override
        {
            if (pending.valid()) { pending.get(); }
            return Status::OK();
        }
    } delayed;
    Detail::Dictionary c;
    c.Set("unique_id", "drain_" + std::to_string(getpid()));
    c.Set<StoreV1*>("store_backend", &delayed);
    c.SetNumber("device_id", 0);
    c.SetNumber("block_size", 64);
    c.SetNumber("shard_size", 64);
    c.SetNumber("tensor_size", 64);
    c.SetNumber("context_memory_capacity_bytes", 128);
    c.SetNumber("local_rank_size", 1);
    auto worker = std::unique_ptr<StoreV1>(MakeContextStore());
    ASSERT_TRUE(worker->Setup(c).Success());
    ASSERT_TRUE(worker->ObserveRequest("r", 1, 1, {Id(1), Id(2)}).Success());
    std::array<unsigned char, 128> output{};
    auto task = worker->Load({
        {Id(1), 0, {output.data()}     },
        {Id(2), 0, {output.data() + 64}}
    });
    ASSERT_TRUE(task);
    delayed.entered.get_future().wait();
    auto finished = std::async(std::launch::async, [&] { return worker->Wait(task.Value()); });
    EXPECT_EQ(finished.wait_for(std::chrono::milliseconds(20)), std::future_status::timeout);
    delayed.release.set_value();
    EXPECT_TRUE(finished.get().Failure());
}

TEST_F(ContextStoreTest, DestructionDrainsSubmittedLoads)
{
    Open();
    Observe({1});
    backend.data[{Id(1), 0}].fill(77);
    std::array<unsigned char, 64> output{};
    auto task = store->Load({
        {Id(1), 0, {output.data()}}
    });
    ASSERT_TRUE(task);
    store.reset();
    EXPECT_TRUE(std::all_of(output.begin(), output.end(), [](auto v) { return v == 77; }));
}
}  // namespace
