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
#include <gtest/gtest.h>
#include <random>
#include <sys/wait.h>
#include <unistd.h>
#include "context_index.h"
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
class ContextStoreTest : public ::testing::Test {
protected:
    std::unique_ptr<StoreV1> store, watcher;
    std::string name;
    uint64_t clock = 0;
    void Open(size_t memory = 2, size_t ssd = 8, int64_t retention = -1, size_t shards = 1,
              bool shared = false)
    {
        static unsigned next = 0;
        name = "test_" + std::to_string(getpid()) + "_" + std::to_string(++next);
        Detail::Dictionary config;
        config.Set("unique_id", name);
        config.Set("share_buffer_enable", shared);
        watcher.reset(MakeContextStore());
        ASSERT_TRUE(watcher->Setup(config).Success());
        config.SetNumber("device_id", 0);
        config.SetNumber("block_size", 64 * shards);
        config.SetNumber("shard_size", 64);
        config.SetNumber("tensor_size", 64);
        config.SetNumber("context_memory_capacity_bytes", memory * 64 * shards);
        config.SetNumber("context_simulated_ssd_capacity_bytes", ssd * 64 * shards);
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
TEST_F(ContextStoreTest, SharedMlaLeasePreventsEviction)
{
    Open(1, 8, -1, 1, true);
    Observe({1});
    ASSERT_TRUE(Dump(1).Success());
    Context::SharedMetadata peer;
    ASSERT_TRUE(peer.Setup(name + "_mla", false).Success());
    uint64_t layout = 1469598103934665603ULL;
    for (size_t value : {64, 64, 1, 8, 64}) {
        layout = (layout ^ uint64_t(value)) * 1099511628211ULL;
    }
    EXPECT_FALSE(bool(peer.Acquire(Id(1), layout + 1)));
    auto lease = peer.Acquire(Id(1), layout);
    ASSERT_TRUE(bool(lease));
    EXPECT_TRUE(lease.Value().memory);
    Observe({2});
    EXPECT_EQ(Dump(2), Status::NoSpace());
    EXPECT_TRUE(Found(1));
    peer.Release(Id(1));
    ASSERT_TRUE(Dump(2).Success());
    lease = peer.Acquire(Id(1), layout);
    ASSERT_TRUE(bool(lease));
    EXPECT_FALSE(lease.Value().memory);
    peer.Release(Id(1));
    EXPECT_EQ(store->ContextStats()["ssd_write_bytes"], 64);
    ASSERT_TRUE(Load(1).Success());
}

TEST_F(ContextStoreTest, SharedMlaEvictionReservationExcludesReaders)
{
    Context::SharedMetadata owner, peer;
    std::string id = "lease_" + std::to_string(getpid());
    ASSERT_TRUE(owner.Setup(id, true, 8, 123).Success());
    ASSERT_TRUE(peer.Setup(id, false).Success());
    ASSERT_TRUE(owner.Publish(Id(1), 1, 4, 0).Success());
    ASSERT_TRUE(owner.Publish(Id(2), 1, 5, 0).Success());
    ASSERT_TRUE(bool(peer.Acquire(Id(2), 123)));
    EXPECT_FALSE(owner.ReserveEviction({Id(1), Id(2)}));
    // Failed reservation is atomic: it must not leave the first entry busy.
    ASSERT_TRUE(bool(peer.Acquire(Id(1), 123)));
    peer.Release(Id(1));
    peer.Release(Id(2));
    ASSERT_TRUE(owner.ReserveEviction({Id(1), Id(2)}));
    EXPECT_FALSE(bool(peer.Acquire(Id(1), 123)));
    ASSERT_TRUE(owner.Publish(Id(1), 2, 0, 6).Success());
    auto lease = peer.Acquire(Id(1), 123);
    ASSERT_TRUE(bool(lease));
    EXPECT_FALSE(lease.Value().memory);
    EXPECT_EQ(lease.Value().slot, 6);
    peer.Release(Id(1));
    owner.Deactivate();
    EXPECT_FALSE(bool(peer.Acquire(Id(1), 123)));
}

TEST_F(ContextStoreTest, WritesOnlyOnEvictionAndLoadsSsdBytes)
{
    Open();
    Observe({1});
    ASSERT_TRUE(Dump(1).Success());
    Observe({2});
    ASSERT_TRUE(Dump(2).Success());
    EXPECT_EQ(store->ContextStats()["ssd_write_blocks"], 0);
    EXPECT_TRUE(Found(1));
    Observe({3});
    ASSERT_TRUE(Dump(3).Success());
    auto stats = store->ContextStats();
    EXPECT_EQ(stats["ssd_write_blocks"], 1);
    EXPECT_EQ(stats["memory_blocks"], 2);
    ASSERT_TRUE(Load(1).Success());
    EXPECT_EQ(store->ContextStats()["ssd_read_bytes"], 64);
    EXPECT_EQ(store->ContextStats()["memory_blocks"], 2);
    // Saving an SSD-resident key admits memory, but cannot duplicate its SSD write.
    Observe({1});
    ASSERT_TRUE(Dump(1).Success());
    auto before = store->ContextStats()["ssd_write_blocks"];
    Observe({4});
    ASSERT_TRUE(Dump(4).Success());
    Observe({5});
    ASSERT_TRUE(Dump(5).Success());
    EXPECT_EQ(store->ContextStats()["ssd_write_skipped_blocks"], 1);
    EXPECT_EQ(store->ContextStats()["ssd_write_blocks"], before + 1);
}
TEST_F(ContextStoreTest, DropDoesNotWriteAndMissingLoadFails)
{
    Open(1, 2, 5);
    Observe({1}, 1);
    ASSERT_TRUE(Dump(1).Success());
    Observe({2}, 10);
    ASSERT_TRUE(Dump(2).Success());
    EXPECT_FALSE(Found(1));
    EXPECT_TRUE(Found(2));
    EXPECT_EQ(store->ContextStats()["ssd_write_blocks"], 0);
    EXPECT_EQ(store->ContextStats()["drop_blocks"], 1);
    EXPECT_TRUE(Load(1).Failure());
}
TEST_F(ContextStoreTest, WholeBlockReadinessAndNoSpace)
{
    Open(1, 2, -1, 2);
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
    EXPECT_EQ(store->ContextStats()["ssd_write_bytes"], 128);
    ASSERT_TRUE(Load(1, 1, 83).Success());
}
TEST_F(ContextStoreTest, FullSsdPreservesVictim)
{
    Open(1, 1);
    Observe({1});
    ASSERT_TRUE(Dump(1).Success());
    Observe({2});
    ASSERT_TRUE(Dump(2).Success());
    Observe({3});
    EXPECT_EQ(Dump(3), Status::NoSpace());
    EXPECT_TRUE(Found(1));
    EXPECT_TRUE(Found(2));
    EXPECT_FALSE(Found(3));
    ASSERT_TRUE(Load(1).Success());
    ASSERT_TRUE(Load(2).Success());
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
    Open(1, 2, -1, 2);
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
    Open(1, 2, 0);
    Observe({1}, 1);
    ASSERT_TRUE(Dump(1).Success());
    EXPECT_TRUE(Found(1));
    watcher.reset();
    Detail::Dictionary config;
    config.Set("unique_id", name);
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
