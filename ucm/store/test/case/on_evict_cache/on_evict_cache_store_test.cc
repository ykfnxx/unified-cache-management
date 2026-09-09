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
#include "on_evict_cache/cc/on_evict_cache_store.cc"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "detail/mock_store.h"
#include "detail/types_helper.h"

namespace {

UC::Detail::TaskDesc Task(const UC::Detail::BlockId& block)
{
    return UC::Detail::TaskDesc{
        UC::Detail::Shard{block, 0, {}}
    };
}

UC::Detail::TaskDesc Task(std::initializer_list<UC::Detail::BlockId> blocks)
{
    UC::Detail::TaskDesc task;
    for (const auto& block : blocks) { task.push_back(UC::Detail::Shard{block, 0, {}}); }
    return task;
}

UC::Detail::RequestAwareDumpContext Context(
    std::initializer_list<UC::Detail::BlockId> requestBlocks,
    std::initializer_list<UC::Detail::BlockId> dumpBlocks)
{
    UC::Detail::RequestAwareDumpContext context;
    context.push_back(UC::Detail::RequestBlockContext{
        std::vector<UC::Detail::BlockId>(requestBlocks),
        std::vector<UC::Detail::BlockId>(dumpBlocks),
    });
    return context;
}

void SetupStore(UC::OnEvictCacheStore::OnEvictCacheStore& store, UC::StoreV1& backend,
                const std::string& policy, size_t capacityBlocks, size_t timeoutSeconds = 0)
{
    UC::Detail::Dictionary config;
    config.Set("store_backend", &backend);
    config.SetNumber("block_size", (size_t(1) << 30) / capacityBlocks);
    config.SetNumber("on_evict_cache_capacity_gb", size_t(1));
    config.Set("on_evict_cache_policy", policy);
    config.SetNumber("on_evict_cache_dump_max_idle_s", timeoutSeconds);
    ASSERT_TRUE(store.Setup(config).Success());
}

TEST(UCOnEvictCacheStoreTest, EvictsLeastRecentlyUsedBlock)
{
    testing::StrictMock<UC::Test::Detail::MockStore> backend;
    UC::OnEvictCacheStore::OnEvictCacheStore store;
    UC::Detail::Dictionary config;
    config.Set("store_backend", static_cast<UC::StoreV1*>(&backend));
    config.SetNumber("block_size", size_t(1) << 29);
    config.SetNumber("on_evict_cache_capacity_gb", size_t(1));
    config.Set("on_evict_cache_policy", std::string("lru"));
    ASSERT_TRUE(store.Setup(config).Success());

    const auto a = UC::Test::Detail::TypesHelper::MakeBlockId("block-a");
    const auto b = UC::Test::Detail::TypesHelper::MakeBlockId("block-b");
    const auto c = UC::Test::Detail::TypesHelper::MakeBlockId("block-c");
    ASSERT_TRUE(store.Dump(Task(a)));
    ASSERT_TRUE(store.Dump(Task(b)));
    ASSERT_TRUE(store.Lookup(&a, 1).Value()[0]);

    EXPECT_CALL(backend, Dump(testing::_))
        .WillOnce(testing::Invoke([&b](UC::Detail::TaskDesc task) {
            EXPECT_EQ(task.size(), size_t{1});
            EXPECT_EQ(task[0].owner, b);
            return UC::Expected<UC::Detail::TaskHandle>(UC::Detail::TaskHandle{7});
        }));
    EXPECT_CALL(backend, Wait(7)).WillOnce(testing::Return(UC::Status::OK()));
    ASSERT_TRUE(store.Dump(Task(c)));

    const UC::Detail::BlockId blocks[]{a, b, c};
    EXPECT_CALL(backend, Lookup(testing::_, 1))
        .WillOnce(testing::Invoke([&b](const UC::Detail::BlockId* blocks, size_t) {
            EXPECT_EQ(blocks[0], b);
            return UC::Expected<std::vector<uint8_t>>(std::vector<uint8_t>{true});
        }));
    EXPECT_EQ(store.Lookup(blocks, 3).Value(), (std::vector<uint8_t>{true, true, true}));
}

TEST(UCOnEvictCacheStoreTest, ExistingBlockDoesNotWriteBackend)
{
    testing::StrictMock<UC::Test::Detail::MockStore> backend;
    UC::OnEvictCacheStore::OnEvictCacheStore store;
    UC::Detail::Dictionary config;
    config.Set("store_backend", static_cast<UC::StoreV1*>(&backend));
    config.SetNumber("block_size", size_t(1) << 30);
    config.SetNumber("on_evict_cache_capacity_gb", size_t(1));
    config.Set("on_evict_cache_policy", std::string("lru"));
    ASSERT_TRUE(store.Setup(config).Success());

    const auto block = UC::Test::Detail::TypesHelper::MakeBlockId("block-a");
    ASSERT_TRUE(store.Dump(Task(block)));
    ASSERT_TRUE(store.Dump(Task(block)));
}

TEST(UCOnEvictCacheStoreTest, EvictsOldestLeafUpToResidentBranch)
{
    testing::StrictMock<UC::Test::Detail::MockStore> backend;
    UC::OnEvictCacheStore::OnEvictCacheStore store;
    SetupStore(store, backend, "radix_lru", 5);

    const auto a = UC::Test::Detail::TypesHelper::MakeBlockId("a");
    const auto b = UC::Test::Detail::TypesHelper::MakeBlockId("b");
    const auto c = UC::Test::Detail::TypesHelper::MakeBlockId("c");
    const auto d = UC::Test::Detail::TypesHelper::MakeBlockId("d");
    const auto e = UC::Test::Detail::TypesHelper::MakeBlockId("e");
    const auto f = UC::Test::Detail::TypesHelper::MakeBlockId("f");

    ASSERT_TRUE(store.Dump(Task({a, b, c, d}), Context({a, b, c, d}, {a, b, c, d})));
    ASSERT_TRUE(store.Dump(Task(e), Context({a, b, e}, {e})));

    {
        testing::InSequence sequence;
        EXPECT_CALL(backend, Dump(testing::_))
            .WillOnce(testing::Invoke([&d](UC::Detail::TaskDesc task) {
                EXPECT_EQ(task[0].owner, d);
                return UC::Expected<UC::Detail::TaskHandle>(UC::Detail::TaskHandle{11});
            }));
        EXPECT_CALL(backend, Wait(11)).WillOnce(testing::Return(UC::Status::OK()));
        EXPECT_CALL(backend, Dump(testing::_))
            .WillOnce(testing::Invoke([&c](UC::Detail::TaskDesc task) {
                EXPECT_EQ(task[0].owner, c);
                return UC::Expected<UC::Detail::TaskHandle>(UC::Detail::TaskHandle{12});
            }));
        EXPECT_CALL(backend, Wait(12)).WillOnce(testing::Return(UC::Status::OK()));
    }
    ASSERT_TRUE(store.Dump(Task(f), Context({f}, {f})));

    const UC::Detail::BlockId resident[]{a, b, e, f};
    EXPECT_EQ(store.Lookup(resident, 4).Value(), (std::vector<uint8_t>{true, true, true, true}));
}

TEST(UCOnEvictCacheStoreTest, StopsBeforeRecentlyObservedAncestor)
{
    testing::StrictMock<UC::Test::Detail::MockStore> backend;
    UC::OnEvictCacheStore::OnEvictCacheStore store;
    SetupStore(store, backend, "radix_lru", 3);

    const auto a = UC::Test::Detail::TypesHelper::MakeBlockId("a");
    const auto b = UC::Test::Detail::TypesHelper::MakeBlockId("b");
    const auto c = UC::Test::Detail::TypesHelper::MakeBlockId("c");
    const auto f = UC::Test::Detail::TypesHelper::MakeBlockId("f");
    ASSERT_TRUE(store.Dump(Task({a, b, c}), Context({a, b, c}, {a, b, c})));

    const UC::Detail::BlockId recentPrefix[]{a, b};
    ASSERT_TRUE(store.ObserveRequest(recentPrefix, 2).Success());
    EXPECT_CALL(backend, Dump(testing::_))
        .WillOnce(testing::Invoke([&c](UC::Detail::TaskDesc task) {
            EXPECT_EQ(task[0].owner, c);
            return UC::Expected<UC::Detail::TaskHandle>(UC::Detail::TaskHandle{21});
        }));
    EXPECT_CALL(backend, Wait(21)).WillOnce(testing::Return(UC::Status::OK()));
    ASSERT_TRUE(store.Dump(Task(f), Context({f}, {f})));

    const UC::Detail::BlockId resident[]{a, b, f};
    EXPECT_EQ(store.Lookup(resident, 3).Value(), (std::vector<uint8_t>{true, true, true}));
}

TEST(UCOnEvictCacheStoreTest, DiscardsExpiredVictimWithoutBackendDump)
{
    testing::StrictMock<UC::Test::Detail::MockStore> backend;
    UC::OnEvictCacheStore::OnEvictCacheStore store;
    SetupStore(store, backend, "radix_lru", 1, 1);

    const auto a = UC::Test::Detail::TypesHelper::MakeBlockId("a");
    const auto b = UC::Test::Detail::TypesHelper::MakeBlockId("b");
    ASSERT_TRUE(store.Dump(Task(a), Context({a}, {a}), 0));
    ASSERT_TRUE(store.Dump(Task(b), Context({b}, {b}), 1'000'000'000));

    EXPECT_EQ(store.Lookup(&b, 1).Value(), (std::vector<uint8_t>{true}));
}

TEST(UCOnEvictCacheStoreTest, LogicalLookupRefreshesVictimAccessTime)
{
    testing::StrictMock<UC::Test::Detail::MockStore> backend;
    UC::OnEvictCacheStore::OnEvictCacheStore store;
    SetupStore(store, backend, "radix_lru", 1, 1);

    const auto a = UC::Test::Detail::TypesHelper::MakeBlockId("a");
    const auto b = UC::Test::Detail::TypesHelper::MakeBlockId("b");
    ASSERT_TRUE(store.Dump(Task(a), Context({a}, {a}), 0));
    EXPECT_EQ(store.LookupOnPrefix(&a, 1, 2'000'000'000).Value(), 0);

    EXPECT_CALL(backend, Dump(testing::_))
        .WillOnce(testing::Invoke([&a](UC::Detail::TaskDesc task) {
            EXPECT_EQ(task[0].owner, a);
            return UC::Expected<UC::Detail::TaskHandle>(UC::Detail::TaskHandle{31});
        }));
    EXPECT_CALL(backend, Wait(31)).WillOnce(testing::Return(UC::Status::OK()));
    ASSERT_TRUE(store.Dump(Task(b), Context({b}, {b}), 2'000'000'000));
}

}  // namespace

#ifdef UCM_TEST_SIMU
#include <array>
#include <cstdlib>
#include <gtest/gtest.h>
#include <sys/wait.h>
#include "detail/types_helper.h"

namespace UC::OnEvictCacheStore {
class OnEvictCacheStoreTestPeer {
public:
    static Status DumpWithStream(OnEvictCacheStore& store, Detail::TaskDesc task,
                                 const Detail::RequestAwareDumpContext& context,
                                 Trans::Stream& stream)
    { return store.DumpKV(task, context, OnEvictCacheStore::Clock::time_point{}, stream); }
    // Exercise eviction with small real payloads instead of allocating GiBs.
    static void SetCapacity(OnEvictCacheStore& store, size_t blocks)
    { store.capacityBlocks_ = blocks; }
    static size_t Capacity(const OnEvictCacheStore& store) { return store.capacityBlocks_; }
    static size_t Resident(const OnEvictCacheStore& store) { return store.residentBlocks_; }
};
}  // namespace UC::OnEvictCacheStore

namespace {
using Store = UC::OnEvictCacheStore::OnEvictCacheStore;
using Peer = UC::OnEvictCacheStore::OnEvictCacheStoreTestPeer;
using BlockId = UC::Detail::BlockId;
using Data = std::array<uint8_t, 16>;

class OnEvictRealTest : public testing::Test {
protected:
    UC::Detail::Dictionary Config(const std::string& policy = "radix_lru")
    {
        UC::Detail::Dictionary config;
        config.Set("unique_id", name);
        config.SetNumber("fake_res_cap", 1);
        config.SetNumber("device_id", 0);
        config.SetNumber("block_size", 32);
        config.SetNumber("shard_size", 16);
        config.SetNumber("tensor_size", 16);
        config.SetNumber("on_evict_cache_dump_max_idle_s", 1);
        config.Set("on_evict_cache_policy", policy);
        return config;
    }
    void SetUp() override
    {
        static size_t sequence = 0;
        name = "real_test_" + std::to_string(getpid()) + "_" + std::to_string(++sequence);
        ASSERT_TRUE(store.Setup(Config()).Success());
        Peer::SetCapacity(store, 2);
    }
    BlockId Id(const char* value) { return UC::Test::Detail::TypesHelper::MakeBlockId(value); }
    UC::Detail::RequestAwareDumpContext Context(std::initializer_list<BlockId> path,
                                                std::initializer_list<BlockId> dump)
    {
        return {
            {std::vector<BlockId>(path), std::vector<BlockId>(dump)}
        };
    }
    UC::Status Dump(BlockId id, uint64_t time, Data& first, Data& second,
                    UC::Detail::RequestAwareDumpContext context)
    {
        UC::Detail::TaskDesc task{
            {id, 0, {first.data()} },
            {id, 1, {second.data()}}
        };
        auto submitted = store.Dump(std::move(task), context, time);
        if (!submitted) { return submitted.Error(); }
        return store.Wait(submitted.Value());
    }
    void ExpectData(Store& reader, BlockId id, const Data& first, const Data& second)
    {
        Data outFirst{}, outSecond{};
        auto task = reader.Load({
            {id, 0, {outFirst.data()} },
            {id, 1, {outSecond.data()}}
        });
        ASSERT_TRUE(task);
        ASSERT_TRUE(reader.Wait(task.Value()).Success());
        EXPECT_EQ(first, outFirst);
        EXPECT_EQ(second, outSecond);
    }
    std::string name;
    Store store;
    Data first{1, 2, 3, 4}, second{5, 6, 7, 8};
};

TEST_F(OnEvictRealTest, CapacityUsesGiB)
{
    Store other;
    ASSERT_TRUE(other.Setup(Config()).Success());
    EXPECT_EQ(Peer::Capacity(other), (size_t{1} << 30) / 32);
}

TEST_F(OnEvictRealTest, PartialBlockStaysInvisibleUntilAllShardsComplete)
{
    auto a = Id("a");
    auto context = Context({a}, {a});
    auto task = store.Dump(
        {
            {a, 1, {second.data()}}
    },
        context, 0);
    ASSERT_TRUE(task);
    ASSERT_TRUE(store.Wait(task.Value()).Success());
    EXPECT_FALSE(store.Lookup(&a, 1).Value()[0]);
    EXPECT_EQ(Peer::Resident(store), size_t{0});
    task = store.Dump(
        {
            {a, 0, {first.data()}}
    },
        context, 0);
    ASSERT_TRUE(task);
    ASSERT_TRUE(store.Wait(task.Value()).Success());
    EXPECT_TRUE(store.Lookup(&a, 1).Value()[0]);
    EXPECT_EQ(Peer::Resident(store), size_t{1});
    ExpectData(store, a, first, second);
}

TEST_F(OnEvictRealTest, EvictedDumpKeepsPayloadAndLoadDoesNotPromote)
{
    Peer::SetCapacity(store, 1);
    auto a = Id("a"), b = Id("b"), c = Id("c");
    ASSERT_TRUE(Dump(a, 0, first, second, Context({a}, {a})).Success());
    ASSERT_TRUE(Dump(b, 0, second, first, Context({b}, {b})).Success());
    ExpectData(store, a, first, second);
    EXPECT_EQ(Peer::Resident(store), size_t{1});
    // a is Dumped and stays recoverable even when b expires and is dropped.
    ASSERT_TRUE(Dump(c, 1'000'000'000, first, second, Context({c}, {c})).Success());
    EXPECT_FALSE(store.Lookup(&b, 1).Value()[0]);
    ExpectData(store, a, first, second);
    Data out{};
    auto missing = store.Load({
        {b, 0, {out.data()}}
    });
    ASSERT_TRUE(missing);
    EXPECT_EQ(store.Wait(missing.Value()), UC::Status::NotFound());
}

TEST_F(OnEvictRealTest, DropAllowsFreshAdmissionWithoutOldPayload)
{
    Peer::SetCapacity(store, 1);
    auto a = Id("a"), b = Id("b");
    ASSERT_TRUE(Dump(a, 0, first, second, Context({a}, {a})).Success());
    ASSERT_TRUE(Dump(b, 1'000'000'000, first, second, Context({b}, {b})).Success());
    ASSERT_TRUE(Dump(a, 2'000'000'000, second, first, Context({a}, {a})).Success());
    ExpectData(store, a, second, first);
}

TEST_F(OnEvictRealTest, ProtectedRequestReturnsNoSpaceWithoutPublishingNewBlock)
{
    Peer::SetCapacity(store, 1);
    auto a = Id("a"), b = Id("b");
    ASSERT_TRUE(Dump(a, 0, first, second, Context({a}, {a})).Success());
    EXPECT_EQ(Dump(b, 0, first, second, Context({a, b}, {b})), UC::Status::NoSpace());
    EXPECT_FALSE(store.Lookup(&b, 1).Value()[0]);
    ExpectData(store, a, first, second);
    // A later unprotected admission can use the completed, unpublished payload.
    ASSERT_TRUE(Dump(b, 0, first, second, Context({b}, {b})).Success());
    ExpectData(store, b, first, second);
}

TEST_F(OnEvictRealTest, AnotherProcessCanLookupAndLoadPublishedKV)
{
    auto a = Id("a");
    if (const auto* sharedName = std::getenv("UCM_ON_EVICT_TEST_READER")) {
        auto config = Config();
        config.Set("unique_id", std::string(sharedName));
        Store reader;
        ASSERT_TRUE(reader.Setup(config).Success());
        ASSERT_TRUE(reader.Lookup(&a, 1).Value()[0]);
        ExpectData(reader, a, first, second);
        return;
    }
    ASSERT_TRUE(Dump(a, 0, first, second, Context({a}, {a})).Success());
    ASSERT_EQ(setenv("UCM_ON_EVICT_TEST_READER", name.c_str(), 1), 0);
    auto pid = fork();
    ASSERT_GE(pid, 0);
    if (pid == 0) {
        // exec avoids inheriting the parent's transfer threads and payload map.
        execl("/proc/self/exe", "ucmstore.test",
              "--gtest_filter=OnEvictRealTest.AnotherProcessCanLookupAndLoadPublishedKV", nullptr);
        _exit(127);
    }
    unsetenv("UCM_ON_EVICT_TEST_READER");
    int status = 0;
    ASSERT_EQ(waitpid(pid, &status, 0), pid);
    ASSERT_TRUE(WIFEXITED(status));
    EXPECT_EQ(WEXITSTATUS(status), 0);
}

TEST_F(OnEvictRealTest, SchedulerSeesDropAndDumpAcrossStoreInstances)
{
    Peer::SetCapacity(store, 1);
    auto config = Config();
    config.SetNumber("device_id", -1);
    Store scheduler;
    ASSERT_TRUE(scheduler.Setup(config).Success());
    auto a = Id("a"), b = Id("b"), c = Id("c");
    ASSERT_TRUE(Dump(a, 0, first, second, Context({a}, {a})).Success());
    ASSERT_TRUE(Dump(b, 0, first, second, Context({b}, {b})).Success());
    ASSERT_TRUE(Dump(c, 1'000'000'000, first, second, Context({c}, {c})).Success());
    BlockId ids[]{a, b, c};
    EXPECT_EQ(scheduler.Lookup(ids, 3).Value(), (std::vector<uint8_t>{1, 0, 1}));
}

TEST_F(OnEvictRealTest, LruBatchRefreshesExistingPayloadBeforeEviction)
{
    Store lru;
    auto config = Config("lru");
    config.Set("unique_id", name + "_batch");
    ASSERT_TRUE(lru.Setup(config).Success());
    Peer::SetCapacity(lru, 1);
    auto a = Id("a"), b = Id("b");
    auto task = lru.Dump(
        {
            {a, 0, {first.data()} },
            {a, 1, {second.data()}}
    },
        {}, 0);
    ASSERT_TRUE(task);
    ASSERT_TRUE(lru.Wait(task.Value()).Success());
    task = lru.Dump(
        {
            {b, 0, {first.data()} },
            {b, 1, {second.data()}},
            {a, 0, {first.data()} },
            {a, 1, {second.data()}}
    },
        {}, 1'000'000'000);
    ASSERT_TRUE(task);
    ASSERT_TRUE(lru.Wait(task.Value()).Success());
    ExpectData(lru, a, first, second);
    ExpectData(lru, b, first, second);
}

TEST_F(OnEvictRealTest, LruAlsoStoresAndDropsRealKV)
{
    Store lru;
    auto config = Config("lru");
    config.Set("unique_id", name + "_lru");
    ASSERT_TRUE(lru.Setup(config).Success());
    Peer::SetCapacity(lru, 1);
    auto a = Id("a"), b = Id("b");
    auto task = lru.Dump(
        {
            {a, 0, {first.data()} },
            {a, 1, {second.data()}}
    },
        {}, 0);
    ASSERT_TRUE(task);
    ASSERT_TRUE(lru.Wait(task.Value()).Success());
    ExpectData(lru, a, first, second);
    task = lru.Dump(
        {
            {b, 0, {first.data()} },
            {b, 1, {second.data()}}
    },
        {}, 1'000'000'000);
    ASSERT_TRUE(task);
    ASSERT_TRUE(lru.Wait(task.Value()).Success());
    EXPECT_FALSE(lru.Lookup(&a, 1).Value()[0]);
}
}  // namespace

TEST(OnEvictTransferTest, CheckWaitAndFailureFollowActualTaskCompletion)
{
    UC::OnEvictCacheStore::TransferQueue queue;
    ASSERT_TRUE(queue.Setup(0, 30000).Success());
    std::promise<void> entered, release;
    auto released = release.get_future();
    auto task = queue.Submit({1,
                              [&](UC::Trans::Stream&) {
                                  entered.set_value();
                                  released.wait();
                                  return UC::Status::NoSpace();
                              },
                              UC::Status::OK()});
    ASSERT_TRUE(task);
    entered.get_future().wait();
    EXPECT_FALSE(queue.Check(task.Value()).Value());
    release.set_value();
    EXPECT_EQ(queue.Wait(task.Value()), UC::Status::NoSpace());
}

#ifdef UCM_TEST_SIMU
#include "trans/simu/simu_stream.h"
namespace {
class FailingStream : public UC::Trans::SimuStream {
public:
    bool failWait{false};
    bool failSync{false};
    uintptr_t waitedEvent{0};
    UC::Status WaitEvent(const UC::Trans::Event& event) override
    {
        waitedEvent = event.NativeHandle();
        return failWait ? UC::Status::Error("wait failed") : UC::Status::OK();
    }
    UC::Status Synchronized() override
    {
        auto status = SimuStream::Synchronized();
        return failSync ? UC::Status::Error("sync failed") : status;
    }
};

TEST_F(OnEvictRealTest, PrerequisiteAndCopyFailureDoNotPublishKV)
{
    FailingStream stream;
    ASSERT_TRUE(stream.Setup().Success());
    auto a = Id("a");
    UC::Detail::TaskDesc task{
        {a, 0, {first.data()} },
        {a, 1, {second.data()}}
    };
    task.prerequisiteHandle = 123;
    stream.failWait = true;
    EXPECT_TRUE(Peer::DumpWithStream(store, task, Context({a}, {a}), stream).Failure());
    EXPECT_EQ(stream.waitedEvent, uintptr_t{123});
    EXPECT_FALSE(store.Lookup(&a, 1).Value()[0]);
    stream.failWait = false;
    stream.failSync = true;
    EXPECT_TRUE(Peer::DumpWithStream(store, task, Context({a}, {a}), stream).Failure());
    EXPECT_FALSE(store.Lookup(&a, 1).Value()[0]);
    // Retry must copy again after a failed synchronization.
    first.fill(42);
    stream.failSync = false;
    ASSERT_TRUE(Peer::DumpWithStream(store, task, Context({a}, {a}), stream).Success());
    ExpectData(store, a, first, second);
}
}  // namespace
#endif

#endif  // UCM_TEST_SIMU
