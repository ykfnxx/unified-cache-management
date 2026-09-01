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
#include <chrono>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <thread>
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
    ASSERT_TRUE(store.Dump(Task(a), Context({a}, {a})));
    std::this_thread::sleep_for(std::chrono::milliseconds(1100));
    ASSERT_TRUE(store.Dump(Task(b), Context({b}, {b})));

    EXPECT_EQ(store.Lookup(&b, 1).Value(), (std::vector<uint8_t>{true}));
}

}  // namespace
