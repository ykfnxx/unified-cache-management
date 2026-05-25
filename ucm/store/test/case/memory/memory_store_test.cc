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
#include <gtest/gtest.h>
#include <memory>
#include <type_traits>
#include "cache/cc/copy_stream.h"
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

TEST(UCMMemoryStoreStructureTest, CopyStreamMovesThroughTransferStream)
{
    UC::CacheStore::CopyStream copyStream;
    ASSERT_EQ(copyStream.Setup(0, 1, false), UC::Status::OK());

    std::array<std::byte, 4> src{std::byte{9}, std::byte{8}, std::byte{7}, std::byte{6}};
    std::array<std::byte, 4> dst{};
    auto stream = copyStream.NextStream();
    ASSERT_NE(stream, nullptr);
    ASSERT_EQ(stream->HostToDeviceAsync(src.data(), dst.data(), src.size()), UC::Status::OK());
    ASSERT_EQ(copyStream.Synchronize(), UC::Status::OK());
    EXPECT_EQ(std::memcmp(src.data(), dst.data(), src.size()), 0);
}

TEST(UCMMemoryStoreStructureTest, TransBufferSetupDoesNotTakeConfigByValue)
{
    using SetupSignature = decltype(&UC::MemoryStore::TransBuffer::Setup);
    using ExpectedSignature =
        UC::Status (UC::MemoryStore::TransBuffer::*)(const UC::MemoryStore::Config&);
    EXPECT_TRUE((std::is_same_v<SetupSignature, ExpectedSignature>));
}
