/**
 * MIT License
 *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
 * */
#ifndef UNIFIEDCACHE_MEMORY_STORE_CC_TRANS_BUFFER_H
#define UNIFIEDCACHE_MEMORY_STORE_CC_TRANS_BUFFER_H

#include <list>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include "global_config.h"
#include "status/status.h"

namespace UC::MemoryStore {

class TransBuffer {
    using BlockId = Detail::BlockId;
    using TensorType = Detail::TensorType;

    struct BlockLayerKey {
        BlockId block{};
        size_t layer{0};
        bool operator==(const BlockLayerKey& other) const noexcept
        {
            return block == other.block && layer == other.layer;
        }
    };
    struct ChunkKey {
        BlockId block{};
        size_t layer{0};
        size_t chunkId{0};
        TensorType tensorType{0};
        bool operator==(const ChunkKey& other) const noexcept
        {
            return block == other.block && layer == other.layer && chunkId == other.chunkId &&
                   tensorType == other.tensorType;
        }
    };
    struct BlockLayerHasher {
        size_t operator()(const BlockLayerKey& key) const noexcept;
    };
    struct ChunkHasher {
        size_t operator()(const ChunkKey& key) const noexcept;
    };
    struct Chunk {
        ChunkKey key{};
        size_t payloadSize{0};
        std::vector<std::byte> data{};
        std::vector<uint8_t> ready{};
        std::list<ChunkKey>::iterator lruIt{};
    };

public:
    Status Setup(Config config);
    const Config& GetConfig() const noexcept { return config_; }
    Expected<std::vector<uint8_t>> Lookup(const Detail::BlockId* blocks, size_t num);
    Expected<ssize_t> LookupOnPrefix(const Detail::BlockId* blocks, size_t num);
    Expected<std::vector<uint8_t>> LookupTokens(const Detail::TokenLayerTaskDesc& task);
    Status Load(Detail::TaskDesc& task);
    Status Dump(const Detail::TaskDesc& task);
    Status LoadTokens(Detail::TokenLayerTaskDesc& task);
    Status DumpTokens(const Detail::TokenLayerTaskDesc& task);

private:
    static size_t Sum(const std::vector<size_t>& values);
    static Status CopyFromAddrs(const std::vector<void*>& addrs, const std::vector<size_t>& sizes,
                                std::byte* dst);
    static Status CopyToAddrs(const std::byte* src, const std::vector<size_t>& sizes,
                              const std::vector<void*>& addrs);
    size_t LayerNumber() const noexcept;
    size_t TypePayloadSize(TensorType type) const;
    const std::vector<size_t>& TypeTensorSizes(TensorType type) const;
    Status ValidateTokenKey(const Detail::TokenLayerShard& item) const;
    Status ValidateItem(const Detail::TokenLayerShard& item) const;
    ChunkKey ChunkKeyOf(const Detail::TokenLayerShard& item) const;
    size_t ChunkTokenOffset(size_t tokenOffset) const;
    Chunk& GetOrCreateChunk(const ChunkKey& key);
    void Touch(const ChunkKey& key);
    void EvictOne();
    std::byte* TokenData(const Detail::TokenLayerShard& item, bool create);
    bool TokenReadyNoLock(const Detail::TokenLayerShard& item) const;
    void MarkTokenReady(const Detail::TokenLayerShard& item);
    bool IsFullReadyNoLock(const BlockId& block, size_t layer) const;
    bool UpdateFullReady(const BlockId& block, size_t layer);
    Status SplitFullShard(const BlockId& block, size_t layer, const std::vector<std::byte>& full);
    Status AssembleFullShard(const BlockId& block, size_t layer, std::vector<std::byte>& full);
    Status LoadFullFromBackend(const BlockId& block, size_t layer);
    Status DumpFullToBackend(const BlockId& block, size_t layer, std::vector<std::byte>& full);

private:
    Config config_{};
    mutable std::mutex mutex_{};
    std::list<ChunkKey> lru_;
    std::unordered_map<ChunkKey, Chunk, ChunkHasher> chunks_;
    std::unordered_set<BlockLayerKey, BlockLayerHasher> fullReady_;
    size_t maxChunks_{1};
};

}  // namespace UC::MemoryStore

#endif
