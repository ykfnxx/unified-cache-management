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
    Status Setup(const Config& config);
    Expected<std::vector<uint8_t>> Lookup(const Detail::BlockId* blocks, size_t num);
    Expected<ssize_t> LookupOnPrefix(const Detail::BlockId* blocks, size_t num);
    Expected<std::vector<uint8_t>> LookupTokens(const Detail::TokenLayerTaskDesc& task);
    Status Load(Detail::TaskDesc& task);
    Status Dump(const Detail::TaskDesc& task);
    Status LoadTokens(Detail::TokenLayerTaskDesc& task);
    Status DumpTokens(const Detail::TokenLayerTaskDesc& task);
    Expected<size_t> PhysicalShardIndex(const Detail::TokenLayerShard& item) const;
    Status ReadFull(const BlockId& block, size_t layer, std::vector<std::byte>& full);
    Status ReadToken(const Detail::TokenLayerShard& item, std::vector<std::byte>& data);
    Status CommitFull(const BlockId& block, size_t layer, const std::vector<std::byte>& full);
    Status CommitToken(const Detail::TokenLayerShard& item, const std::vector<std::byte>& data,
                       std::vector<std::byte>* full = nullptr, size_t* physicalShard = nullptr);

private:
    enum class LayoutMode : uint8_t { ORDINARY, LAYERWISE };
    static size_t Sum(const std::vector<size_t>& values);
    static Status CopyFromAddrs(const std::vector<void*>& addrs, const std::vector<size_t>& sizes,
                                std::byte* dst);
    static Status CopyToAddrs(const std::byte* src, const std::vector<size_t>& sizes,
                              const std::vector<void*>& addrs);
    Status SetupLayout();
    Status ValidatePhysicalShard(size_t physicalShard) const;
    size_t LayerNumber() const noexcept;
    size_t PhysicalShardIndexNoCheck(size_t layer, TensorType type) const;
    Status DecodeLayerwiseShard(size_t physicalShard, size_t& layer, TensorType& type) const;
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
    bool UpdateFullReady(const BlockId& block, size_t physicalShard);
    Status SplitFullShard(const BlockId& block, size_t layer, const std::vector<std::byte>& full);
    Status AssembleFullShard(const BlockId& block, size_t layer, std::vector<std::byte>& full);
    Status LoadFullFromBackend(const BlockId& block, size_t layer);
    Status DumpFullToBackend(const BlockId& block, size_t layer, std::vector<std::byte>& full);

private:
    StoreV1* storeBackend_{nullptr};
    size_t shardSize_{0};
    size_t blockSize_{0};
    std::vector<size_t> tensorSizeList_{};
    size_t memoryTokenChunkSize_{16};
    std::vector<Detail::TensorType> requiredTensorTypes_{};
    std::unordered_map<Detail::TensorType, std::vector<size_t>> tensorSizesByType_{};
    size_t tokensPerBlock_{0};
    LayoutMode layoutMode_{LayoutMode::ORDINARY};
    size_t physicalShardNumber_{1};
    size_t logicalLayerNumber_{1};
    std::unordered_map<TensorType, size_t> tensorTypeRank_{};
    mutable std::mutex mutex_{};
    std::list<ChunkKey> lru_;
    std::unordered_map<ChunkKey, Chunk, ChunkHasher> chunks_;
    std::unordered_set<BlockLayerKey, BlockLayerHasher> fullReady_;
    size_t maxChunks_{1};
};

}  // namespace UC::MemoryStore

#endif
