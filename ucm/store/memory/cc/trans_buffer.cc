/**
 * MIT License
 *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
 * */
#include "trans_buffer.h"
#include <algorithm>
#include <cstring>
#include <numeric>
#include "logger/logger.h"

namespace UC::MemoryStore {

namespace {
template <typename T>
void HashCombine(size_t& seed, const T& value)
{
    constexpr auto golden = 0x9e3779b97f4a7c15ULL;
    seed ^= std::hash<T>{}(value) + golden + (seed << 6) + (seed >> 2);
}
}  // namespace

size_t TransBuffer::BlockLayerHasher::operator()(const BlockLayerKey& key) const noexcept
{
    size_t seed = Detail::BlockIdHasher{}(key.block);
    HashCombine(seed, key.layer);
    return seed;
}

size_t TransBuffer::ChunkHasher::operator()(const ChunkKey& key) const noexcept
{
    size_t seed = Detail::BlockIdHasher{}(key.block);
    HashCombine(seed, key.layer);
    HashCombine(seed, key.chunkId);
    HashCombine(seed, key.tensorType);
    return seed;
}

Status TransBuffer::Setup(Config config)
{
    config_ = std::move(config);
    size_t maxPayload = 1;
    for (const auto& [_, sizes] : config_.tensorSizesByType) {
        maxPayload = std::max(maxPayload, Sum(sizes));
    }
    const auto chunkBytes = std::max<size_t>(1, maxPayload * config_.memoryTokenChunkSize);
    maxChunks_ = std::max<size_t>(1, config_.memoryBufferCapacity / chunkBytes);
    UC_INFO("MemoryStore buffer setup with tokens_per_block={}, chunk_tokens={}, chunks={}.",
            config_.tokensPerBlock, config_.memoryTokenChunkSize, maxChunks_);
    return Status::OK();
}

Expected<std::vector<uint8_t>> TransBuffer::Lookup(const Detail::BlockId* blocks, size_t num)
{
    std::vector<uint8_t> result;
    result.reserve(num);
    std::lock_guard<std::mutex> guard(mutex_);
    for (size_t i = 0; i < num; ++i) {
        bool hit = false;
        for (size_t layer = 0; layer < LayerNumber(); ++layer) {
            if (IsFullReadyNoLock(blocks[i], layer)) {
                hit = true;
                break;
            }
        }
        result.push_back(hit);
    }
    return result;
}

Expected<ssize_t> TransBuffer::LookupOnPrefix(const Detail::BlockId* blocks, size_t num)
{
    auto res = Lookup(blocks, num);
    if (!res) { return res.Error(); }
    const auto& hits = res.Value();
    ssize_t idx = -1;
    for (size_t i = 0; i < hits.size(); ++i) {
        if (!hits[i]) { break; }
        idx = static_cast<ssize_t>(i);
    }
    return idx;
}

Expected<std::vector<uint8_t>> TransBuffer::LookupTokens(const Detail::TokenLayerTaskDesc& task)
{
    std::vector<uint8_t> result;
    result.reserve(task.size());
    std::lock_guard<std::mutex> guard(mutex_);
    for (const auto& item : task) { result.push_back(TokenReadyNoLock(item)); }
    return result;
}

Status TransBuffer::Load(Detail::TaskDesc& task)
{
    std::lock_guard<std::mutex> guard(mutex_);
    for (const auto& shard : task) {
        if (!IsFullReadyNoLock(shard.owner, shard.index)) {
            auto s = LoadFullFromBackend(shard.owner, shard.index);
            if (s.Failure()) { return s; }
        }
        std::vector<std::byte> full;
        auto s = AssembleFullShard(shard.owner, shard.index, full);
        if (s.Failure()) { return s; }
        s = CopyToAddrs(full.data(), config_.tensorSizes, shard.addrs);
        if (s.Failure()) { return s; }
    }
    return Status::OK();
}

Status TransBuffer::Dump(const Detail::TaskDesc& task)
{
    std::lock_guard<std::mutex> guard(mutex_);
    for (const auto& shard : task) {
        std::vector<std::byte> full(config_.shardSize);
        auto s = CopyFromAddrs(shard.addrs, config_.tensorSizes, full.data());
        if (s.Failure()) { return s; }
        s = SplitFullShard(shard.owner, shard.index, full);
        if (s.Failure()) { return s; }
        fullReady_.insert({shard.owner, shard.index});
        s = DumpFullToBackend(shard.owner, shard.index, full);
        if (s.Failure()) { return s; }
    }
    return Status::OK();
}

Status TransBuffer::LoadTokens(Detail::TokenLayerTaskDesc& task)
{
    std::lock_guard<std::mutex> guard(mutex_);
    for (const auto& item : task) {
        if (!TokenReadyNoLock(item)) {
            auto s = LoadFullFromBackend(item.owner, item.layer);
            if (s.Failure()) { return s; }
        }
        auto* p = TokenData(item, false);
        auto s = CopyToAddrs(p, TypeTensorSizes(item.tensorType), item.addrs);
        if (s.Failure()) { return s; }
        Touch(ChunkKeyOf(item));
    }
    return Status::OK();
}

Status TransBuffer::DumpTokens(const Detail::TokenLayerTaskDesc& task)
{
    std::lock_guard<std::mutex> guard(mutex_);
    for (const auto& item : task) {
        auto s = ValidateItem(item);
        if (s.Failure()) { return s; }
        auto* p = TokenData(item, true);
        s = CopyFromAddrs(item.addrs, TypeTensorSizes(item.tensorType), p);
        if (s.Failure()) { return s; }
        MarkTokenReady(item);
        if (!UpdateFullReady(item.owner, item.layer)) { continue; }
        std::vector<std::byte> full;
        s = AssembleFullShard(item.owner, item.layer, full);
        if (s.Failure()) { return s; }
        s = DumpFullToBackend(item.owner, item.layer, full);
        if (s.Failure()) { return s; }
    }
    return Status::OK();
}

size_t TransBuffer::Sum(const std::vector<size_t>& values)
{
    return std::accumulate(values.begin(), values.end(), size_t{0});
}

Status TransBuffer::CopyFromAddrs(const std::vector<void*>& addrs,
                                  const std::vector<size_t>& sizes, std::byte* dst)
{
    if (addrs.size() != sizes.size()) {
        return Status::InvalidParam("invalid source addr number({},{})", addrs.size(),
                                    sizes.size());
    }
    size_t offset = 0;
    for (size_t i = 0; i < addrs.size(); ++i) {
        if (!addrs[i]) { return Status::InvalidParam("invalid source addr"); }
        std::memcpy(dst + offset, addrs[i], sizes[i]);
        offset += sizes[i];
    }
    return Status::OK();
}

Status TransBuffer::CopyToAddrs(const std::byte* src, const std::vector<size_t>& sizes,
                                const std::vector<void*>& addrs)
{
    if (addrs.size() != sizes.size()) {
        return Status::InvalidParam("invalid destination addr number({},{})", addrs.size(),
                                    sizes.size());
    }
    size_t offset = 0;
    for (size_t i = 0; i < addrs.size(); ++i) {
        if (!addrs[i]) { return Status::InvalidParam("invalid destination addr"); }
        std::memcpy(addrs[i], src + offset, sizes[i]);
        offset += sizes[i];
    }
    return Status::OK();
}

size_t TransBuffer::LayerNumber() const noexcept { return config_.blockSize / config_.shardSize; }

size_t TransBuffer::TypePayloadSize(TensorType type) const
{
    auto iter = config_.tensorSizesByType.find(type);
    if (iter == config_.tensorSizesByType.end()) { return 0; }
    return Sum(iter->second);
}

const std::vector<size_t>& TransBuffer::TypeTensorSizes(TensorType type) const
{
    return config_.tensorSizesByType.at(type);
}

Status TransBuffer::ValidateTokenKey(const Detail::TokenLayerShard& item) const
{
    if (config_.tensorSizesByType.find(item.tensorType) == config_.tensorSizesByType.end()) {
        return Status::InvalidParam("invalid tensor type({})", item.tensorType);
    }
    if (item.layer >= LayerNumber()) {
        return Status::InvalidParam("invalid layer({})", item.layer);
    }
    if (item.tokenOffset >= config_.tokensPerBlock) {
        return Status::InvalidParam("invalid token offset({})", item.tokenOffset);
    }
    return Status::OK();
}

Status TransBuffer::ValidateItem(const Detail::TokenLayerShard& item) const
{
    auto s = ValidateTokenKey(item);
    if (s.Failure()) { return s; }
    const auto& sizes = TypeTensorSizes(item.tensorType);
    if (item.addrs.size() != sizes.size()) {
        return Status::InvalidParam("invalid addr number({},{})", item.addrs.size(),
                                    sizes.size());
    }
    return Status::OK();
}

TransBuffer::ChunkKey TransBuffer::ChunkKeyOf(const Detail::TokenLayerShard& item) const
{
    return {item.owner, item.layer, item.tokenOffset / config_.memoryTokenChunkSize,
            item.tensorType};
}

size_t TransBuffer::ChunkTokenOffset(size_t tokenOffset) const
{
    return tokenOffset % config_.memoryTokenChunkSize;
}

TransBuffer::Chunk& TransBuffer::GetOrCreateChunk(const ChunkKey& key)
{
    auto iter = chunks_.find(key);
    if (iter != chunks_.end()) {
        Touch(key);
        return iter->second;
    }
    while (chunks_.size() >= maxChunks_) { EvictOne(); }
    auto lruIt = lru_.insert(lru_.begin(), key);
    Chunk chunk;
    chunk.key = key;
    chunk.payloadSize = TypePayloadSize(key.tensorType);
    chunk.data.resize(chunk.payloadSize * config_.memoryTokenChunkSize);
    chunk.ready.assign(config_.memoryTokenChunkSize, false);
    chunk.lruIt = lruIt;
    auto [inserted, _] = chunks_.emplace(key, std::move(chunk));
    return inserted->second;
}

void TransBuffer::Touch(const ChunkKey& key)
{
    auto iter = chunks_.find(key);
    if (iter == chunks_.end()) { return; }
    lru_.splice(lru_.begin(), lru_, iter->second.lruIt);
    iter->second.lruIt = lru_.begin();
}

void TransBuffer::EvictOne()
{
    if (lru_.empty()) { return; }
    auto key = lru_.back();
    lru_.pop_back();
    chunks_.erase(key);
    fullReady_.erase({key.block, key.layer});
}

std::byte* TransBuffer::TokenData(const Detail::TokenLayerShard& item, bool create)
{
    auto key = ChunkKeyOf(item);
    auto iter = chunks_.find(key);
    if (iter == chunks_.end()) {
        if (!create) { return nullptr; }
        auto& chunk = GetOrCreateChunk(key);
        iter = chunks_.find(chunk.key);
    }
    Touch(key);
    auto& chunk = iter->second;
    return chunk.data.data() + ChunkTokenOffset(item.tokenOffset) * chunk.payloadSize;
}

bool TransBuffer::TokenReadyNoLock(const Detail::TokenLayerShard& item) const
{
    if (ValidateTokenKey(item).Failure()) { return false; }
    auto iter = chunks_.find(ChunkKeyOf(item));
    if (iter == chunks_.end()) { return false; }
    return iter->second.ready[ChunkTokenOffset(item.tokenOffset)];
}

void TransBuffer::MarkTokenReady(const Detail::TokenLayerShard& item)
{
    auto& chunk = GetOrCreateChunk(ChunkKeyOf(item));
    chunk.ready[ChunkTokenOffset(item.tokenOffset)] = true;
}

bool TransBuffer::IsFullReadyNoLock(const BlockId& block, size_t layer) const
{
    return fullReady_.count({block, layer}) > 0;
}

bool TransBuffer::UpdateFullReady(const BlockId& block, size_t layer)
{
    for (size_t token = 0; token < config_.tokensPerBlock; ++token) {
        for (const auto type : config_.requiredTensorTypes) {
            Detail::TokenLayerShard item{block, layer, token, type, {}};
            if (!TokenReadyNoLock(item)) { return false; }
        }
    }
    fullReady_.insert({block, layer});
    return true;
}

Status TransBuffer::SplitFullShard(const BlockId& block, size_t layer,
                                   const std::vector<std::byte>& full)
{
    if (full.size() != config_.shardSize) { return Status::InvalidParam("invalid shard"); }
    size_t offset = 0;
    for (size_t token = 0; token < config_.tokensPerBlock; ++token) {
        for (const auto type : config_.requiredTensorTypes) {
            Detail::TokenLayerShard item{block, layer, token, type, {}};
            auto* p = TokenData(item, true);
            const auto size = TypePayloadSize(type);
            std::memcpy(p, full.data() + offset, size);
            MarkTokenReady(item);
            offset += size;
        }
    }
    return Status::OK();
}

Status TransBuffer::AssembleFullShard(const BlockId& block, size_t layer,
                                      std::vector<std::byte>& full)
{
    full.resize(config_.shardSize);
    size_t offset = 0;
    for (size_t token = 0; token < config_.tokensPerBlock; ++token) {
        for (const auto type : config_.requiredTensorTypes) {
            Detail::TokenLayerShard item{block, layer, token, type, {}};
            if (!TokenReadyNoLock(item)) { return Status::NotFound(); }
            auto* p = TokenData(item, false);
            const auto size = TypePayloadSize(type);
            std::memcpy(full.data() + offset, p, size);
            offset += size;
        }
    }
    return Status::OK();
}

Status TransBuffer::LoadFullFromBackend(const BlockId& block, size_t layer)
{
    if (!config_.storeBackend) { return Status::NotFound(); }
    std::vector<std::byte> full(config_.shardSize);
    Detail::TaskDesc backendTask;
    backendTask.brief = "Backend2Memory";
    backendTask.push_back(Detail::Shard{block, layer, {full.data()}});
    auto res = config_.storeBackend->Load(std::move(backendTask));
    if (!res) { return res.Error(); }
    auto s = config_.storeBackend->Wait(res.Value());
    if (s.Failure()) { return s; }
    s = SplitFullShard(block, layer, full);
    if (s.Failure()) { return s; }
    fullReady_.insert({block, layer});
    return Status::OK();
}

Status TransBuffer::DumpFullToBackend(const BlockId& block, size_t layer,
                                      std::vector<std::byte>& full)
{
    if (!config_.storeBackend) { return Status::OK(); }
    Detail::TaskDesc backendTask;
    backendTask.brief = "Memory2Backend";
    backendTask.push_back(Detail::Shard{block, layer, {full.data()}});
    auto res = config_.storeBackend->Dump(std::move(backendTask));
    if (!res) { return res.Error(); }
    return config_.storeBackend->Wait(res.Value());
}

}  // namespace UC::MemoryStore
