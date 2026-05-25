/**
 * MIT License
 *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
 * */
#include "memory_store.h"
#include <numeric>
#include "logger/logger.h"

namespace UC::MemoryStore {

Status MemoryStore::Setup(const Config& inConfig)
{
    auto config = inConfig;
    auto s = CheckConfig(config);
    if (s.Failure()) {
        UC_ERROR("Failed to check memory store config: {}.", s);
        return s;
    }
    s = buffer_.Setup(config);
    if (s.Failure()) {
        UC_ERROR("Failed({}) to setup memory buffer.", s);
        return s;
    }
    transEnable_ = config.deviceId >= 0;
    if (transEnable_) {
        s = transMgr_.Setup(config, &buffer_);
        if (s.Failure()) {
            UC_ERROR("Failed({}) to setup memory transfer manager.", s);
            return s;
        }
    }
    ShowConfig(config);
    return Status::OK();
}

std::string MemoryStore::Readme() const { return "MemoryStore"; }

Expected<std::vector<uint8_t>> MemoryStore::Lookup(const Detail::BlockId* blocks, size_t num)
{
    auto res = buffer_.Lookup(blocks, num);
    if (!res) { UC_ERROR("Failed({}) to lookup memory blocks({}).", res.Error(), num); }
    return res;
}

Expected<ssize_t> MemoryStore::LookupOnPrefix(const Detail::BlockId* blocks, size_t num)
{
    auto res = buffer_.LookupOnPrefix(blocks, num);
    if (!res) { UC_ERROR("Failed({}) to lookup memory prefix blocks({}).", res.Error(), num); }
    return res;
}

void MemoryStore::Prefetch(const Detail::BlockId*, size_t) {}

Expected<std::vector<uint8_t>> MemoryStore::LookupTokens(const Detail::TokenLayerTaskDesc& task)
{
    auto res = buffer_.LookupTokens(task);
    if (!res) {
        UC_ERROR("Failed({}) to lookup memory token-layer items({}).", res.Error(),
                 task.size());
    }
    return res;
}

Expected<Detail::TaskHandle> MemoryStore::Load(Detail::TaskDesc task)
{
    if (!transEnable_) { return Status::Error("transfer is not enable"); }
    auto res = transMgr_.Submit({TransTask::Type::LOAD, std::move(task)});
    if (!res) { UC_ERROR("Failed({}) to submit memory load task.", res.Error()); }
    return res;
}

Expected<Detail::TaskHandle> MemoryStore::Dump(Detail::TaskDesc task)
{
    if (!transEnable_) { return Status::Error("transfer is not enable"); }
    auto res = transMgr_.Submit({TransTask::Type::DUMP, std::move(task)});
    if (!res) { UC_ERROR("Failed({}) to submit memory dump task.", res.Error()); }
    return res;
}

Expected<Detail::TaskHandle> MemoryStore::LoadTokens(Detail::TokenLayerTaskDesc task)
{
    if (!transEnable_) { return Status::Error("transfer is not enable"); }
    auto res = transMgr_.Submit({TransTask::Type::LOAD_TOKENS, std::move(task)});
    if (!res) { UC_ERROR("Failed({}) to submit memory token load task.", res.Error()); }
    return res;
}

Expected<Detail::TaskHandle> MemoryStore::DumpTokens(Detail::TokenLayerTaskDesc task)
{
    if (!transEnable_) { return Status::Error("transfer is not enable"); }
    auto res = transMgr_.Submit({TransTask::Type::DUMP_TOKENS, std::move(task)});
    if (!res) { UC_ERROR("Failed({}) to submit memory token dump task.", res.Error()); }
    return res;
}

Expected<bool> MemoryStore::Check(Detail::TaskHandle taskId)
{
    if (!transEnable_) { return Status::Error("transfer is not enable"); }
    auto res = transMgr_.Check(taskId);
    if (!res) { UC_ERROR("Failed({}) to check memory task({}).", res.Error(), taskId); }
    return res;
}

Status MemoryStore::Wait(Detail::TaskHandle taskId)
{
    if (!transEnable_) { return Status::Error("transfer is not enable"); }
    auto s = transMgr_.Wait(taskId);
    if (s.Failure()) { UC_ERROR("Failed({}) to wait memory task({}).", s, taskId); }
    return s;
}

size_t MemoryStore::Sum(const std::vector<size_t>& values)
{
    return std::accumulate(values.begin(), values.end(), size_t{0});
}

Status MemoryStore::CheckConfig(Config& config)
{
    if (config.deviceId < -1) {
        return Status::InvalidParam("invalid device({})", config.deviceId);
    }
    if (config.shardSize == 0) { return Status::InvalidParam("invalid shard size"); }
    if (config.blockSize == 0 || config.blockSize % config.shardSize != 0) {
        return Status::InvalidParam("invalid block size({})", config.blockSize);
    }
    if (config.tensorSizeList.empty()) { config.tensorSizeList = {config.shardSize}; }
    if (Sum(config.tensorSizeList) != config.shardSize) {
        return Status::InvalidParam("invalid tensor sizes on shard({})", config.shardSize);
    }
    if (config.memoryTokenChunkSize == 0) {
        return Status::InvalidParam("invalid memory token chunk size");
    }
    if (config.waitingQueueDepth <= 1 || config.runningQueueDepth <= 1) {
        return Status::InvalidParam("invalid queue depth({},{})", config.waitingQueueDepth,
                                    config.runningQueueDepth);
    }
    if (config.requiredTensorTypes.empty()) {
        return Status::InvalidParam("invalid memory required tensor types");
    }
    size_t perTokenSize = 0;
    for (const auto type : config.requiredTensorTypes) {
        auto iter = config.tensorSizesByType.find(type);
        if (iter == config.tensorSizesByType.end() || iter->second.empty()) {
            return Status::InvalidParam("invalid tensor size for type({})", type);
        }
        perTokenSize += Sum(iter->second);
    }
    if (perTokenSize == 0 || config.shardSize % perTokenSize != 0) {
        return Status::InvalidParam("invalid token payload size({}) on shard({})", perTokenSize,
                                    config.shardSize);
    }
    config.tokensPerBlock = config.shardSize / perTokenSize;
    return Status::OK();
}

void MemoryStore::ShowConfig(const Config& config) const
{
    constexpr const char* ns = "MemoryStore";
    std::string buildType = UCM_BUILD_TYPE;
    if (buildType.empty()) { buildType = "Release"; }
    UC_INFO("{}-{}({}).", ns, UCM_COMMIT_ID, buildType);
    UC_INFO("Set {}::StoreBackend to {}.", ns,
            config.storeBackend ? reinterpret_cast<StoreV1*>(config.storeBackend)->Readme()
                                : "null");
    UC_INFO("Set {}::DeviceId to {}.", ns, config.deviceId);
    UC_INFO("Set {}::ShardSize to {}.", ns, config.shardSize);
    UC_INFO("Set {}::BlockSize to {}.", ns, config.blockSize);
    UC_INFO("Set {}::TensorSizeList length to {}.", ns, config.tensorSizeList.size());
    UC_INFO("Set {}::MemoryTokenChunkSize to {}.", ns, config.memoryTokenChunkSize);
    UC_INFO("Set {}::TokensPerBlock to {}.", ns, config.tokensPerBlock);
    UC_INFO("Set {}::WaitingQueueDepth to {}.", ns, config.waitingQueueDepth);
    UC_INFO("Set {}::RunningQueueDepth to {}.", ns, config.runningQueueDepth);
    UC_INFO("Set {}::TimeoutMs to {}.", ns, config.timeoutMs);
}

}  // namespace UC::MemoryStore
