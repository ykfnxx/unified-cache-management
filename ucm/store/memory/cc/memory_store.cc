/**
 * MIT License
 *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
 * */
#include <numeric>
#include "logger/logger.h"
#include "trans_manager.h"

namespace UC::MemoryStore {

class MemoryStore : public StoreV1 {
    TransBuffer buffer_;
    TransManager transMgr_;

public:
    Status Setup(const Detail::Dictionary& inConfig) override
    {
        auto config = ParseConfig(inConfig);
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
        s = transMgr_.Setup(config, &buffer_);
        if (s.Failure()) {
            UC_ERROR("Failed({}) to setup memory transfer manager.", s);
            return s;
        }
        ShowConfig(config);
        return Status::OK();
    }

    std::string Readme() const override { return "MemoryStore"; }

    Expected<std::vector<uint8_t>> Lookup(const Detail::BlockId* blocks, size_t num) override
    {
        auto res = buffer_.Lookup(blocks, num);
        if (!res) { UC_ERROR("Failed({}) to lookup memory blocks({}).", res.Error(), num); }
        return res;
    }

    Expected<ssize_t> LookupOnPrefix(const Detail::BlockId* blocks, size_t num) override
    {
        auto res = buffer_.LookupOnPrefix(blocks, num);
        if (!res) { UC_ERROR("Failed({}) to lookup memory prefix blocks({}).", res.Error(), num); }
        return res;
    }

    void Prefetch(const Detail::BlockId*, size_t) override {}

    Expected<std::vector<uint8_t>> LookupTokens(const Detail::TokenLayerTaskDesc& task) override
    {
        auto res = buffer_.LookupTokens(task);
        if (!res) {
            UC_ERROR("Failed({}) to lookup memory token-layer items({}).", res.Error(),
                     task.size());
        }
        return res;
    }

    Expected<Detail::TaskHandle> Load(Detail::TaskDesc task) override
    {
        auto res = transMgr_.Submit({TransTask::Type::LOAD, std::move(task)});
        if (!res) { UC_ERROR("Failed({}) to submit memory load task.", res.Error()); }
        return res;
    }

    Expected<Detail::TaskHandle> Dump(Detail::TaskDesc task) override
    {
        auto res = transMgr_.Submit({TransTask::Type::DUMP, std::move(task)});
        if (!res) { UC_ERROR("Failed({}) to submit memory dump task.", res.Error()); }
        return res;
    }

    Expected<Detail::TaskHandle> LoadTokens(Detail::TokenLayerTaskDesc task) override
    {
        auto res = transMgr_.Submit({TransTask::Type::LOAD_TOKENS, std::move(task)});
        if (!res) { UC_ERROR("Failed({}) to submit memory token load task.", res.Error()); }
        return res;
    }

    Expected<Detail::TaskHandle> DumpTokens(Detail::TokenLayerTaskDesc task) override
    {
        auto res = transMgr_.Submit({TransTask::Type::DUMP_TOKENS, std::move(task)});
        if (!res) { UC_ERROR("Failed({}) to submit memory token dump task.", res.Error()); }
        return res;
    }

    Expected<bool> Check(Detail::TaskHandle taskId) override
    {
        auto res = transMgr_.Check(taskId);
        if (!res) { UC_ERROR("Failed({}) to check memory task({}).", res.Error(), taskId); }
        return res;
    }

    Status Wait(Detail::TaskHandle taskId) override
    {
        auto s = transMgr_.Wait(taskId);
        if (s.Failure()) { UC_ERROR("Failed({}) to wait memory task({}).", s, taskId); }
        return s;
    }

private:
    Config ParseConfig(const Detail::Dictionary& dict)
    {
        Config config;
        dict.Get("store_backend", config.storeBackend);
        dict.Get("unique_id", config.uniqueId);
        dict.GetNumber("device_id", config.deviceId);
        size_t tensorSize = 0;
        dict.GetNumber("tensor_size", tensorSize);
        dict.GetNumber("shard_size", config.shardSize);
        dict.GetNumber("block_size", config.blockSize);
        if (tensorSize != 0 && config.shardSize != 0) {
            config.tensorSizes.assign(config.shardSize / tensorSize, tensorSize);
        } else {
            dict.GetNumbers("tensor_size_list", config.tensorSizes);
        }
        dict.GetNumber("memory_token_chunk_size", config.memoryTokenChunkSize);
        size_t capacityGb = 0;
        dict.GetNumber("memory_buffer_capacity_gb", capacityGb);
        if (capacityGb > 0) { config.memoryBufferCapacity = capacityGb << 30; }
        dict.GetNumber("waiting_queue_depth", config.waitingQueueDepth);
        dict.GetNumber("running_queue_depth", config.runningQueueDepth);
        dict.GetNumber("timeout_ms", config.timeoutMs);
        dict.GetNumber("cache_stream_number", config.streamNumber);
        dict.GetNumber("memory_stream_number", config.streamNumber);
        dict.Get("use_gdr", config.useGdr);
        dict.Get("cpu_affinity_cores", config.cpuAffinityCores);

        std::vector<size_t> required;
        dict.GetNumbers("memory_required_tensor_types", required);
        config.requiredTensorTypes.assign(required.begin(), required.end());
        for (const auto type : config.requiredTensorTypes) {
            std::vector<size_t> sizes;
            dict.GetNumbers("memory_tensor_size_by_type_" + std::to_string(type), sizes);
            if (!sizes.empty()) { config.tensorSizesByType[type] = std::move(sizes); }
        }
        return config;
    }

    static size_t Sum(const std::vector<size_t>& values)
    {
        return std::accumulate(values.begin(), values.end(), size_t{0});
    }

    Status CheckConfig(Config& config)
    {
        if (config.deviceId < 0) {
            return Status::InvalidParam("invalid device({})", config.deviceId);
        }
        if (config.shardSize == 0) { return Status::InvalidParam("invalid shard size"); }
        if (config.blockSize == 0 || config.blockSize % config.shardSize != 0) {
            return Status::InvalidParam("invalid block size({})", config.blockSize);
        }
        if (config.tensorSizes.empty()) { config.tensorSizes = {config.shardSize}; }
        if (Sum(config.tensorSizes) != config.shardSize) {
            return Status::InvalidParam("invalid tensor sizes on shard({})", config.shardSize);
        }
        if (config.memoryTokenChunkSize == 0) {
            return Status::InvalidParam("invalid memory token chunk size");
        }
        if (config.waitingQueueDepth <= 1 || config.runningQueueDepth <= 1) {
            return Status::InvalidParam("invalid queue depth({},{})", config.waitingQueueDepth,
                                        config.runningQueueDepth);
        }
        if (config.streamNumber < 1 || config.streamNumber > 32) {
            return Status::InvalidParam("invalid stream number({})", config.streamNumber);
        }
#ifdef CPU_SETSIZE
        for (const auto core : config.cpuAffinityCores) {
            if (core < 0 || core >= CPU_SETSIZE) {
                return Status::InvalidParam("invalid cpu core({})", core);
            }
        }
#endif
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

    void ShowConfig(const Config& config)
    {
        constexpr const char* ns = "MemoryStore";
        UC_INFO("{}-{}({}).", ns, UCM_COMMIT_ID, UCM_BUILD_TYPE);
        UC_INFO("Set {}::StoreBackend to {}.", ns,
                config.storeBackend ? config.storeBackend->Readme() : "null");
        UC_INFO("Set {}::DeviceId to {}.", ns, config.deviceId);
        UC_INFO("Set {}::ShardSize to {}.", ns, config.shardSize);
        UC_INFO("Set {}::BlockSize to {}.", ns, config.blockSize);
        UC_INFO("Set {}::TensorSizes to {}.", ns, config.tensorSizes);
        UC_INFO("Set {}::MemoryTokenChunkSize to {}.", ns, config.memoryTokenChunkSize);
        UC_INFO("Set {}::TokensPerBlock to {}.", ns, config.tokensPerBlock);
        UC_INFO("Set {}::CpuAffinityCores to {}.", ns, config.cpuAffinityCores);
        UC_INFO("Set {}::WaitingQueueDepth to {}.", ns, config.waitingQueueDepth);
        UC_INFO("Set {}::RunningQueueDepth to {}.", ns, config.runningQueueDepth);
        UC_INFO("Set {}::TimeoutMs to {}.", ns, config.timeoutMs);
        UC_INFO("Set {}::StreamNumber to {}.", ns, config.streamNumber);
        UC_INFO("Set {}::UseGdr to {}.", ns, config.useGdr);
    }
};

}  // namespace UC::MemoryStore

extern "C" UC::StoreV1* MakeMemoryStore() { return new UC::MemoryStore::MemoryStore(); }
