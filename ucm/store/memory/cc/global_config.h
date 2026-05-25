/**
 * MIT License
 *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
 * */
#ifndef UNIFIEDCACHE_MEMORY_STORE_CC_GLOBAL_CONFIG_H
#define UNIFIEDCACHE_MEMORY_STORE_CC_GLOBAL_CONFIG_H

#include <cstdint>
#include <unordered_map>
#include "ucmstore_v1.h"

namespace UC::MemoryStore {

struct Config {
    StoreV1* storeBackend{nullptr};
    std::string uniqueId{};
    int32_t deviceId{0};
    size_t shardSize{0};
    size_t blockSize{0};
    std::vector<size_t> tensorSizes{};
    size_t memoryTokenChunkSize{16};
    size_t memoryBufferCapacity{1ULL << 30};
    size_t waitingQueueDepth{8192};
    size_t runningQueueDepth{524288};
    size_t timeoutMs{30000};
    size_t streamNumber{1};
    bool useGdr{false};
    std::vector<ssize_t> cpuAffinityCores{};
    std::vector<Detail::TensorType> requiredTensorTypes{};
    std::unordered_map<Detail::TensorType, std::vector<size_t>> tensorSizesByType{};
    size_t tokensPerBlock{0};
};

}  // namespace UC::MemoryStore

#endif
