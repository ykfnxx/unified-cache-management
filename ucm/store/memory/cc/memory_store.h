/**
 * MIT License
 *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
 * */
#ifndef UNIFIEDCACHE_MEMORY_STORE_CC_MEMORY_STORE_H
#define UNIFIEDCACHE_MEMORY_STORE_CC_MEMORY_STORE_H

#include "global_config.h"
#include "trans_manager.h"
#include "trans_buffer.h"
#include "ucmstore_v1.h"

namespace UC::MemoryStore {

class MemoryStore : public StoreV1 {
public:
    ~MemoryStore() override = default;

    Status Setup(const Config& config);
    std::string Readme() const override;
    Expected<std::vector<uint8_t>> Lookup(const Detail::BlockId* blocks, size_t num) override;
    Expected<ssize_t> LookupOnPrefix(const Detail::BlockId* blocks, size_t num) override;
    void Prefetch(const Detail::BlockId* blocks, size_t num) override;
    Expected<std::vector<uint8_t>> LookupTokens(const Detail::TokenLayerTaskDesc& task) override;
    Expected<Detail::TaskHandle> Load(Detail::TaskDesc task) override;
    Expected<Detail::TaskHandle> Dump(Detail::TaskDesc task) override;
    Expected<Detail::TaskHandle> LoadTokens(Detail::TokenLayerTaskDesc task) override;
    Expected<Detail::TaskHandle> DumpTokens(Detail::TokenLayerTaskDesc task) override;
    Expected<bool> Check(Detail::TaskHandle taskId) override;
    Status Wait(Detail::TaskHandle taskId) override;

private:
    static size_t Sum(const std::vector<size_t>& values);
    Status CheckConfig(Config& config);
    void ShowConfig(const Config& config) const;

private:
    TransBuffer buffer_;
    StoreV1* storeBackend_{nullptr};
    bool bufferEnable_{false};
    bool transEnable_{false};
    TransManager transMgr_;
};

}  // namespace UC::MemoryStore

#endif
