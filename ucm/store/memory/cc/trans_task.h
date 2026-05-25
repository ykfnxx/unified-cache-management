/**
 * MIT License
 *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
 * */
#ifndef UNIFIEDCACHE_MEMORY_STORE_CC_TRANS_TASK_H
#define UNIFIEDCACHE_MEMORY_STORE_CC_TRANS_TASK_H

#include <atomic>
#include <cstdint>
#include "ucmstore_v1.h"

namespace UC::MemoryStore {

class TransTask {
public:
    enum class Type : uint8_t { LOAD, DUMP, LOAD_TOKENS, DUMP_TOKENS };

public:
    Detail::TaskHandle id{0};
    Type type{Type::DUMP};
    Detail::TaskDesc desc;
    Detail::TokenLayerTaskDesc tokenDesc;

public:
    TransTask(Type inType, Detail::TaskDesc inDesc)
        : id{NextId()}, type{inType}, desc{std::move(inDesc)}
    {
    }
    TransTask(Type inType, Detail::TokenLayerTaskDesc inDesc)
        : id{NextId()}, type{inType}, tokenDesc{std::move(inDesc)}
    {
    }

private:
    static size_t NextId() noexcept
    {
        static std::atomic<size_t> id{1};
        return id.fetch_add(1, std::memory_order_relaxed);
    }
};

}  // namespace UC::MemoryStore

#endif
