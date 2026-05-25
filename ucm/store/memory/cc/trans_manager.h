/**
 * MIT License
 *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
 * */
#ifndef UNIFIEDCACHE_MEMORY_STORE_CC_TRANS_MANAGER_H
#define UNIFIEDCACHE_MEMORY_STORE_CC_TRANS_MANAGER_H

#include "dump_queue.h"
#include "load_queue.h"
#include "logger/logger.h"
#include "template/task_wrapper.h"
#include "trans_task.h"

namespace UC::MemoryStore {

class TransManager : public Detail::TaskWrapper<TransTask, Detail::TaskHandle> {
    LoadQueue loadQ_;
    DumpQueue dumpQ_;

public:
    Status Setup(const Config& config, TransBuffer* buffer)
    {
        timeoutMs_ = config.timeoutMs;
        auto s = loadQ_.Setup(config, &failureSet_, buffer);
        if (s.Failure()) { return s; }
        return dumpQ_.Setup(config, &failureSet_, buffer);
    }

protected:
    void Dispatch(TaskPtr t, WaiterPtr w) override
    {
        UC_DEBUG("Memory task({},{},{}) dispatching.", t->id, Brief(*t), TaskSize(*t));
        if (t->type == TransTask::Type::LOAD || t->type == TransTask::Type::LOAD_TOKENS) {
            loadQ_.Submit(t, w);
        } else {
            dumpQ_.Submit(t, w);
        }
    }

private:
    static size_t TaskSize(const TransTask& task)
    {
        if (task.type == TransTask::Type::LOAD || task.type == TransTask::Type::DUMP) {
            return task.desc.size();
        }
        return task.tokenDesc.size();
    }
    static const char* Brief(const TransTask& task)
    {
        switch (task.type) {
            case TransTask::Type::LOAD:
                return "Load";
            case TransTask::Type::DUMP:
                return "Dump";
            case TransTask::Type::LOAD_TOKENS:
                return "LoadTokens";
            case TransTask::Type::DUMP_TOKENS:
                return "DumpTokens";
        }
        return "Unknown";
    }
};

}  // namespace UC::MemoryStore

#endif
