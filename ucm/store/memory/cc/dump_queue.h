/**
 * MIT License
 *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
 * */
#ifndef UNIFIEDCACHE_MEMORY_STORE_CC_DUMP_QUEUE_H
#define UNIFIEDCACHE_MEMORY_STORE_CC_DUMP_QUEUE_H

#include <future>
#include <thread>
#include "cache/cc/copy_stream.h"
#include "template/hashset.h"
#include "template/spsc_ring_queue.h"
#include "thread/latch.h"
#include "trans_buffer.h"
#include "trans_task.h"

namespace UC::MemoryStore {

class DumpQueue {
    using TaskPtr = std::shared_ptr<TransTask>;
    using WaiterPtr = std::shared_ptr<Latch>;
    using TaskPair = std::pair<TaskPtr, WaiterPtr>;
    using TaskIdSet = HashSet<Detail::TaskHandle>;
    struct DumpTask {
        Detail::TaskHandle taskHandle;
        Detail::TaskDesc hostTask;
        Detail::TokenLayerTaskDesc hostTokenTask;
        std::vector<std::byte> hostBuffer;
        WaiterPtr waiter;
    };
    struct CopyTask {
        Detail::TaskHandle taskHandle;
        std::vector<size_t> sizes;
        std::vector<void*> deviceAddrs;
        bool waitPrerequisite{false};
        uintptr_t prerequisiteHandle{0};
        DumpTask dumpTask;
    };

public:
    ~DumpQueue();
    Status Setup(const Config& config, TaskIdSet* failureSet, TransBuffer* buffer);
    void Submit(TaskPtr task, WaiterPtr waiter);

private:
    void DispatchStage();
    void DispatchOneTask(TaskPair&& pair);
    void TransferStage(std::promise<Status>& started);
    void TransferOneTask(UC::CacheStore::CopyStream& stream, CopyTask&& task);
    void BackendDumpStage();
    Status DeviceToHostGatherAsync(std::shared_ptr<Trans::Stream> stream, void** device,
                                   const std::vector<size_t>& sizes, void* host);

private:
    alignas(64) std::atomic_bool stop_{false};
    TaskIdSet* failureSet_{nullptr};
    TransBuffer* buffer_{nullptr};
    int32_t deviceId_{0};
    std::vector<size_t> tensorSizes_{};
    std::unordered_map<Detail::TensorType, std::vector<size_t>> tensorSizesByType_{};
    size_t streamNumber_{1};
    bool useGdr_{false};
    std::vector<ssize_t> cpuAffinityCores_{};
    SpscRingQueue<TaskPair> waiting_;
    SpscRingQueue<CopyTask> running_;
    SpscRingQueue<DumpTask> dumping_;
    std::thread dispatcher_;
    std::thread transfer_;
    std::thread dumper_;
    std::vector<DumpTask> holder_;
};

}  // namespace UC::MemoryStore

#endif
