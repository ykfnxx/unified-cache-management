/**
 * MIT License
 *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
 * */
#ifndef UNIFIEDCACHE_MEMORY_STORE_CC_DUMP_QUEUE_H
#define UNIFIEDCACHE_MEMORY_STORE_CC_DUMP_QUEUE_H

#include <future>
#include <thread>
#include "template/hashset.h"
#include "template/spsc_ring_queue.h"
#include "thread/latch.h"
#include "trans/device.h"
#include "trans_buffer.h"
#include "trans_task.h"

namespace UC::MemoryStore {

class DumpQueue {
    using TaskPtr = std::shared_ptr<TransTask>;
    using WaiterPtr = std::shared_ptr<Latch>;
    using TaskPair = std::pair<TaskPtr, WaiterPtr>;
    using TaskIdSet = HashSet<Detail::TaskHandle>;
    struct BackendShard {
        Detail::BlockId block{};
        size_t layer{0};
        std::vector<std::byte> full{};
    };
    struct DumpTask {
        Detail::TaskHandle taskHandle;
        Detail::TaskHandle backendTaskHandle{0};
        std::vector<BackendShard> backendShards;
        WaiterPtr waiter;
    };

public:
    ~DumpQueue();
    Status Setup(const Config& config, TaskIdSet* failureSet, TransBuffer* buffer);
    void Submit(TaskPtr task, WaiterPtr waiter);

private:
    void DispatchStage(std::promise<Status>& started);
    void DispatchOneTask(TaskPair&& pair);
    Status DumpTaskDesc(TaskPtr task, WaiterPtr waiter);
    Status DumpTokenTaskDesc(TaskPtr task, WaiterPtr waiter);
    Status SetupTransferStreams();
    std::shared_ptr<Trans::Stream> NextStream() noexcept;
    Status WaitEventOnStreams(void* event) noexcept;
    Status SynchronizeStreams() noexcept;
    void BackendDumpStage();

private:
    alignas(64) std::atomic_bool stop_{false};
    TaskIdSet* failureSet_{nullptr};
    TransBuffer* buffer_{nullptr};
    StoreV1* backend_{nullptr};
    int32_t deviceId_{0};
    size_t shardSize_{0};
    std::vector<size_t> tensorSizes_{};
    std::unordered_map<Detail::TensorType, std::vector<size_t>> tensorSizesByType_{};
    size_t streamNumber_{1};
    bool useGdr_{false};
    std::vector<ssize_t> cpuAffinityCores_{};
    SpscRingQueue<TaskPair> waiting_;
    SpscRingQueue<DumpTask> dumping_;
    std::thread dispatcher_;
    std::thread dumper_;
    size_t streamIndex_{0};
    std::vector<std::shared_ptr<Trans::Stream>> streams_;
};

}  // namespace UC::MemoryStore

#endif
