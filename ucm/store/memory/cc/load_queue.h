/**
 * MIT License
 *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
 * */
#ifndef UNIFIEDCACHE_MEMORY_STORE_CC_LOAD_QUEUE_H
#define UNIFIEDCACHE_MEMORY_STORE_CC_LOAD_QUEUE_H

#include <future>
#include <thread>
#include "template/hashset.h"
#include "template/spsc_ring_queue.h"
#include "thread/latch.h"
#include "trans/stream.h"
#include "trans_buffer.h"
#include "trans_task.h"
#include "ucmstore_v1.h"

namespace UC::MemoryStore {

class LoadQueue {
    using TaskPtr = std::shared_ptr<TransTask>;
    using WaiterPtr = std::shared_ptr<Latch>;
    using TaskPair = std::pair<TaskPtr, WaiterPtr>;
    using TaskIdSet = HashSet<Detail::TaskHandle>;

    enum class CopyType : uint8_t { FULL, TOKEN };

    struct CopyTask {
        Detail::TaskHandle taskHandle{0};
        CopyType type{CopyType::FULL};
        Detail::BlockId block{};
        size_t layer{0};
        size_t tokenOffset{0};
        Detail::TensorType tensorType{0};
        Detail::TaskHandle backendTaskHandle{0};
        bool backendResultNeedsCommit{false};
        std::vector<size_t> sizes;
        std::vector<std::byte> hostBuffer;
        std::vector<std::byte> backendBuffer;
        std::vector<void*> deviceAddrs;
        WaiterPtr waiter;
    };

public:
    ~LoadQueue();
    Status Setup(const Config& config, TaskIdSet* failureSet, TransBuffer* buffer);
    void Submit(TaskPtr task, WaiterPtr waiter);

private:
    void DispatchStage();
    void DispatchOneTask(TaskPair&& pair);
    void TransferStage(int32_t deviceId, std::promise<Status>& started);
    void TransferOneTask(Trans::Stream* stream, CopyTask&& task);
    Status WaitBackendTaskReady(CopyTask& task);
    Status HostToDeviceScatterAsync(Trans::Stream* stream, void* host,
                                    const std::vector<size_t>& sizes, void** device);

private:
    alignas(64) std::atomic_bool stop_{false};
    Detail::TaskHandle finishedBackendTaskHandle_{0};
    TaskIdSet* failureSet_{nullptr};
    TransBuffer* buffer_{nullptr};
    StoreV1* backend_{nullptr};
    size_t shardSize_{0};
    std::vector<size_t> tensorSizeList_{};
    std::unordered_map<Detail::TensorType, std::vector<size_t>> tensorSizesByType_{};
    SpscRingQueue<TaskPair> waiting_;
    SpscRingQueue<CopyTask> running_;
    std::thread dispatcher_;
    std::thread transfer_;
    std::vector<CopyTask> holder_;
};

}  // namespace UC::MemoryStore

#endif
