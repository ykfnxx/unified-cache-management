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
#include "trans/device.h"
#include "trans_buffer.h"
#include "trans_task.h"

namespace UC::MemoryStore {

class LoadQueue {
    using TaskPtr = std::shared_ptr<TransTask>;
    using WaiterPtr = std::shared_ptr<Latch>;
    using TaskPair = std::pair<TaskPtr, WaiterPtr>;
    using TaskIdSet = HashSet<Detail::TaskHandle>;
    enum class CopyType : uint8_t { FULL, TOKEN };
    struct CopyTask {
        Detail::TaskHandle taskHandle;
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
    void TransferStage(std::promise<Status>& started);
    void TransferOneTask(CopyTask&& task);
    Status WaitBackendTaskReady(CopyTask& task);
    Status SetupTransferStreams();
    std::shared_ptr<Trans::Stream> NextStream() noexcept;
    Status SynchronizeStreams() noexcept;
    Status HostToDeviceScatterAsync(std::shared_ptr<Trans::Stream> stream, void* host,
                                    const std::vector<size_t>& sizes, void** device);

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
    SpscRingQueue<CopyTask> running_;
    std::thread dispatcher_;
    std::thread transfer_;
    std::vector<CopyTask> holder_;
    size_t streamIndex_{0};
    std::vector<std::shared_ptr<Trans::Stream>> streams_;
};

}  // namespace UC::MemoryStore

#endif
