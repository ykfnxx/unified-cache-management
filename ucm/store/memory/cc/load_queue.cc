/**
 * MIT License
 *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
 * */
#include "load_queue.h"
#include "logger/logger.h"
#include "trans/device.h"

namespace UC::MemoryStore {

LoadQueue::~LoadQueue()
{
    stop_.store(true);
    if (dispatcher_.joinable()) { dispatcher_.join(); }
    if (transfer_.joinable()) { transfer_.join(); }
}

Status LoadQueue::Setup(const Config& config, TaskIdSet* failureSet, TransBuffer* buffer)
{
    failureSet_ = failureSet;
    buffer_ = buffer;
    backend_ = reinterpret_cast<StoreV1*>(config.storeBackend);
    shardSize_ = config.shardSize;
    tensorSizeList_ = config.tensorSizeList;
    tensorSizesByType_ = config.tensorSizesByType;
    waiting_.Setup(config.waitingQueueDepth);
    running_.Setup(config.runningQueueDepth);
    holder_.reserve(1024);
    dispatcher_ = std::thread{&LoadQueue::DispatchStage, this};
    std::promise<Status> started;
    auto fut = started.get_future();
    transfer_ = std::thread{&LoadQueue::TransferStage, this, config.deviceId, std::ref(started)};
    return fut.get();
}

void LoadQueue::Submit(TaskPtr task, WaiterPtr waiter)
{
    waiter->Up();
    auto success = waiting_.TryPush({task, waiter});
    if (success) { return; }
    UC_ERROR("Waiting queue full, submit memory load task({}) failed.", task->id);
    failureSet_->Insert(task->id);
    waiter->Done();
}

void LoadQueue::DispatchStage() { waiting_.ConsumerLoop(stop_, &LoadQueue::DispatchOneTask, this); }

void LoadQueue::DispatchOneTask(TaskPair&& pair)
{
    auto& task = pair.first;
    auto& waiter = pair.second;
    if (failureSet_->Contains(task->id)) {
        waiter->Done();
        return;
    }

    auto s = Status::OK();
    if (task->type == TransTask::Type::LOAD) {
        const auto nShard = task->desc.size();
        if (nShard == 0) {
            waiter->Done();
            return;
        }

        std::vector<CopyTask> copyTasks(nShard);
        Detail::TaskDesc backendTask;
        backendTask.brief = "Backend2Memory";
        for (size_t i = 0; i < nShard; ++i) {
            const auto& shard = task->desc[i];
            auto& copyTask = copyTasks[i];
            copyTask.taskHandle = task->id;
            copyTask.type = CopyType::FULL;
            copyTask.block = shard.owner;
            copyTask.layer = shard.index;
            copyTask.sizes = tensorSizeList_;
            copyTask.deviceAddrs = shard.addrs;
            copyTask.waiter = (i + 1 < nShard) ? nullptr : waiter;

            s = buffer_->ReadFull(shard.owner, shard.index, copyTask.hostBuffer);
            if (s == Status::NotFound() && backend_) {
                copyTask.hostBuffer.resize(shardSize_);
                backendTask.push_back({shard.owner, shard.index, {copyTask.hostBuffer.data()}});
                copyTask.backendResultNeedsCommit = true;
                s = Status::OK();
            }
            if (s.Failure()) { break; }
        }
        if (s.Success() && !backendTask.empty()) {
            auto res = backend_->Load(std::move(backendTask));
            if (!res) {
                s = res.Error();
            } else {
                for (auto& copyTask : copyTasks) {
                    if (copyTask.backendResultNeedsCommit) {
                        copyTask.backendTaskHandle = res.Value();
                    }
                }
            }
        }
        if (s.Success()) {
            for (auto& copyTask : copyTasks) { running_.Push(std::move(copyTask)); }
        }
    } else {
        const auto nItem = task->tokenDesc.size();
        if (nItem == 0) {
            waiter->Done();
            return;
        }

        std::vector<CopyTask> copyTasks(nItem);
        Detail::TaskDesc backendTask;
        backendTask.brief = "Backend2Memory";
        for (size_t i = 0; i < nItem; ++i) {
            const auto& item = task->tokenDesc[i];
            auto& copyTask = copyTasks[i];
            copyTask.taskHandle = task->id;
            copyTask.type = CopyType::TOKEN;
            copyTask.block = item.owner;
            copyTask.layer = item.layer;
            copyTask.tokenOffset = item.tokenOffset;
            copyTask.tensorType = item.tensorType;
            copyTask.sizes = tensorSizesByType_.at(item.tensorType);
            copyTask.deviceAddrs = item.addrs;
            copyTask.waiter = (i + 1 < nItem) ? nullptr : waiter;

            s = buffer_->ReadToken(item, copyTask.hostBuffer);
            if (s == Status::NotFound() && backend_) {
                copyTask.backendBuffer.resize(shardSize_);
                backendTask.push_back({item.owner, item.layer, {copyTask.backendBuffer.data()}});
                copyTask.backendResultNeedsCommit = true;
                s = Status::OK();
            }
            if (s.Failure()) { break; }
        }
        if (s.Success() && !backendTask.empty()) {
            auto res = backend_->Load(std::move(backendTask));
            if (!res) {
                s = res.Error();
            } else {
                for (auto& copyTask : copyTasks) {
                    if (copyTask.backendResultNeedsCommit) {
                        copyTask.backendTaskHandle = res.Value();
                    }
                }
            }
        }
        if (s.Success()) {
            for (auto& copyTask : copyTasks) { running_.Push(std::move(copyTask)); }
        }
    }

    if (s.Failure()) {
        UC_ERROR("Failed({}) to run memory load task({}).", s, task->id);
        failureSet_->Insert(task->id);
        waiter->Done();
    }
}

void LoadQueue::TransferStage(int32_t deviceId, std::promise<Status>& started)
{
    Trans::Device device;
    auto s = device.Setup(deviceId);
    if (s.Failure()) {
        UC_ERROR("Failed({}) to setup device({}).", s, deviceId);
        started.set_value(s);
        return;
    }
    auto stream = device.MakeStream();
    if (!stream) {
        started.set_value(Status::Error());
        return;
    }
    started.set_value(Status::OK());
    running_.ConsumerLoop(stop_, &LoadQueue::TransferOneTask, this, stream.get());
}

void LoadQueue::TransferOneTask(Trans::Stream* stream, CopyTask&& task)
{
    if (failureSet_->Contains(task.taskHandle)) {
        if (task.waiter) { task.waiter->Done(); }
        return;
    }

    auto s = WaitBackendTaskReady(task);
    if (s.Success()) {
        s = HostToDeviceScatterAsync(stream, task.hostBuffer.data(), task.sizes,
                                     task.deviceAddrs);
    }
    if (s.Success()) {
        holder_.push_back(std::move(task));
        if (!holder_.back().waiter) { return; }
        s = stream->Synchronized();
        auto waiter = holder_.back().waiter;
        holder_.clear();
        if (s.Failure()) {
            failureSet_->Insert(task.taskHandle);
            UC_ERROR("Failed({}) to synchronize memory load stream.", s);
        }
        waiter->Done();
        return;
    }

    UC_ERROR("Failed({}) to run memory load transfer task({}).", s, task.taskHandle);
    failureSet_->Insert(task.taskHandle);
    if (task.waiter) { task.waiter->Done(); }
}

Status LoadQueue::WaitBackendTaskReady(CopyTask& task)
{
    if (task.backendTaskHandle == 0) { return Status::OK(); }
    if (task.backendTaskHandle > finishedBackendTaskHandle_) {
        auto s = backend_->Wait(task.backendTaskHandle);
        if (s.Failure()) { return s; }
        finishedBackendTaskHandle_ = task.backendTaskHandle;
    }
    if (!task.backendResultNeedsCommit) { return Status::OK(); }
    if (task.type == CopyType::FULL) {
        auto s = buffer_->CommitFull(task.block, task.layer, task.hostBuffer);
        if (s.Failure()) { return s; }
    } else {
        auto s = buffer_->CommitFull(task.block, task.layer, task.backendBuffer);
        if (s.Failure()) { return s; }
        Detail::TokenLayerShard item{task.block, task.layer, task.tokenOffset, task.tensorType,
                                     task.deviceAddrs};
        s = buffer_->ReadToken(item, task.hostBuffer);
        if (s.Failure()) { return s; }
        task.backendBuffer.clear();
    }
    task.backendResultNeedsCommit = false;
    return Status::OK();
}

Status LoadQueue::HostToDeviceScatterAsync(Trans::Stream* stream, void* host,
                                           const std::vector<size_t>& sizes,
                                           const std::vector<void*>& device)
{
    if (device.size() != sizes.size()) {
        return Status::InvalidParam("invalid destination addr number({},{})", device.size(),
                                    sizes.size());
    }
    if (!host) { return Status::InvalidParam("invalid source host addr"); }
    size_t offset = 0;
    for (size_t i = 0; i < sizes.size(); ++i) {
        if (!device[i]) { return Status::InvalidParam("invalid destination addr"); }
        auto pHost = static_cast<void*>(static_cast<int8_t*>(host) + offset);
        auto s = stream->HostToDeviceAsync(pHost, device[i], sizes[i]);
        if (s.Failure()) { return s; }
        offset += sizes[i];
    }
    return Status::OK();
}

}  // namespace UC::MemoryStore
