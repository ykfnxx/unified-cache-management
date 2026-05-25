/**
 * MIT License
 *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
 * */
#include "load_queue.h"
#include "logger/logger.h"
#include "thread/cpu_affinity.h"

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
    backend_ = config.storeBackend;
    deviceId_ = config.deviceId;
    shardSize_ = config.shardSize;
    tensorSizes_ = config.tensorSizes;
    tensorSizesByType_ = config.tensorSizesByType;
    streamNumber_ = config.streamNumber;
    useGdr_ = config.useGdr;
    cpuAffinityCores_ = config.cpuAffinityCores;
    waiting_.Setup(config.waitingQueueDepth);
    running_.Setup(config.runningQueueDepth);
    holder_.reserve(1024);
    dispatcher_ = std::thread{&LoadQueue::DispatchStage, this};
    std::promise<Status> started;
    auto fut = started.get_future();
    transfer_ = std::thread{&LoadQueue::TransferStage, this, std::ref(started)};
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

void LoadQueue::DispatchStage()
{
    if (!cpuAffinityCores_.empty()) {
        auto s = CpuAffinity::SetCpuAffinity4CurrentThread(cpuAffinityCores_);
        if (s.Failure()) { UC_WARN("Failed({}) to set affinity.", s); }
    }
    waiting_.ConsumerLoop(stop_, &LoadQueue::DispatchOneTask, this);
}

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
        for (size_t i = 0; i < nShard; ++i) {
            const auto& shard = task->desc[i];
            CopyTask copyTask;
            copyTask.taskHandle = task->id;
            copyTask.type = CopyType::FULL;
            copyTask.block = shard.owner;
            copyTask.layer = shard.index;
            copyTask.sizes = tensorSizes_;
            copyTask.deviceAddrs = shard.addrs;
            copyTask.waiter = (i + 1 < nShard) ? nullptr : waiter;
            s = buffer_->ReadFull(shard.owner, shard.index, copyTask.hostBuffer);
            if (s == Status::NotFound() && backend_) {
                copyTask.hostBuffer.resize(shardSize_);
                Detail::TaskDesc backendTask;
                backendTask.brief = "Backend2Memory";
                backendTask.push_back({shard.owner, shard.index, {copyTask.hostBuffer.data()}});
                auto res = backend_->Load(std::move(backendTask));
                if (!res) {
                    s = res.Error();
                } else {
                    copyTask.backendTaskHandle = res.Value();
                    copyTask.backendResultNeedsCommit = true;
                    s = Status::OK();
                }
            }
            if (s.Failure()) { break; }
            running_.Push(std::move(copyTask));
        }
    } else {
        const auto nItem = task->tokenDesc.size();
        if (nItem == 0) {
            waiter->Done();
            return;
        }
        for (size_t i = 0; i < nItem; ++i) {
            const auto& item = task->tokenDesc[i];
            CopyTask copyTask;
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
                Detail::TaskDesc backendTask;
                backendTask.brief = "Backend2Memory";
                backendTask.push_back({item.owner, item.layer, {copyTask.backendBuffer.data()}});
                auto res = backend_->Load(std::move(backendTask));
                if (!res) {
                    s = res.Error();
                } else {
                    copyTask.backendTaskHandle = res.Value();
                    copyTask.backendResultNeedsCommit = true;
                    s = Status::OK();
                }
            }
            if (s.Failure()) { break; }
            running_.Push(std::move(copyTask));
        }
    }
    if (s.Failure()) {
        UC_ERROR("Failed({}) to run memory load task({}).", s, task->id);
        failureSet_->Insert(task->id);
        waiter->Done();
    }
}

void LoadQueue::TransferStage(std::promise<Status>& started)
{
    auto s = SetupTransferStreams();
    started.set_value(s);
    if (s.Failure()) { return; }
    if (!cpuAffinityCores_.empty()) {
        s = CpuAffinity::SetCpuAffinity4CurrentThread(cpuAffinityCores_);
        if (s.Failure()) { UC_WARN("Failed({}) to set affinity.", s); }
    }
    running_.ConsumerLoop(stop_, &LoadQueue::TransferOneTask, this);
}

void LoadQueue::TransferOneTask(CopyTask&& task)
{
    if (failureSet_->Contains(task.taskHandle)) {
        if (task.waiter) { task.waiter->Done(); }
        return;
    }
    auto s = WaitBackendTaskReady(task);
    auto copyStream = NextStream();
    if (s.Success() && !copyStream) {
        s = Status::Error("invalid memory load stream");
    } else if (s.Success()) {
        s = HostToDeviceScatterAsync(copyStream, task.hostBuffer.data(), task.sizes,
                                     task.deviceAddrs.data());
    }
    if (s.Success() && !task.waiter) {
        holder_.push_back(std::move(task));
        return;
    }
    if (s.Success()) {
        holder_.push_back(std::move(task));
        s = SynchronizeStreams();
    }
    if (s.Failure()) {
        UC_ERROR("Failed({}) to run memory load transfer task({}).", s, task.taskHandle);
        failureSet_->Insert(task.taskHandle);
    }
    auto waiter = task.waiter;
    if (!waiter && !holder_.empty()) { waiter = holder_.back().waiter; }
    holder_.clear();
    if (waiter) { waiter->Done(); }
}

Status LoadQueue::WaitBackendTaskReady(CopyTask& task)
{
    if (task.backendTaskHandle == 0) { return Status::OK(); }
    auto s = backend_->Wait(task.backendTaskHandle);
    if (s.Failure()) {
        UC_ERROR("Failed({}) to wait backend({}) for memory load task({}).", s,
                 task.backendTaskHandle, task.taskHandle);
        return s;
    }
    if (!task.backendResultNeedsCommit) { return Status::OK(); }
    if (task.type == CopyType::FULL) {
        s = buffer_->CommitFull(task.block, task.layer, task.hostBuffer);
    } else {
        s = buffer_->CommitFull(task.block, task.layer, task.backendBuffer);
        if (s.Success()) {
            Detail::TokenLayerShard item{task.block, task.layer, task.tokenOffset, task.tensorType,
                                         task.deviceAddrs};
            s = buffer_->ReadToken(item, task.hostBuffer);
        }
    }
    if (s.Failure()) {
        UC_ERROR("Failed({}) to commit backend result for memory load task({}).", s,
                 task.taskHandle);
        return s;
    }
    task.backendTaskHandle = 0;
    task.backendResultNeedsCommit = false;
    task.backendBuffer.clear();
    return Status::OK();
}

Status LoadQueue::SetupTransferStreams()
{
    Trans::Device device;
    auto s = device.Setup(deviceId_);
    if (s.Failure()) {
        UC_ERROR("Failed({}) to setup device({}).", s, deviceId_);
        return s;
    }
    streamIndex_ = 0;
    streams_.clear();
    streams_.reserve(streamNumber_);
    for (size_t i = 0; i < streamNumber_; ++i) {
        std::shared_ptr<Trans::Stream> stream =
            useGdr_ ? device.MakeGdrStream() : device.MakeSharedStream();
        if (!stream) {
            UC_ERROR("Failed to make memory transfer stream on device({}).", deviceId_);
            return Status::Error();
        }
        streams_.push_back(std::move(stream));
    }
    return Status::OK();
}

std::shared_ptr<Trans::Stream> LoadQueue::NextStream() noexcept
{
    if (streams_.empty()) { return nullptr; }
    auto& stream = streams_[streamIndex_];
    streamIndex_ = (streamIndex_ + 1) % streams_.size();
    return stream;
}

Status LoadQueue::SynchronizeStreams() noexcept
{
    auto status = Status::OK();
    for (auto& stream : streams_) {
        auto s = stream->Synchronized();
        if (s.Success()) { continue; }
        UC_ERROR("Failed({}) to synchronize memory load stream on device({}).", s, deviceId_);
        if (status.Success()) { status = s; }
    }
    return status;
}

Status LoadQueue::HostToDeviceScatterAsync(std::shared_ptr<Trans::Stream> stream, void* host,
                                           const std::vector<size_t>& sizes, void** device)
{
    size_t offset = 0;
    for (size_t i = 0; i < sizes.size(); ++i) {
        auto pHost = static_cast<void*>(static_cast<int8_t*>(host) + offset);
        auto s = stream->HostToDeviceAsync(pHost, device[i], sizes[i]);
        if (s.Failure()) { return s; }
        offset += sizes[i];
    }
    return Status::OK();
}

}  // namespace UC::MemoryStore
