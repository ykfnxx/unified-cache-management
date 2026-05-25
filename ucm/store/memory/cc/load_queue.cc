/**
 * MIT License
 *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
 * */
#include "load_queue.h"
#include <numeric>
#include "logger/logger.h"
#include "thread/cpu_affinity.h"

namespace UC::MemoryStore {

namespace {
size_t Sum(const std::vector<size_t>& sizes)
{
    return std::accumulate(sizes.begin(), sizes.end(), size_t{0});
}

std::vector<void*> BuildHostAddrs(std::vector<std::byte>& buffer, const std::vector<size_t>& sizes)
{
    std::vector<void*> addrs;
    addrs.reserve(sizes.size());
    size_t offset = 0;
    for (const auto size : sizes) {
        addrs.push_back(buffer.data() + offset);
        offset += size;
    }
    return addrs;
}
}  // namespace

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
    deviceId_ = config.deviceId;
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
        Detail::TaskDesc hostTask;
        hostTask.brief = task->desc.brief;
        for (size_t i = 0; i < nShard; ++i) {
            const auto& shard = task->desc[i];
            std::vector<std::byte> hostBuffer(Sum(tensorSizes_));
            auto hostAddrs = BuildHostAddrs(hostBuffer, tensorSizes_);
            hostTask.clear();
            hostTask.push_back({shard.owner, shard.index, hostAddrs});
            s = buffer_->Load(hostTask);
            if (s.Failure()) { break; }
            running_.Push({task->id, tensorSizes_, std::move(hostBuffer), shard.addrs,
                           (i + 1 < nShard) ? nullptr : waiter});
        }
    } else {
        const auto nItem = task->tokenDesc.size();
        if (nItem == 0) {
            waiter->Done();
            return;
        }
        Detail::TokenLayerTaskDesc hostTask;
        hostTask.brief = task->tokenDesc.brief;
        for (size_t i = 0; i < nItem; ++i) {
            const auto& item = task->tokenDesc[i];
            const auto& sizes = tensorSizesByType_.at(item.tensorType);
            std::vector<std::byte> hostBuffer(Sum(sizes));
            auto hostAddrs = BuildHostAddrs(hostBuffer, sizes);
            hostTask.clear();
            hostTask.push_back(
                {item.owner, item.layer, item.tokenOffset, item.tensorType, hostAddrs});
            s = buffer_->LoadTokens(hostTask);
            if (s.Failure()) { break; }
            running_.Push({task->id, sizes, std::move(hostBuffer), item.addrs,
                           (i + 1 < nItem) ? nullptr : waiter});
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
    UC::CacheStore::CopyStream stream;
    auto s = stream.Setup(deviceId_, streamNumber_, useGdr_);
    started.set_value(s);
    if (s.Failure()) { return; }
    if (!cpuAffinityCores_.empty()) {
        s = CpuAffinity::SetCpuAffinity4CurrentThread(cpuAffinityCores_);
        if (s.Failure()) { UC_WARN("Failed({}) to set affinity.", s); }
    }
    running_.ConsumerLoop(stop_, &LoadQueue::TransferOneTask, this, stream);
}

void LoadQueue::TransferOneTask(UC::CacheStore::CopyStream& stream, CopyTask&& task)
{
    if (failureSet_->Contains(task.taskHandle)) {
        if (task.waiter) { task.waiter->Done(); }
        return;
    }
    auto s = Status::OK();
    auto copyStream = stream.NextStream();
    if (!copyStream) {
        s = Status::Error("invalid memory load stream");
    } else {
        s = HostToDeviceScatterAsync(copyStream, task.hostBuffer.data(), task.sizes,
                                     task.deviceAddrs.data());
    }
    if (s.Success() && !task.waiter) {
        holder_.push_back(std::move(task));
        return;
    }
    if (s.Success()) {
        holder_.push_back(std::move(task));
        s = stream.Synchronize();
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
