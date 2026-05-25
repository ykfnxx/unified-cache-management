/**
 * MIT License
 *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
 * */
#include "dump_queue.h"
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

DumpQueue::~DumpQueue()
{
    stop_.store(true);
    if (dispatcher_.joinable()) { dispatcher_.join(); }
    if (transfer_.joinable()) { transfer_.join(); }
    if (dumper_.joinable()) { dumper_.join(); }
}

Status DumpQueue::Setup(const Config& config, TaskIdSet* failureSet, TransBuffer* buffer)
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
    dumping_.Setup(config.runningQueueDepth);
    holder_.reserve(1024);
    dispatcher_ = std::thread{&DumpQueue::DispatchStage, this};
    std::promise<Status> started;
    auto fut = started.get_future();
    transfer_ = std::thread{&DumpQueue::TransferStage, this, std::ref(started)};
    dumper_ = std::thread{&DumpQueue::BackendDumpStage, this};
    return fut.get();
}

void DumpQueue::Submit(TaskPtr task, WaiterPtr waiter)
{
    waiter->Up();
    auto success = waiting_.TryPush({task, waiter});
    if (success) { return; }
    UC_ERROR("Waiting queue full, submit memory dump task({}) failed.", task->id);
    failureSet_->Insert(task->id);
    waiter->Done();
}

void DumpQueue::DispatchStage()
{
    if (!cpuAffinityCores_.empty()) {
        auto s = CpuAffinity::SetCpuAffinity4CurrentThread(cpuAffinityCores_);
        if (s.Failure()) { UC_WARN("Failed({}) to set affinity.", s); }
    }
    waiting_.ConsumerLoop(stop_, &DumpQueue::DispatchOneTask, this);
}

void DumpQueue::DispatchOneTask(TaskPair&& pair)
{
    auto& task = pair.first;
    auto& waiter = pair.second;
    if (failureSet_->Contains(task->id)) {
        waiter->Done();
        return;
    }
    if (task->type == TransTask::Type::DUMP) {
        const auto nShard = task->desc.size();
        if (nShard == 0) {
            waiter->Done();
            return;
        }
        for (size_t i = 0; i < nShard; ++i) {
            const auto& shard = task->desc[i];
            DumpTask dumpTask;
            dumpTask.taskHandle = task->id;
            dumpTask.hostTask.brief = task->desc.brief;
            dumpTask.hostBuffer.resize(Sum(tensorSizes_));
            auto hostAddrs = BuildHostAddrs(dumpTask.hostBuffer, tensorSizes_);
            dumpTask.hostTask.push_back({shard.owner, shard.index, hostAddrs});
            dumpTask.waiter = (i + 1 < nShard) ? nullptr : waiter;
            running_.Push({task->id, tensorSizes_, shard.addrs,
                           i == 0 && task->desc.prerequisiteHandle != 0,
                           task->desc.prerequisiteHandle, std::move(dumpTask)});
        }
        return;
    }

    const auto nItem = task->tokenDesc.size();
    if (nItem == 0) {
        waiter->Done();
        return;
    }
    for (size_t i = 0; i < nItem; ++i) {
        const auto& item = task->tokenDesc[i];
        const auto& sizes = tensorSizesByType_.at(item.tensorType);
        DumpTask dumpTask;
        dumpTask.taskHandle = task->id;
        dumpTask.hostTokenTask.brief = task->tokenDesc.brief;
        dumpTask.hostBuffer.resize(Sum(sizes));
        auto hostAddrs = BuildHostAddrs(dumpTask.hostBuffer, sizes);
        dumpTask.hostTokenTask.push_back(
            {item.owner, item.layer, item.tokenOffset, item.tensorType, hostAddrs});
        dumpTask.waiter = (i + 1 < nItem) ? nullptr : waiter;
        running_.Push({task->id, sizes, item.addrs,
                       i == 0 && task->tokenDesc.prerequisiteHandle != 0,
                       task->tokenDesc.prerequisiteHandle, std::move(dumpTask)});
    }
}

void DumpQueue::TransferStage(std::promise<Status>& started)
{
    Trans::CopyStream stream;
    auto s = stream.Setup(deviceId_, streamNumber_, useGdr_);
    started.set_value(s);
    if (s.Failure()) { return; }
    if (!cpuAffinityCores_.empty()) {
        s = CpuAffinity::SetCpuAffinity4CurrentThread(cpuAffinityCores_);
        if (s.Failure()) { UC_WARN("Failed({}) to set affinity.", s); }
    }
    running_.ConsumerLoop(stop_, &DumpQueue::TransferOneTask, this, stream);
}

void DumpQueue::TransferOneTask(Trans::CopyStream& stream, CopyTask&& task)
{
    if (failureSet_->Contains(task.taskHandle)) {
        if (task.dumpTask.waiter) { task.dumpTask.waiter->Done(); }
        return;
    }
    auto s = Status::OK();
    if (task.waitPrerequisite) {
        s = stream.WaitEvent(reinterpret_cast<void*>(task.prerequisiteHandle));
    }
    auto copyStream = stream.NextStream();
    if (s.Success() && !copyStream) { s = Status::Error("invalid memory dump stream"); }
    if (s.Success()) {
        s = DeviceToHostGatherAsync(copyStream, task.deviceAddrs.data(), task.sizes,
                                    task.dumpTask.hostBuffer.data());
    }
    if (s.Success() && !task.dumpTask.waiter) {
        holder_.push_back(std::move(task.dumpTask));
        return;
    }
    if (s.Success()) {
        holder_.push_back(std::move(task.dumpTask));
        s = stream.Synchronize();
    }
    if (s.Failure()) {
        UC_ERROR("Failed({}) to run memory dump transfer task({}).", s, task.taskHandle);
        failureSet_->Insert(task.taskHandle);
        auto waiter = task.dumpTask.waiter;
        if (!waiter && !holder_.empty()) { waiter = holder_.back().waiter; }
        holder_.clear();
        if (waiter) { waiter->Done(); }
        return;
    }
    for (auto& dumpTask : holder_) { dumping_.Push(std::move(dumpTask)); }
    holder_.clear();
}

void DumpQueue::BackendDumpStage()
{
    if (!cpuAffinityCores_.empty()) {
        auto s = CpuAffinity::SetCpuAffinity4CurrentThread(cpuAffinityCores_);
        if (s.Failure()) { UC_WARN("Failed({}) to set affinity.", s); }
    }
    dumping_.ConsumerLoop(stop_, [this](DumpTask&& task) {
        if (failureSet_->Contains(task.taskHandle)) {
            if (task.waiter) { task.waiter->Done(); }
            return;
        }
        auto s = task.hostTask.empty() ? buffer_->DumpTokens(task.hostTokenTask)
                                       : buffer_->Dump(task.hostTask);
        if (s.Failure()) {
            UC_ERROR("Failed({}) to run memory backend dump task({}).", s, task.taskHandle);
            failureSet_->Insert(task.taskHandle);
        }
        if (task.waiter) { task.waiter->Done(); }
    });
}

Status DumpQueue::DeviceToHostGatherAsync(std::shared_ptr<Trans::Stream> stream, void** device,
                                          const std::vector<size_t>& sizes, void* host)
{
    size_t offset = 0;
    for (size_t i = 0; i < sizes.size(); ++i) {
        auto pHost = static_cast<void*>(static_cast<int8_t*>(host) + offset);
        auto s = stream->DeviceToHostAsync(device[i], pHost, sizes[i]);
        if (s.Failure()) { return s; }
        offset += sizes[i];
    }
    return Status::OK();
}

}  // namespace UC::MemoryStore
