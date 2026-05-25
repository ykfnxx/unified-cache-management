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
}  // namespace

DumpQueue::~DumpQueue()
{
    stop_.store(true);
    if (dispatcher_.joinable()) { dispatcher_.join(); }
    if (dumper_.joinable()) { dumper_.join(); }
}

Status DumpQueue::Setup(const Config& config, TaskIdSet* failureSet, TransBuffer* buffer)
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
    dumping_.Setup(config.runningQueueDepth);
    dumper_ = std::thread{&DumpQueue::BackendDumpStage, this};
    std::promise<Status> started;
    auto fut = started.get_future();
    dispatcher_ = std::thread{&DumpQueue::DispatchStage, this, std::ref(started)};
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

void DumpQueue::DispatchStage(std::promise<Status>& started)
{
    auto s = SetupTransferStreams();
    started.set_value(s);
    if (s.Failure()) { return; }
    if (!cpuAffinityCores_.empty()) {
        s = CpuAffinity::SetCpuAffinity4CurrentThread(cpuAffinityCores_);
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
    auto s = task->type == TransTask::Type::DUMP ? DumpTaskDesc(task, waiter)
                                                 : DumpTokenTaskDesc(task, waiter);
    if (s.Success()) { return; }
    UC_ERROR("Failed({}) to run memory dump task({}).", s, task->id);
    failureSet_->Insert(task->id);
    waiter->Done();
}

Status DumpQueue::DumpTaskDesc(TaskPtr task, WaiterPtr waiter)
{
    const auto nShard = task->desc.size();
    if (nShard == 0) {
        waiter->Done();
        return Status::OK();
    }
    if (task->desc.prerequisiteHandle != 0) {
        auto s = WaitEventOnStreams(reinterpret_cast<void*>(task->desc.prerequisiteHandle));
        if (s.Failure()) { return s; }
    }

    DumpTask dumpTask;
    dumpTask.taskHandle = task->id;
    dumpTask.waiter = waiter;
    dumpTask.backendShards.reserve(nShard);
    for (const auto& shard : task->desc) {
        BackendShard backendShard;
        backendShard.block = shard.owner;
        backendShard.layer = shard.index;
        backendShard.full.resize(shardSize_);
        size_t offset = 0;
        for (size_t i = 0; i < tensorSizes_.size(); ++i) {
            auto stream = NextStream();
            if (!stream) { return Status::Error("invalid memory dump stream"); }
            auto pHost = static_cast<void*>(backendShard.full.data() + offset);
            auto s = stream->DeviceToHostAsync(shard.addrs[i], pHost, tensorSizes_[i]);
            if (s.Failure()) { return s; }
            offset += tensorSizes_[i];
        }
        dumpTask.backendShards.push_back(std::move(backendShard));
    }
    auto s = SynchronizeStreams();
    if (s.Failure()) { return s; }

    Detail::TaskDesc backendTask;
    backendTask.brief = "Memory2Backend";
    for (auto& shard : dumpTask.backendShards) {
        s = buffer_->CommitFull(shard.block, shard.layer, shard.full);
        if (s.Failure()) { return s; }
        if (!backend_) { continue; }
        backendTask.push_back({shard.block, shard.layer, {shard.full.data()}});
    }
    if (backendTask.empty()) {
        waiter->Done();
        return Status::OK();
    }
    auto res = backend_->Dump(std::move(backendTask));
    if (!res) { return res.Error(); }
    dumpTask.backendTaskHandle = res.Value();
    dumping_.Push(std::move(dumpTask));
    return Status::OK();
}

Status DumpQueue::DumpTokenTaskDesc(TaskPtr task, WaiterPtr waiter)
{
    const auto nItem = task->tokenDesc.size();
    if (nItem == 0) {
        waiter->Done();
        return Status::OK();
    }
    if (task->tokenDesc.prerequisiteHandle != 0) {
        auto s = WaitEventOnStreams(reinterpret_cast<void*>(task->tokenDesc.prerequisiteHandle));
        if (s.Failure()) { return s; }
    }

    struct TokenCopy {
        Detail::TokenLayerShard item;
        std::vector<std::byte> data;
    };

    std::vector<TokenCopy> tokenCopies;
    tokenCopies.reserve(nItem);
    for (const auto& item : task->tokenDesc) {
        auto iter = tensorSizesByType_.find(item.tensorType);
        if (iter == tensorSizesByType_.end()) {
            return Status::InvalidParam("invalid tensor type({})", item.tensorType);
        }
        TokenCopy tokenCopy;
        tokenCopy.item = item;
        tokenCopy.data.resize(Sum(iter->second));
        size_t offset = 0;
        for (size_t i = 0; i < iter->second.size(); ++i) {
            auto stream = NextStream();
            if (!stream) { return Status::Error("invalid memory dump stream"); }
            auto pHost = static_cast<void*>(tokenCopy.data.data() + offset);
            auto s = stream->DeviceToHostAsync(item.addrs[i], pHost, iter->second[i]);
            if (s.Failure()) { return s; }
            offset += iter->second[i];
        }
        tokenCopies.push_back(std::move(tokenCopy));
    }
    auto s = SynchronizeStreams();
    if (s.Failure()) { return s; }

    DumpTask dumpTask;
    dumpTask.taskHandle = task->id;
    dumpTask.waiter = waiter;
    Detail::TaskDesc backendTask;
    backendTask.brief = "Memory2Backend";
    for (auto& tokenCopy : tokenCopies) {
        std::vector<std::byte> full;
        s = buffer_->CommitToken(tokenCopy.item, tokenCopy.data, &full);
        if (s.Failure()) { return s; }
        if (full.empty() || !backend_) { continue; }
        bool replaced = false;
        for (auto& shard : dumpTask.backendShards) {
            if (shard.block != tokenCopy.item.owner || shard.layer != tokenCopy.item.layer) {
                continue;
            }
            shard.full = std::move(full);
            replaced = true;
            break;
        }
        if (!replaced) {
            dumpTask.backendShards.push_back(
                {tokenCopy.item.owner, tokenCopy.item.layer, std::move(full)});
        }
    }
    if (dumpTask.backendShards.empty()) {
        waiter->Done();
        return Status::OK();
    }
    for (auto& shard : dumpTask.backendShards) {
        backendTask.push_back({shard.block, shard.layer, {shard.full.data()}});
    }
    auto res = backend_->Dump(std::move(backendTask));
    if (!res) { return res.Error(); }
    dumpTask.backendTaskHandle = res.Value();
    dumping_.Push(std::move(dumpTask));
    return Status::OK();
}

Status DumpQueue::SetupTransferStreams()
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

std::shared_ptr<Trans::Stream> DumpQueue::NextStream() noexcept
{
    if (streams_.empty()) { return nullptr; }
    auto& stream = streams_[streamIndex_];
    streamIndex_ = (streamIndex_ + 1) % streams_.size();
    return stream;
}

Status DumpQueue::WaitEventOnStreams(void* event) noexcept
{
    auto status = Status::OK();
    for (auto& stream : streams_) {
        auto s = stream->WaitEvent(event);
        if (s.Success()) { continue; }
        UC_ERROR("Failed({}) to wait event on memory dump stream on device({}).", s, deviceId_);
        if (status.Success()) { status = s; }
    }
    return status;
}

Status DumpQueue::SynchronizeStreams() noexcept
{
    auto status = Status::OK();
    for (auto& stream : streams_) {
        auto s = stream->Synchronized();
        if (s.Success()) { continue; }
        UC_ERROR("Failed({}) to synchronize memory dump stream on device({}).", s, deviceId_);
        if (status.Success()) { status = s; }
    }
    return status;
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
        auto s = backend_->Wait(task.backendTaskHandle);
        if (s.Failure()) {
            UC_ERROR("Failed({}) to wait backend({}) for memory dump task({}).", s,
                     task.backendTaskHandle, task.taskHandle);
            failureSet_->Insert(task.taskHandle);
        }
        if (task.waiter) { task.waiter->Done(); }
    });
}

}  // namespace UC::MemoryStore
