/**
 * MIT License
 *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
 * */
#include "dump_queue.h"
#include <numeric>
#include "logger/logger.h"
#include "trans/device.h"

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
    backend_ = reinterpret_cast<StoreV1*>(config.storeBackend);
    shardSize_ = config.shardSize;
    tensorSizeList_ = config.tensorSizeList;
    tensorSizesByType_ = config.tensorSizesByType;
    waiting_.Setup(config.waitingQueueDepth);
    dumping_.Setup(config.runningQueueDepth);
    dumper_ = std::thread{&DumpQueue::BackendDumpStage, this};
    std::promise<Status> started;
    auto fut = started.get_future();
    dispatcher_ = std::thread{&DumpQueue::DispatchStage, this, config.deviceId, std::ref(started)};
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

void DumpQueue::DispatchStage(int32_t deviceId, std::promise<Status>& started)
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
    waiting_.ConsumerLoop(stop_, &DumpQueue::DispatchOneTask, this, stream.get());
}

void DumpQueue::DispatchOneTask(Trans::Stream* stream, TaskPair&& pair)
{
    auto& task = pair.first;
    auto& waiter = pair.second;
    if (failureSet_->Contains(task->id)) {
        waiter->Done();
        return;
    }
    auto s = task->type == TransTask::Type::DUMP ? DumpTaskDesc(stream, task, waiter)
                                                 : DumpTokenTaskDesc(stream, task, waiter);
    if (s.Success()) { return; }
    UC_ERROR("Failed({}) to run memory dump task({}).", s, task->id);
    failureSet_->Insert(task->id);
    waiter->Done();
}

Status DumpQueue::DumpTaskDesc(Trans::Stream* stream, TaskPtr task, WaiterPtr waiter)
{
    const auto nShard = task->desc.size();
    if (nShard == 0) {
        waiter->Done();
        return Status::OK();
    }
    if (task->desc.prerequisiteHandle != 0) {
        return Status::Unsupported();
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
        auto s = DeviceToHostGatherAsync(stream, shard.addrs, tensorSizeList_,
                                         backendShard.full.data());
        if (s.Failure()) { return s; }
        dumpTask.backendShards.push_back(std::move(backendShard));
    }
    auto s = stream->Synchronized();
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

Status DumpQueue::DumpTokenTaskDesc(Trans::Stream* stream, TaskPtr task, WaiterPtr waiter)
{
    const auto nItem = task->tokenDesc.size();
    if (nItem == 0) {
        waiter->Done();
        return Status::OK();
    }
    if (task->tokenDesc.prerequisiteHandle != 0) {
        return Status::Unsupported();
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
        auto s = DeviceToHostGatherAsync(stream, item.addrs, iter->second,
                                         tokenCopy.data.data());
        if (s.Failure()) { return s; }
        tokenCopies.push_back(std::move(tokenCopy));
    }
    auto s = stream->Synchronized();
    if (s.Failure()) { return s; }

    DumpTask dumpTask;
    dumpTask.taskHandle = task->id;
    dumpTask.waiter = waiter;
    Detail::TaskDesc backendTask;
    backendTask.brief = "Memory2Backend";
    for (auto& tokenCopy : tokenCopies) {
        std::vector<std::byte> full;
        size_t physicalShard = 0;
        s = buffer_->CommitToken(tokenCopy.item, tokenCopy.data, &full, &physicalShard);
        if (s.Failure()) { return s; }
        if (full.empty() || !backend_) { continue; }
        bool replaced = false;
        for (auto& shard : dumpTask.backendShards) {
            if (shard.block != tokenCopy.item.owner || shard.layer != physicalShard) {
                continue;
            }
            shard.full = std::move(full);
            replaced = true;
            break;
        }
        if (!replaced) {
            dumpTask.backendShards.push_back(
                {tokenCopy.item.owner, physicalShard, std::move(full)});
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

Status DumpQueue::DeviceToHostGatherAsync(Trans::Stream* stream, const std::vector<void*>& device,
                                          const std::vector<size_t>& sizes, void* host)
{
    if (device.size() != sizes.size()) {
        return Status::InvalidParam("invalid source addr number({},{})", device.size(),
                                    sizes.size());
    }
    if (!host) { return Status::InvalidParam("invalid destination host addr"); }
    size_t offset = 0;
    for (size_t i = 0; i < sizes.size(); ++i) {
        if (!device[i]) { return Status::InvalidParam("invalid source addr"); }
        auto pHost = static_cast<void*>(static_cast<int8_t*>(host) + offset);
        auto s = stream->DeviceToHostAsync(device[i], pHost, sizes[i]);
        if (s.Failure()) { return s; }
        offset += sizes[i];
    }
    return Status::OK();
}

void DumpQueue::BackendDumpStage()
{
    dumping_.ConsumerLoop(stop_, [this](DumpTask&& task) {
        if (failureSet_->Contains(task.taskHandle)) {
            if (task.waiter) { task.waiter->Done(); }
            return;
        }
        if (task.backendTaskHandle > finishedBackendTaskHandle_) {
            auto s = backend_->Wait(task.backendTaskHandle);
            if (s.Failure()) {
                UC_ERROR("Failed({}) to wait backend({}) for memory dump task({}).", s,
                         task.backendTaskHandle, task.taskHandle);
                failureSet_->Insert(task.taskHandle);
            } else {
                finishedBackendTaskHandle_ = task.backendTaskHandle;
            }
        }
        if (task.waiter) { task.waiter->Done(); }
    });
}

}  // namespace UC::MemoryStore
