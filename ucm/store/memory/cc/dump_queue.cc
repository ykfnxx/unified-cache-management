/**
 * MIT License
 *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
 * */
#include "dump_queue.h"
#include <numeric>
#include "logger/logger.h"

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
    waiting_.Setup(config.waitingQueueDepth);
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
    CopyStream stream;
    auto s = stream.Setup(deviceId_, streamNumber_, useGdr_);
    started.set_value(s);
    if (s.Failure()) { return; }
    waiting_.ConsumerLoop(stop_, &DumpQueue::DispatchOneTask, this, stream);
}

void DumpQueue::DispatchOneTask(CopyStream& stream, TaskPair&& pair)
{
    auto& task = pair.first;
    auto& waiter = pair.second;
    if (failureSet_->Contains(task->id)) {
        waiter->Done();
        return;
    }
    auto s = Status::OK();
    auto copyStream = stream.NextStream();
    if (!copyStream) {
        s = Status::Error("invalid memory dump stream");
    } else if (task->type == TransTask::Type::DUMP) {
        Detail::TaskDesc hostTask;
        hostTask.brief = task->desc.brief;
        for (const auto& shard : task->desc) {
            std::vector<std::byte> hostBuffer(Sum(tensorSizes_));
            s = DeviceToHostGatherAsync(copyStream, const_cast<void**>(shard.addrs.data()),
                                        tensorSizes_, hostBuffer.data());
            if (s.Failure()) { break; }
            auto hostAddrs = BuildHostAddrs(hostBuffer, tensorSizes_);
            hostTask.clear();
            hostTask.push_back({shard.owner, shard.index, hostAddrs});
            s = buffer_->Dump(hostTask);
            if (s.Failure()) { break; }
        }
    } else {
        Detail::TokenLayerTaskDesc hostTask;
        hostTask.brief = task->tokenDesc.brief;
        for (const auto& item : task->tokenDesc) {
            const auto& sizes = tensorSizesByType_.at(item.tensorType);
            std::vector<std::byte> hostBuffer(Sum(sizes));
            s = DeviceToHostGatherAsync(copyStream, const_cast<void**>(item.addrs.data()), sizes,
                                        hostBuffer.data());
            if (s.Failure()) { break; }
            auto hostAddrs = BuildHostAddrs(hostBuffer, sizes);
            hostTask.clear();
            hostTask.push_back(
                {item.owner, item.layer, item.tokenOffset, item.tensorType, hostAddrs});
            s = buffer_->DumpTokens(hostTask);
            if (s.Failure()) { break; }
        }
    }
    if (s.Failure()) {
        UC_ERROR("Failed({}) to run memory dump task({}).", s, task->id);
        failureSet_->Insert(task->id);
    }
    waiter->Done();
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
    return stream->Synchronized();
}

}  // namespace UC::MemoryStore
