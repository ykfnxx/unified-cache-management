/**
 * MIT License
 *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 * */
#include "dump_queue.h"
#include <algorithm>
#include <atomic>
#include <memory>
#include "logger/logger.h"
#include "metrics_api.h"
#include "thread/cpu_affinity.h"

namespace UC::Context {

DumpQueue::~DumpQueue()
{
    if (dispatcher_.joinable()) {
        waiting_.Push(TaskPair{});
        dispatcher_.join();
    }
}

Status DumpQueue::Setup(const Config& config, TaskIdSet* failureSet, TransBuffer* buffer)
{
    failureSet_ = failureSet;
    buffer_ = buffer;
    deviceId_ = config.deviceId;
    tensorSizes_ = config.tensorSizes;
    streamNumber_ = config.EffectiveStreamNumber();
    useGdr_ = config.useGdr;
    cacheIOAggregation_ = config.cacheIOAggregation;
    cacheSdmaDirect_ = config.cacheSdmaDirect;
    cpuAffinityCores_ = config.cpuAffinityCores;
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
    UC_ERROR("Waiting queue full, submit dump task({}) failed.", task->id);
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("context_transfer_dump_queue_full_total"), 1.0);
    failureSet_->Insert(task->id);
    waiter->Done();
}

void DumpQueue::DispatchStage(std::promise<Status>& started)
{
    auto nameStatus = CpuAffinity::SetCurrentThreadName("ucm_dump_disp");
    if (nameStatus.Failure()) {
        UC_WARN("Failed({}) to set UCM dump dispatcher name.", nameStatus);
    }
    CopyStream stream;
    auto s = Status::OK();
    if (cacheIOAggregation_) {
        s = stream.SetupIoAggregation(deviceId_, useGdr_);
    } else if (cacheSdmaDirect_) {
        s = stream.SetupSdmaDirect(deviceId_, useGdr_);
    } else {
        s = stream.Setup(deviceId_, streamNumber_, useGdr_);
    }
    started.set_value(s);
    if (s.Failure()) [[unlikely]] { return; }
    if (!cpuAffinityCores_.empty()) {
        s = CpuAffinity::SetCpuAffinity4CurrentThread(cpuAffinityCores_);
        if (s.Failure()) { UC_WARN("Failed({}) to set affinity.", s); }
    }
    waiting_.ConsumerLoop(stop_, &DumpQueue::DispatchOneTask, this, stream);
}

void DumpQueue::DispatchOneTask(CopyStream& stream, TaskPair&& pair)
{
    if (!pair.first) {
        stop_.store(true);
        return;
    }
    auto& task = pair.first;
    auto& waiter = pair.second;
    auto wait = NowTime::Now() - waiter->startTp;
    UC_DEBUG("Context task({}) start running, wait {:.3f}ms.", task->id, wait * 1e3);
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("context_transfer_dump_queue_wait_duration_ms"),
                             wait * 1e3);
    if (!failureSet_->Contains(task->id)) {
        auto s = DumpOneTask(stream, task);
        if (s.Failure()) [[unlikely]] {
            task->Fail(s);
            failureSet_->Insert(task->id);
        }
    }
    waiter->Done();
}

Status DumpQueue::DumpOneTask(CopyStream& stream, TaskPtr task)
{
    if (task->desc.prerequisiteHandle) {
        auto s = stream.WaitEvent(Trans::Event{task->desc.prerequisiteHandle});
        if (s.Failure()) { return s; }
    }
    std::vector<TransBuffer::Handle> handles;
    auto status = Status::OK();
    for (auto& shard : task->desc) {
        auto got = buffer_->Get(shard.owner, shard.index);
        if (!got) {
            status = got.Error();
            break;
        }
        auto handle = std::move(got.Value());
        if (!handle.Owner() || handle.Ready()) { continue; }
        auto* host = cacheSdmaDirect_ ? handle.DeviceData() : handle.Data();
        status = DeviceToHostAsync(stream, shard.addrs.data(), host);
        handles.push_back(std::move(handle));
        if (status.Failure()) { break; }
    }
    auto sync = stream.Synchronize();
    if (sync.Failure()) { status = sync; }
    for (auto& handle : handles) {
        if (status.Success()) {
            handle.MarkReady();
        } else {
            handle.MarkFailed(status);
        }
    }
    return status;
}

Status DumpQueue::DeviceToHostAsync(CopyStream& stream, void** device, void* host)
{ return stream.DeviceToHostAsync(device, host, tensorSizes_); }

}  // namespace UC::Context
