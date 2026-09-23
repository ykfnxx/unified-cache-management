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
#include "load_queue.h"
#include "logger/logger.h"
#include "metrics_api.h"
#include "thread/cpu_affinity.h"

namespace UC::Context {

LoadQueue::~LoadQueue()
{
    if (dispatcher_.joinable()) {
        waiting_.Push(TaskPair{});
        dispatcher_.join();
    }
    if (transfer_.joinable()) { transfer_.join(); }
}

Status LoadQueue::Setup(const Config& config, TaskIdSet* failureSet, TransBuffer* buffer)
{
    failureSet_ = failureSet;
    buffer_ = buffer;
    backend_ = config.storeBackend;
    deviceId_ = config.deviceId;
    tensorSizes_ = config.tensorSizes;
    timeoutMs_ = config.timeoutMs;
    streamNumber_ = config.EffectiveStreamNumber();
    useGdr_ = config.useGdr;
    cacheIOAggregation_ = config.cacheIOAggregation;
    cacheSdmaDirect_ = config.cacheSdmaDirect;
    cpuAffinityCores_ = config.cpuAffinityCores;
    localRankSize_ = config.localRankSize;
    nShardPerBlock_ = config.blockSize / config.shardSize;
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
    UC_ERROR("Waiting queue full, submit load task({}) failed.", task->id);
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("context_transfer_load_queue_full_total"), 1.0);
    RecordFailedShards(task->desc.size());
    failureSet_->Insert(task->id);
    waiter->Done();
}

void LoadQueue::DispatchStage()
{
    auto nameStatus = CpuAffinity::SetCurrentThreadName("ucm_load_disp");
    if (nameStatus.Failure()) {
        UC_WARN("Failed({}) to set UCM load dispatcher name.", nameStatus);
    }
    if (!cpuAffinityCores_.empty()) {
        auto s = CpuAffinity::SetCpuAffinity4CurrentThread(cpuAffinityCores_);
        if (s.Failure()) { UC_WARN("Failed({}) to set affinity.", s); }
    }
    waiting_.ConsumerLoop(stop_, &LoadQueue::DispatchOneTask, this);
}

static std::vector<size_t> RearrangeIndex(size_t n, size_t iProc, size_t nProc)
{
    std::vector<size_t> order;
    order.reserve(n);
    for (size_t r = 0; r < nProc; ++r) {
        size_t slice = (iProc + r) % nProc;
        for (size_t j = 0;; ++j) {
            size_t i = slice + j * nProc;
            if (i >= n) { break; }
            order.push_back(i);
        }
    }
    return order;
}

void LoadQueue::DispatchOneTask(TaskPair&& pair)
{
    if (!pair.first) {
        stop_.store(true);
        running_.Push(ShardTask{});
        return;
    }
    auto& task = pair.first;
    auto& waiter = pair.second;
    if (failureSet_->Contains(task->id)) {
        waiter->Done();
        return;
    }
    const auto nShard = task->desc.size();
    const auto indexes = RearrangeIndex(nShard, deviceId_, localRankSize_);
    size_t misses = 0, backendCount = 0;
    auto started = NowTime::Now();
    Metrics::UpdateStats(NAME_TO_METRIC_ID("context_transfer_load_queue_wait_duration_ms"),
                         (started - waiter->startTp) * 1e3);
    for (size_t i = 0; i < nShard; ++i) {
        auto& shard = task->desc[indexes[i]];
        ShardTask item;
        item.task = task;
        auto handle = buffer_->Get(shard.owner, shard.index);
        if (!handle) {
            UC_ERROR("Context Load allocation failed: device={}, task={}, shard={}, status={}",
                     deviceId_, task->id, shard.index, handle.Error());
            RecordLoadSourceShards(i, misses);
            RecordFailedShards(nShard - i);
            Metrics::UpdateStats(NAME_TO_METRIC_ID("context_transfer_load_backend_shards_total"),
                                 backendCount);
            task->Fail(handle.Error());
            failureSet_->Insert(task->id);
            item.waiter = waiter;
            running_.Push(std::move(item));
            return;
        }
        item.bufferHandle = std::move(handle.Value());
        item.wasNotReady = !item.bufferHandle.Ready();
        misses += item.wasNotReady;
        if (item.bufferHandle.Owner() && item.wasNotReady) {
            auto res = backend_->Load({
                {buffer_->BackendKey(shard.owner), shard.index, {item.bufferHandle.Data()}}
            });
            if (!res) {
                UC_ERROR(
                    "Context backend Load submission failed: device={}, task={}, shard={}, "
                    "status={}",
                    deviceId_, task->id, shard.index, res.Error());
                RecordLoadSourceShards(i + 1, misses);
                RecordFailedShards(nShard - i - 1);
                Metrics::UpdateStats(
                    NAME_TO_METRIC_ID("context_transfer_load_backend_shards_total"), backendCount);
                item.bufferHandle.MarkFailed(res.Error());
                task->Fail(res.Error());
                failureSet_->Insert(task->id);
                item.waiter = waiter;
                running_.Push(std::move(item));
                return;
            }
            item.backendTaskHandle = res.Value();
            ++backendCount;
        }
        item.shard = std::move(shard);
        item.waiter = i + 1 == nShard ? waiter : nullptr;
        running_.Push(std::move(item));
    }
    RecordLoadSourceShards(nShard, misses);
    Metrics::UpdateStats(NAME_TO_METRIC_ID("context_transfer_load_backend_shards_total"),
                         backendCount);
    Metrics::UpdateStats(NAME_TO_METRIC_ID("context_transfer_load_backend_submit_duration_ms"),
                         (NowTime::Now() - started) * 1e3);
    for (const auto i : indexes) {
        const auto& shard = task->desc[i];
        if (shard.index + 1 < nShardPerBlock_) { buffer_->Prealloc(shard.owner, shard.index + 1); }
    }
}

void LoadQueue::TransferStage(std::promise<Status>& started)
{
    auto nameStatus = CpuAffinity::SetCurrentThreadName("ucm_load_xfer");
    if (nameStatus.Failure()) { UC_WARN("Failed({}) to set UCM load transfer name.", nameStatus); }
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
    running_.ConsumerLoop(transferStop_, &LoadQueue::TransferOneTask, this, stream);
}

void LoadQueue::TransferOneTask(CopyStream& stream, ShardTask&& item)
{
    if (!item.task) {
        transferStop_.store(true);
        return;
    }
    // Even failed preparation drains backend writes and DMA before releasing handles.
    auto task = item.task;
    auto waiter = item.waiter;
    if (item.bufferHandle) {
        auto waitStart = NowTime::Now();
        auto s = WaitBackendTaskReady(item);
        Metrics::UpdateStats(NAME_TO_METRIC_ID("context_transfer_shard_backend_wait_ms"),
                             (NowTime::Now() - waitStart) * 1e3);
        if (s.Success() && !failureSet_->Contains(task->id)) {
            auto* host =
                cacheSdmaDirect_ ? item.bufferHandle.DeviceData() : item.bufferHandle.Data();
            s = HostToDeviceAsync(stream, host, item.shard.addrs.data());
        }
        if (s.Failure()) {
            task->Fail(s);
            failureSet_->Insert(task->id);
        }
    }
    if (!waiter) {
        holder_.push_back(std::move(item));
        return;
    }
    auto start = NowTime::Now();
    auto s = stream.Synchronize();
    RecordH2dSyncMetrics((NowTime::Now() - start) * 1e3);
    if (s.Failure()) {
        task->Fail(s);
        failureSet_->Insert(task->id);
    }
    const auto* extra = item.bufferHandle ? &item : nullptr;
    RecordShardResults(holder_, extra, !failureSet_->Contains(task->id));
    if (!failureSet_->Contains(task->id)) {
        buffer_->RecordRead(holder_.size() + (extra != nullptr));
    }
    holder_.clear();
    waiter->Done();
}

Status LoadQueue::WaitBackendTaskReady(ShardTask& task)
{
    if (task.backendTaskHandle != 0) {
        auto s = backend_->Wait(task.backendTaskHandle);
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Failed({}) to wait backend({}) for task({}).", s, task.backendTaskHandle,
                     task.task->id);
            UC::Metrics::UpdateStats(
                NAME_TO_METRIC_ID("context_transfer_backend_load_wait_errors_total"), 1.0);
            task.bufferHandle.MarkFailed(s);
            return s;
        }
        task.bufferHandle.MarkReady(true);
        return Status::OK();
    }
    if (task.bufferHandle.Ready()) { return Status::OK(); }
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs_);
    for (;;) {
        auto state = task.bufferHandle.GetState();
        if (state == TransBuffer::State::READY) { return Status::OK(); }
        if (state == TransBuffer::State::FAILED) { return task.bufferHandle.FailureStatus(); }
        if (failureSet_->Contains(task.task->id)) { return task.task->FailureStatus(); }
        if (std::chrono::steady_clock::now() >= deadline) { return Status::Timeout(); }
        std::this_thread::yield();
    }
}

Status LoadQueue::HostToDeviceAsync(CopyStream& stream, void* host, void** device)
{ return stream.HostToDeviceAsync(host, device, tensorSizes_); }

void LoadQueue::RecordShardResults(const std::vector<ShardTask>& tasks, const ShardTask* extra,
                                   bool success) const
{
    size_t cache = 0;
    size_t posix = 0;
    for (const auto& task : tasks) {
        if (task.wasNotReady) {
            ++posix;
        } else {
            ++cache;
        }
    }
    if (extra != nullptr) {
        if (extra->wasNotReady) {
            ++posix;
        } else {
            ++cache;
        }
    }
    if (!success) {
        RecordFailedShards(cache + posix);
        return;
    }
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("context_transfer_load_success_shards_total"),
                             static_cast<double>(cache));
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("context_transfer_wait_load_success_shards_total"),
                             static_cast<double>(posix));
}

void LoadQueue::RecordFailedShards(size_t count) const
{
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("context_transfer_load_failed_shards_total"),
                             static_cast<double>(count));
}

void LoadQueue::RecordLoadSourceShards(size_t total, size_t wait) const
{
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("context_transfer_load_shards_total"),
                             static_cast<double>(total));
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("context_transfer_load_wait_shards_total"),
                             static_cast<double>(wait));
}

void LoadQueue::RecordH2dSyncMetrics(double h2dSyncMs) const
{ UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("context_transfer_h2d_sync_ms"), h2dSyncMs); }

}  // namespace UC::Context
