/**
 * MIT License
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
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
 */
#pragma once

#include <chrono>
#include <functional>
#include "template/task_wrapper.h"
#include "thread/thread_pool.h"
#include "trans/device.h"
#include "type/types.h"

namespace UC::OnEvictCacheStore {

struct TransferTask {
    Detail::TaskHandle id;
    std::function<Status(Trans::Stream&)> run;
    Status status;
    const char* operation{"internal"};
    size_t shards{0};
    size_t bytes{0};
    std::chrono::steady_clock::time_point submitted{std::chrono::steady_clock::now()};
};

// One worker preserves submission order; TaskWrapper owns completion and errors.
class TransferQueue : public Detail::TaskWrapper<TransferTask, Detail::TaskHandle> {
public:
    Status Setup(int32_t deviceId, size_t timeoutMs)
    {
        timeoutMs_ = timeoutMs;
        worker_.SetNWorker(1).SetWorkerFn([this, deviceId](TaskPair& pair, void* const&) {
            auto& [task, waiter] = pair;
            const auto started = std::chrono::steady_clock::now();
            UC_DEBUG("OnEvict task={} op={} start shards={} bytes={} queue_ms={:.3f}", task->id,
                     task->operation, task->shards, task->bytes,
                     std::chrono::duration<double, std::milli>(started - task->submitted).count());
            if (!stream_) {
                Trans::Device device;
                task->status = device.Setup(deviceId);
                if (task->status.Success()) { stream_ = device.MakeStream(); }
                if (task->status.Success() && !stream_) {
                    task->status = Status::Error("create stream failed");
                }
            }
            if (task->status.Success()) { task->status = task->run(*stream_); }
            if (task->status.Failure()) {
                UC_ERROR("OnEvict task={} op={} failed device={} shards={} bytes={} status={}",
                         task->id, task->operation, deviceId, task->shards, task->bytes,
                         task->status);
                failureSet_.Insert(task->id);
            }
            UC_DEBUG("OnEvict task={} op={} complete status={} run_ms={:.3f}", task->id,
                     task->operation, task->status,
                     std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                               started)
                         .count());
            waiter->Done();
        });
        worker_.SetWorkerExitFn([this](void*&) { stream_.reset(); });
        return worker_.Run() ? Status::OK() : Status::Error("start KV transfer worker failed");
    }
    ~TransferQueue()
    {
        while (!tasks_.empty()) { Wait(tasks_.begin()->first); }
    }

private:
    void Dispatch(TaskPtr task, WaiterPtr waiter) override
    {
        waiter->Up();
        worker_.Push(TaskPair{std::move(task), std::move(waiter)});
    }
    Status FailureStatus(const TaskPtr& task) const override { return task->status; }

    std::unique_ptr<Trans::Stream> stream_;
    ThreadPool<TaskPair> worker_;
};

}  // namespace UC::OnEvictCacheStore
