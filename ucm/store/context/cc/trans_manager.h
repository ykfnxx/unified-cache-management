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
#ifndef UNIFIEDCACHE_CONTEXT_STORE_CC_TRANS_MANAGER_H
#define UNIFIEDCACHE_CONTEXT_STORE_CC_TRANS_MANAGER_H

#include "dump_queue.h"
#include "load_queue.h"
#include "logger/logger.h"
#include "metrics_api.h"
#include "template/task_wrapper.h"
#include "trans_task.h"

namespace UC::Context {

class TransManager : public Detail::TaskWrapper<TransTask, Detail::TaskHandle> {
    size_t shardSize_;
    TransBuffer* buffer_;
    LoadQueue loadQ_;
    DumpQueue dumpQ_;

public:
    Status Setup(const Config& config, TransBuffer* buffer)
    {
        buffer_ = buffer;
        timeoutMs_ = config.timeoutMs;
        shardSize_ = config.shardSize;
        auto s = loadQ_.Setup(config, &failureSet_, buffer);
        if (s.Failure()) [[unlikely]] { return s; }
        return dumpQ_.Setup(config, &failureSet_, buffer);
    }

protected:
    Status FailureStatus(const TaskPtr& task) const override { return task->FailureStatus(); }
    void Dispatch(TaskPtr t, WaiterPtr w) override
    {
        const auto id = t->id;
        const auto& brief = t->desc.brief;
        const auto num = t->desc.size();
        const auto size = shardSize_ * num;
        const auto tp = w->startTp;
        const auto isLoad = t->type == TransTask::Type::LOAD;
        UC_DEBUG("Context task({},{},{},{}) dispatching.", id, brief, num, size);
        w->SetEpilog([this, id, brief = std::move(brief), num, size, tp, isLoad] {
            buffer_->EndTask();
            auto cost = NowTime::Now() - tp;
            auto costMs = cost * 1e3;
            auto bwGbps = cost > 0 ? static_cast<double>(size) / cost / 1e9 : 0.0;
            UC_DEBUG("Context task({},{},{},{}) finished, cost {:.3f}ms.", id, brief, num, size,
                     costMs);
            if (isLoad) {
                UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("context_transfer_load_duration_ms"),
                                         costMs);
                UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("context_transfer_load_bandwidth_gbps"),
                                         bwGbps);
                UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("context_transfer_load_bytes_total"),
                                         static_cast<double>(size));
            } else {
                UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("context_transfer_dump_duration_ms"),
                                         costMs);
                UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("context_transfer_dump_bandwidth_gbps"),
                                         bwGbps);
                UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("context_transfer_dump_shards_total"),
                                         static_cast<double>(num));
                UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("context_transfer_dump_bytes_total"),
                                         static_cast<double>(size));
            }
        });
        if (t->type == TransTask::Type::LOAD) {
            loadQ_.Submit(t, w);
        } else {
            dumpQ_.Submit(t, w);
        }
    }
};

}  // namespace UC::Context

#endif
