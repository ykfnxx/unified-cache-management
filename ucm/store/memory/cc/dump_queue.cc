/**
 * MIT License
 *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
 * */
#include "dump_queue.h"
#include "logger/logger.h"

namespace UC::MemoryStore {

DumpQueue::~DumpQueue()
{
    stop_.store(true);
    if (dispatcher_.joinable()) { dispatcher_.join(); }
}

Status DumpQueue::Setup(const Config& config, TaskIdSet* failureSet, TransBuffer* buffer)
{
    failureSet_ = failureSet;
    buffer_ = buffer;
    waiting_.Setup(config.waitingQueueDepth);
    dispatcher_ = std::thread{&DumpQueue::DispatchStage, this};
    return Status::OK();
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
    auto s = Status::OK();
    if (task->type == TransTask::Type::DUMP) {
        s = buffer_->Dump(task->desc);
    } else {
        s = buffer_->DumpTokens(task->tokenDesc);
    }
    if (s.Failure()) {
        UC_ERROR("Failed({}) to run memory dump task({}).", s, task->id);
        failureSet_->Insert(task->id);
    }
    waiter->Done();
}

}  // namespace UC::MemoryStore
