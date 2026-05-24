/**
 * MIT License
 *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
 * */
#include "load_queue.h"
#include "logger/logger.h"

namespace UC::MemoryStore {

LoadQueue::~LoadQueue()
{
    stop_.store(true);
    if (dispatcher_.joinable()) { dispatcher_.join(); }
}

Status LoadQueue::Setup(const Config& config, TaskIdSet* failureSet, TransBuffer* buffer)
{
    failureSet_ = failureSet;
    buffer_ = buffer;
    waiting_.Setup(config.waitingQueueDepth);
    dispatcher_ = std::thread{&LoadQueue::DispatchStage, this};
    return Status::OK();
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
        s = buffer_->Load(task->desc);
    } else {
        s = buffer_->LoadTokens(task->tokenDesc);
    }
    if (s.Failure()) {
        UC_ERROR("Failed({}) to run memory load task({}).", s, task->id);
        failureSet_->Insert(task->id);
    }
    waiter->Done();
}

}  // namespace UC::MemoryStore
