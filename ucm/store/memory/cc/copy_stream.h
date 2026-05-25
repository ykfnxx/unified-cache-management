/**
 * MIT License
 *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
 * */
#ifndef UNIFIEDCACHE_MEMORY_STORE_CC_COPY_STREAM_H
#define UNIFIEDCACHE_MEMORY_STORE_CC_COPY_STREAM_H

#include <memory>
#include <vector>
#include "logger/logger.h"
#include "status/status.h"
#include "trans/device.h"

namespace UC::MemoryStore {

class CopyStream {
    int32_t deviceId_{0};
    size_t streamNumber_{0};
    size_t streamIndex_{0};
    std::vector<std::shared_ptr<Trans::Stream>> streams_;

public:
    Status Setup(int32_t deviceId, size_t streamNumber, bool useGdr)
    {
        Trans::Device device;
        auto s = device.Setup(deviceId);
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Failed({}) to setup memory copy device({}).", s, deviceId);
            return s;
        }
        streams_.reserve(streamNumber);
        for (size_t i = 0; i < streamNumber; ++i) {
            std::shared_ptr<Trans::Stream> stream =
                useGdr ? device.MakeGdrStream() : device.MakeSharedStream();
            if (!stream) [[unlikely]] {
                UC_ERROR("Failed to make memory copy stream on device({}).", deviceId);
                return Status::Error();
            }
            streams_.push_back(std::move(stream));
        }
        deviceId_ = deviceId;
        streamNumber_ = streamNumber;
        return Status::OK();
    }

    std::shared_ptr<Trans::Stream> NextStream() noexcept
    {
        if (streamNumber_ == 0) [[unlikely]] { return nullptr; }
        auto& stream = streams_[streamIndex_];
        streamIndex_ = (streamIndex_ + 1) % streamNumber_;
        return stream;
    }

    Status WaitEvent(void* event) noexcept
    {
        auto status = Status::OK();
        for (auto& stream : streams_) {
            auto s = stream->WaitEvent(event);
            if (s.Success()) { continue; }
            UC_ERROR("Failed({}) to wait memory copy event on device({}).", s, deviceId_);
            if (status.Success()) { status = s; }
        }
        return status;
    }

    Status Synchronize() noexcept
    {
        auto status = Status::OK();
        for (auto& stream : streams_) {
            auto s = stream->Synchronized();
            if (s.Success()) { continue; }
            UC_ERROR("Failed({}) to synchronize memory copy stream on device({}).", s, deviceId_);
            if (status.Success()) { status = s; }
        }
        return status;
    }
};

}  // namespace UC::MemoryStore

#endif
