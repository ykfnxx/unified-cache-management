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
#pragma once
#include <vector>
#include "trans/buffer.h"
#include "trans/device.h"
namespace UC::Context {
class BufferPool {
public:
    ~BufferPool()
    {
        if (mapped_) { Trans::Buffer::UnregisterHostBuffer(data_.get()); }
    }
    Status Setup(int deviceId, size_t count, size_t bytes, bool mapped)
    {
        Trans::Device device;
        auto status = device.Setup(deviceId);
        if (status.Failure()) { return status; }
        auto allocator = device.MakeBuffer();
        if (!allocator) { return Status::Error("context buffer allocator unavailable"); }
        data_ = allocator->MakeHostBuffer(count * bytes);
        if (!data_) { return Status::OutOfMemory(); }
        bytes_ = bytes;
        deviceData_ = data_.get();
        if (mapped) {
            status = Trans::Buffer::RegisterHostBuffer(data_.get(), count * bytes, &deviceData_);
            if (status.Failure()) { return status; }
            mapped_ = true;
        }
        for (size_t i = count; i > 0; --i) { free_.push_back(i - 1); }
        return Status::OK();
    }
    bool Empty() const { return free_.empty(); }
    size_t FreeCount() const { return free_.size(); }
    size_t Allocate()
    {
        auto i = free_.back();
        free_.pop_back();
        return i;
    }
    void Release(size_t i) { free_.push_back(i); }
    void* Data(size_t i) { return static_cast<char*>(data_.get()) + i * bytes_; }
    void* CopyAddress(size_t i) { return static_cast<char*>(deviceData_) + i * bytes_; }

private:
    std::shared_ptr<void> data_;
    void* deviceData_ = nullptr;
    size_t bytes_ = 0;
    bool mapped_ = false;
    std::vector<size_t> free_;
};
}  // namespace UC::Context
