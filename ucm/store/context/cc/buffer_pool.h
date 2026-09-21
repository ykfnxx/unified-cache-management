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
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>
#include "trans/buffer.h"
#include "trans/device.h"
namespace UC::Context {
class BufferPool {
public:
    ~BufferPool()
    {
        if (mapped_) { Trans::Buffer::UnregisterHostBuffer(data_.get()); }
        if (created_) { shm_unlink(name_.c_str()); }
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
    // Like CacheStore, each device registers its own mapping of the same payload.
    Status SetupShared(const std::string& name, int deviceId, size_t count, size_t bytes,
                       bool owner, bool deviceAddress)
    {
        name_ = "/ucm_context_" + name;
        owner_ = owner;
        deviceId_ = deviceId;
        bytes_ = bytes;
        total_ = count * bytes;
        deviceAddress_ = deviceAddress;
        if (!owner) { return Status::OK(); }  // Readers may initialize before rank 0.
        auto status = MapShared();
        if (status.Failure()) { return status; }
        for (size_t i = count; i > 0; --i) { free_.push_back(i - 1); }
        return Status::OK();
    }
    Status MapShared()
    {
        if (data_) { return Status::OK(); }
        Trans::Device device;
        auto status = device.Setup(deviceId_);
        if (status.Failure()) { return status; }
        int fd = shm_open(name_.c_str(), O_RDWR | (owner_ ? O_CREAT | O_EXCL : 0), 0600);
        if (fd < 0) { return Status::Error("cannot open context shared payload"); }
        if (owner_) { created_ = true; }
        if (owner_ && ftruncate(fd, total_) != 0) {
            close(fd);
            return Status::Error("cannot size context shared payload");
        }
        // Reserve tmpfs backing now: otherwise a later DMA/memcpy could SIGBUS.
        if (owner_ && posix_fallocate(fd, 0, total_) != 0) {
            close(fd);
            return Status::NoSpace();
        }
        struct stat info{};
        if (fstat(fd, &info) != 0 || size_t(info.st_size) != total_) {
            close(fd);
            return Status::InvalidParam("context shared payload layout mismatch");
        }
        void* address = mmap(nullptr, total_, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        close(fd);
        if (address == MAP_FAILED) { return Status::OutOfMemory(); }
        void* deviceAddress = nullptr;
        status = Trans::Buffer::RegisterHostBuffer(address, total_, &deviceAddress);
        if (status.Failure()) {
            munmap(address, total_);
            return status;
        }
        data_ = std::shared_ptr<void>(address, [size = total_](void* p) { munmap(p, size); });
        mapped_ = true;
        deviceData_ = deviceAddress_ ? deviceAddress : address;
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
    std::string name_;
    bool owner_ = false, created_ = false, deviceAddress_ = false;
    int deviceId_ = -1;
    size_t total_ = 0;
    std::shared_ptr<void> data_;
    void* deviceData_ = nullptr;
    size_t bytes_ = 0;
    bool mapped_ = false;
    std::vector<size_t> free_;
};
}  // namespace UC::Context
