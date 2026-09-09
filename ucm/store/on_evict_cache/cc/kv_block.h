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

#include <atomic>
#include <cerrno>
#include <cstdio>
#include <fcntl.h>
#include <memory>
#include <string>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>
#include "trans/buffer.h"

namespace UC::OnEvictCacheStore {

// A block has one shared-memory payload. Publication makes a complete block
// visible to the scheduler and other local workers; eviction-to-dump keeps it.
class KVBlock {
public:
    ~KVBlock()
    {
        if (registered_) { Trans::Buffer::UnregisterHostBuffer(data_); }
        if (data_) { munmap(data_, size_); }
        if (owned_) { shm_unlink(name_.c_str()); }
    }
    Status Create(const std::string& name, size_t size, size_t shards)
    {
        static std::atomic<size_t> sequence{0};
        name_ = name + ".pending." + std::to_string(getpid()) + "." + std::to_string(++sequence);
        size_ = size;
        shardsReady.assign(shards, false);
        int fd = shm_open(name_.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
        if (fd < 0) { return Status{errno, "create KV shared memory"}; }
        owned_ = true;
        auto allocationError = posix_fallocate(fd, 0, size);
        if (allocationError != 0) {
            auto status = Status{allocationError, "allocate KV shared memory"};
            close(fd);
            return status;
        }
        return Map(fd);
    }
    Status Open(const std::string& name, size_t size)
    {
        name_ = name;
        size_ = size;
        int fd = shm_open(name.c_str(), O_RDWR, 0600);
        if (fd < 0) {
            return errno == ENOENT ? Status::NotFound() : Status{errno, "open KV shared memory"};
        }
        return Map(fd);
    }
    static Expected<bool> Exists(const std::string& name)
    {
        int fd = shm_open(name.c_str(), O_RDONLY, 0600);
        if (fd < 0) {
            if (errno == ENOENT) { return false; }
            return Status{errno, "lookup KV shared memory"};
        }
        close(fd);
        return true;
    }
    Status Publish(const std::string& name)
    {
        // Both names refer to POSIX shared memory in tmpfs, never a disk backend.
        if (rename(("/dev/shm" + name_).c_str(), ("/dev/shm" + name).c_str()) != 0) {
            return Status{errno, "publish KV shared memory"};
        }
        name_ = name;
        published = true;
        return Status::OK();
    }
    void* Shard(size_t index, size_t shardSize)
    { return static_cast<std::byte*>(data_) + index * shardSize; }
    std::vector<bool> shardsReady;
    bool published{false};

private:
    Status Map(int fd)
    {
        auto* address = mmap(nullptr, size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        auto error = errno;
        close(fd);
        if (address == MAP_FAILED) { return Status{error, "map KV shared memory"}; }
        data_ = address;
        auto status = Trans::Buffer::RegisterHostBuffer(data_, size_);
        registered_ = status.Success();
        return status;
    }
    std::string name_;
    size_t size_{0};
    void* data_{nullptr};
    bool owned_{false};
    bool registered_{false};
};

}  // namespace UC::OnEvictCacheStore
