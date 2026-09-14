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
#include <chrono>
#include <cstdio>
#include <fcntl.h>
#include <memory>
#include <string>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>
#include "trans/buffer.h"

namespace UC::OnEvictCacheStore {

struct KVBlockCreateTimings {
    std::chrono::steady_clock::duration open{};
    std::chrono::steady_clock::duration allocate{};
    std::chrono::steady_clock::duration map{};
    std::chrono::steady_clock::duration registerHost{};
};

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
    Status Create(const std::string& name, size_t size, size_t shards,
                  KVBlockCreateTimings* timings = nullptr, bool registerHost = true)
    {
        const auto started = std::chrono::steady_clock::now();
        static std::atomic<size_t> sequence{0};
        name_ = name + ".pending." + std::to_string(getpid()) + "." + std::to_string(++sequence);
        size_ = size;
        shardsReady.assign(shards, false);
        int fd = shm_open(name_.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
        const auto opened = std::chrono::steady_clock::now();
        if (timings) { timings->open += opened - started; }
        if (fd < 0) { return Status{errno, "create KV shared memory"}; }
        owned_ = true;
        auto allocationError = posix_fallocate(fd, 0, size);
        const auto allocated = std::chrono::steady_clock::now();
        if (timings) { timings->allocate += allocated - opened; }
        if (allocationError != 0) {
            auto status = Status{allocationError, "allocate KV shared memory"};
            close(fd);
            return status;
        }
        return Map(fd, timings, registerHost);
    }
    Status Open(const std::string& name, size_t size, KVBlockCreateTimings* timings = nullptr,
                bool registerHost = true)
    {
        const auto started = std::chrono::steady_clock::now();
        name_ = name;
        size_ = size;
        int fd = shm_open(name.c_str(), O_RDWR, 0600);
        if (timings) { timings->open += std::chrono::steady_clock::now() - started; }
        if (fd < 0) {
            return errno == ENOENT ? Status::NotFound() : Status{errno, "open KV shared memory"};
        }
        return Map(fd, timings, registerHost);
    }
    static Expected<bool> Exists(const std::string& name)
    {
        const auto path = "/dev/shm" + name;
        if (access(path.c_str(), F_OK) == 0) { return true; }
        return errno == ENOENT ? Expected<bool>{false}
                               : Expected<bool>{Status{errno, "lookup KV shared memory"}};
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
    Status RegisterHost(KVBlockCreateTimings* timings = nullptr)
    {
        if (registered_) { return Status::OK(); }
        const auto started = std::chrono::steady_clock::now();
        auto status = Trans::Buffer::RegisterHostBuffer(data_, size_);
        if (timings) {
            timings->registerHost += std::chrono::steady_clock::now() - started;
        }
        registered_ = status.Success();
        return status;
    }
    void* Shard(size_t index, size_t shardSize)
    { return static_cast<std::byte*>(data_) + index * shardSize; }
    std::vector<bool> shardsReady;
    bool published{false};

private:
    Status Map(int fd, KVBlockCreateTimings* timings = nullptr, bool registerHost = true)
    {
        const auto started = std::chrono::steady_clock::now();
        auto* address = mmap(nullptr, size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        auto error = errno;
        close(fd);
        const auto mapped = std::chrono::steady_clock::now();
        if (timings) { timings->map += mapped - started; }
        if (address == MAP_FAILED) { return Status{error, "map KV shared memory"}; }
        data_ = address;
        return registerHost ? RegisterHost(timings) : Status::OK();
    }
    std::string name_;
    size_t size_{0};
    void* data_{nullptr};
    bool owned_{false};
    bool registered_{false};
};

}  // namespace UC::OnEvictCacheStore
