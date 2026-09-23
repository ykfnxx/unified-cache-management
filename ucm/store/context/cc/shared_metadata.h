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
#include <atomic>
#include <mutex>
#include <string>
#include <vector>
#include "context_index.h"

namespace UC::Context {
// Shared availability and MLA slot leases. No process-local pointers are stored.
class SharedMetadata {
public:
    ~SharedMetadata();
    Status Setup(const std::string& name, bool owner, size_t capacity = 0, uint64_t layout = 0,
                 size_t ranks = 1, size_t batches = 0);
    struct Location {
        size_t slot;
    };
    Status Publish(const Key& key, uint8_t copies, size_t memory = 0);
    Expected<Location> Acquire(const Key& key, uint64_t layout);
    void Release(const Key& key);
    void Release(const Key* keys, size_t count);
    // Batch membership follows FIFO order for identical descriptors on each TP rank.
    Expected<uint64_t> BeginLoad(const Key& signature, size_t rank, uint64_t timeoutMs);
    Expected<Location> WaitAcquire(const Key& key, uint64_t layout, uint64_t batch);
    // Wait only for the first key, then acquire the contiguous ready prefix.
    Expected<size_t> WaitAcquire(const Key* keys, size_t count, Location* locations,
                                 uint64_t layout, uint64_t batch);
    void FailLoad(uint64_t batch, const Status& status);
    // With wait=false, Retry means readers are pending before the batch deadline.
    Status WaitReaders(uint64_t batch, bool wait = true);
    void EndLoad(uint64_t batch, size_t rank);

    bool Evictable(const Key& key);
    bool ReserveEviction(const std::vector<Key>& keys);
    Expected<std::vector<uint8_t>> Lookup(const Key* keys, size_t count);
    void Deactivate();
    void Close();

private:
    struct Header;
    struct Entry;
    struct Batch;
    std::string name_;
    bool owner_ = false;
    void* mapping_ = nullptr;
    size_t bytes_ = 0;
    std::mutex mutex_;
    Header* header_ = nullptr;
    Entry* entries_ = nullptr;
    Batch* batches_ = nullptr;
    Status OpenWatcher();
    Entry* Find(const Key& key);
    Batch* FindBatch(uint64_t id);
    Status WaitChange(const Batch& batch);
};
}  // namespace UC::Context
