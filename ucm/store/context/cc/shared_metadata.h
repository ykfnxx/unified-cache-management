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
    Status Setup(const std::string& name, bool owner, size_t capacity = 0, uint64_t layout = 0);
    struct Location {
        size_t slot;
        bool memory;
    };
    Status Publish(const Key& key, uint8_t copies, size_t memory = 0, size_t ssd = 0);
    Expected<Location> Acquire(const Key& key, uint64_t layout);
    void Release(const Key& key);
    bool Evictable(const Key& key);
    bool ReserveEviction(const std::vector<Key>& keys);
    Expected<std::vector<uint8_t>> Lookup(const Key* keys, size_t count);
    void Deactivate();
    void Close();

private:
    struct Header;
    struct Entry;
    std::string name_;
    bool owner_ = false;
    void* mapping_ = nullptr;
    size_t bytes_ = 0;
    std::mutex mutex_;
    Header* header_ = nullptr;
    Entry* entries_ = nullptr;
    Status OpenWatcher();
    Entry* Find(const Key& key);
};
}  // namespace UC::Context
