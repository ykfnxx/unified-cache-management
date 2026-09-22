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
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <unordered_map>
#include <vector>
#include "status/status.h"
#include "type/types.h"

namespace UC::Context {
using Key = Detail::BlockId;
// Owned and serialized by ContextStore. Segments are maximal unbranched paths.
class ContextIndex {
public:
    struct Node {
        std::optional<Key> parent;
        std::set<Key> children;
        size_t depth = 1, segment = 0, subtree = 0;
        uint64_t sequence = 0, timestamp = 0;
        bool resident = false;
    };
    struct Victim {
        std::vector<Key> blocks;
        uint64_t lastAccess = 0;
    };
    Status Observe(const std::vector<Key>& path, uint64_t timestamp);
    bool Contains(const Key& id) const { return nodes_.count(id); }
    void Insert(const Key& id);
    void Remove(const Key& id);
    void Prune(const Key& id, const std::function<bool(const Key&)>& retained);
    Victim Select(double budget, size_t limit, const std::function<bool(const Key&)>& available);
    size_t Size() const { return nodes_.size(); }

private:
    using Recency = std::pair<uint64_t, Key>;
    struct Segment {
        std::vector<Key> path;
        std::set<Recency> members;
    };
    std::unordered_map<Key, Node, Detail::BlockIdHasher> nodes_;
    std::map<size_t, Segment> segments_;
    std::set<std::pair<uint64_t, size_t>> cold_;
    size_t nextSegment_ = 0;
    uint64_t sequence_ = 0;
    void Unpublish(size_t segment);
    void Publish(size_t segment);
    void Add(const Key& id, std::optional<Key> parent);
    std::map<Key, int64_t> pendingAncestors_;
    void ChangeAncestors(const Key& id, bool insert);
    void FlushAncestors();
};
}  // namespace UC::Context
