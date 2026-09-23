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
#include "context_index.h"
#include <algorithm>
#include <cassert>
#include <tuple>

namespace UC::Context {
void ContextIndex::Unpublish(size_t id)
{
    const auto& s = segments_.at(id);
    if (!s.members.empty()) { cold_.erase({s.members.rbegin()->first, id}); }
}
void ContextIndex::Publish(size_t id)
{
    const auto& s = segments_.at(id);
    if (!s.members.empty()) { cold_.insert({s.members.rbegin()->first, id}); }
}
void ContextIndex::Add(const Key& id, std::optional<Key> parent)
{
    auto& node = nodes_[id];
    node.parent = parent;
    if (parent) {
        auto& p = nodes_.at(*parent);
        node.depth = p.depth + 1;
        if (p.children.empty()) {
            node.segment = p.segment;
            segments_.at(p.segment).path.push_back(id);
            p.children.insert(id);
            return;
        }
        if (p.children.size() == 1) {
            // Split the prefix ending at the new branch point from its old suffix.
            auto oldId = p.segment;
            Unpublish(oldId);
            auto& old = segments_.at(oldId);
            auto split = std::find(old.path.begin(), old.path.end(), *parent) + 1;
            auto prefixId = ++nextSegment_;
            auto& prefix = segments_[prefixId];
            prefix.path.assign(old.path.begin(), split);
            old.path.erase(old.path.begin(), split);
            for (const auto& key : prefix.path) {
                auto& member = nodes_.at(key);
                member.segment = prefixId;
                if (member.resident) {
                    old.members.erase({member.sequence, key});
                    prefix.members.insert({member.sequence, key});
                }
            }
            Publish(oldId);
            Publish(prefixId);
        }
        p.children.insert(id);
    }
    node.segment = ++nextSegment_;
    segments_[node.segment].path.push_back(id);
}
Status ContextIndex::Observe(const std::vector<Key>& path, uint64_t timestamp)
{
    std::optional<Key> parent;
    std::set<Key> seen;
    for (const auto& id : path) {
        auto found = nodes_.find(id);
        if (found != nodes_.end()) {
            // Existing nodes have one immutable parent in an acyclic tree;
            // a repeated node necessarily violates this parent relationship.
            if (found->second.parent != parent) {
                return Status::InvalidParam("block prefix changed");
            }
        } else if (!seen.insert(id).second) {
            return Status::InvalidParam("repeated block in prefix");
        }
        parent = id;
    }
    // Finish topology changes before collecting segment IDs: Add can split a segment.
    parent.reset();
    for (const auto& id : path) {
        if (!nodes_.count(id)) { Add(id, parent); }
        parent = id;
    }
    std::set<size_t> touched;
    for (const auto& id : path) {
        auto& n = nodes_.at(id);
        if (n.resident) {
            if (touched.insert(n.segment).second) { Unpublish(n.segment); }
            segments_.at(n.segment).members.erase({n.sequence, id});
        }
        n.sequence = ++sequence_;
        n.timestamp = timestamp;
        if (n.resident) { segments_.at(n.segment).members.insert({n.sequence, id}); }
    }
    for (auto segment : touched) { Publish(segment); }
    return Status::OK();
}
void ContextIndex::ChangeAncestors(const Key& id, bool insert)
{
    auto& delta = pendingAncestors_[id];
    delta += insert ? 1 : -1;
    if (!delta) { pendingAncestors_.erase(id); }
}
void ContextIndex::FlushAncestors()
{
    if (pendingAncestors_.empty()) { return; }
    // Coalesce mutations until subtree counts are actually needed by Select/Prune.
    // Each affected ancestor is visited once, including across shared prefixes.
    std::map<Key, int64_t> deltas;
    for (const auto& [id, delta] : pendingAncestors_) {
        for (std::optional<Key> key = id; key; key = nodes_.at(*key).parent) {
            if (!deltas.emplace(*key, 0).second) { break; }
        }
        deltas.at(id) += delta;
    }
    std::vector<std::pair<size_t, Key>> order;
    order.reserve(deltas.size());
    for (const auto& [id, delta] : deltas) { order.emplace_back(nodes_.at(id).depth, id); }
    std::sort(order.rbegin(), order.rend());
    for (const auto& [depth, id] : order) {
        auto& n = nodes_.at(id);
        auto delta = deltas.at(id);
        assert(int64_t(n.subtree) + delta >= 0);
        n.subtree = int64_t(n.subtree) + delta;
        if (n.parent) { deltas.at(*n.parent) += delta; }
    }
    pendingAncestors_.clear();
}
void ContextIndex::Insert(const Key& id)
{
    auto& n = nodes_.at(id);
    assert(!n.resident);
    Unpublish(n.segment);
    n.resident = true;
    segments_.at(n.segment).members.insert({n.sequence, id});
    ChangeAncestors(id, true);
    Publish(n.segment);
}
void ContextIndex::Remove(const Key& id)
{
    auto& n = nodes_.at(id);
    assert(n.resident);
    Unpublish(n.segment);
    segments_.at(n.segment).members.erase({n.sequence, id});
    n.resident = false;
    ChangeAncestors(id, false);
    Publish(n.segment);
}
void ContextIndex::Prune(const Key& id, const std::function<bool(const Key&)>& retained)
{
    std::optional<Key> key = id;
    while (key && nodes_.count(*key)) {
        const auto& current = nodes_.at(*key);
        if (current.resident || !current.children.empty() || retained(*key)) { break; }
        const auto n = current;
        FlushAncestors();
        auto& segment = segments_.at(n.segment);
        Unpublish(n.segment);
        assert(segment.path.back() == *key);
        segment.path.pop_back();
        if (segment.path.empty()) {
            segments_.erase(n.segment);
        } else {
            Publish(n.segment);
        }
        nodes_.erase(*key);
        if (n.parent) {
            auto& p = nodes_.at(*n.parent);
            p.children.erase(*key);
            if (p.children.size() == 1) {
                auto suffixId = nodes_.at(*p.children.begin()).segment;
                if (p.segment != suffixId) {
                    Unpublish(p.segment);
                    Unpublish(suffixId);
                    auto& prefix = segments_.at(p.segment);
                    auto& suffix = segments_.at(suffixId);
                    for (const auto& child : suffix.path) { nodes_.at(child).segment = p.segment; }
                    prefix.path.insert(prefix.path.end(), suffix.path.begin(), suffix.path.end());
                    prefix.members.merge(suffix.members);
                    segments_.erase(suffixId);
                    Publish(p.segment);
                }
            }
        }
        key = n.parent;
    }
}
ContextIndex::Victim ContextIndex::Select(double budget, size_t limit,
                                          const std::function<bool(const Key&)>& available)
{
    FlushAncestors();
    using Score = std::tuple<size_t, size_t, uint64_t, Key>;
    std::optional<Score> best;
    Victim result;
    size_t covered = 0;
    for (const auto& rank : cold_) {
        const auto& segment = segments_.at(rank.second);
        const auto& endpoint = segment.path.back();
        const auto& end = nodes_.at(endpoint);
        if (end.subtree != size_t(end.resident)) { continue; }
        std::vector<Key> blocks;
        for (auto it = segment.path.rbegin(); it != segment.path.rend(); ++it) {
            if (!nodes_.at(*it).resident) { continue; }
            if (!available(*it)) { break; }
            blocks.push_back(*it);
            if (blocks.size() == limit) { break; }
        }
        if (blocks.empty()) { continue; }
        Score score{end.depth, segment.members.size(), rank.first, endpoint};
        if (!best || score < *best) {
            best = score;
            std::reverse(blocks.begin(), blocks.end());
            result = {std::move(blocks), nodes_.at(segment.members.rbegin()->second).timestamp};
        }
        covered += segment.members.size();
        if (double(covered) >= budget) { break; }
    }
    return result;
}
}  // namespace UC::Context
