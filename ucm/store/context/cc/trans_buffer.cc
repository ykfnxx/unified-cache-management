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
// Cache-style shard buffer; radix policy controls which blocks may return slots.
// Transfer handles hold only shard references and never access the tree.
#include "trans_buffer.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <mutex>
#include <new>
#include <numeric>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include "context_index.h"
#include "logger/logger.h"
#include "metrics_api.h"
#include "trans/buffer.h"
#include "trans/device.h"

namespace UC::Context {
namespace {
constexpr size_t nil = std::numeric_limits<size_t>::max();
constexpr uint64_t magic = 0x55434d4354580007ULL;
constexpr size_t buckets = 16411;
constexpr size_t evicting = size_t(1) << 63;
struct Mutex {
    pthread_mutex_t value;
    void Init()
    {
        pthread_mutexattr_t a;
        pthread_mutexattr_init(&a);
        pthread_mutexattr_setpshared(&a, PTHREAD_PROCESS_SHARED);
        pthread_mutexattr_settype(&a, PTHREAD_MUTEX_ADAPTIVE_NP);
        pthread_mutex_init(&value, &a);
        pthread_mutexattr_destroy(&a);
    }
    void lock() { pthread_mutex_lock(&value); }
    void unlock() { pthread_mutex_unlock(&value); }
};
// Same short per-node critical sections as CacheStore's shared node lock.
struct SpinLock {
    pthread_spinlock_t value;
    void Init() { pthread_spin_init(&value, PTHREAD_PROCESS_SHARED); }
    void lock() { pthread_spin_lock(&value); }
    void unlock() { pthread_spin_unlock(&value); }
};
size_t Align(size_t n, size_t a) { return (n + a - 1) / a * a; }
size_t Hash(const Key& key) { return Detail::BlockIdHasher{}(key) % buckets; }
size_t ShardHash(const Key& key, size_t layer)
{
    const auto h = Detail::BlockIdHasher{}(key);
    return (h ^ (layer + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2))) % buckets;
}
}  // namespace
struct TransBuffer::Impl {
    struct alignas(64) Block {
        Key key;
        size_t next = nil, pending = nil, firstShard = nil;
        std::atomic<size_t> ready{0}, failed{0};
        std::atomic<size_t> activeShards{0};
        std::atomic<bool> loaded{false};
    };
    struct alignas(64) Shard {
        Key key;
        SpinLock mutex;
        size_t references = 0;
        size_t block = nil, layer = 0, next = nil, nextBlock = nil;
        alignas(64) std::atomic<State> state{State::LOADING};
        std::atomic<int32_t> error{0};
    };
    struct Header {
        uint64_t version, layout;
        size_t count, layers, shardBytes, metadataBytes, totalBytes;
        Mutex allocation, metadata, recycled;
        size_t freeHead = nil;
        alignas(64) std::atomic<size_t> cursor{0};
        alignas(64) std::atomic<size_t> freeCount{0};
        pthread_cond_t changed;
        size_t freeBlock, blockCount = 0, pendingHead = nil;
        size_t pressure = 0, completed = 0;
        int32_t allocationError = 0;
        size_t allocationEpoch = 0;
        std::atomic<size_t> released{0};
        std::atomic<bool> ownerAlive{false}, pressureActive{false};
        std::atomic<size_t> tasks{0}, highWater{0}, failures{0};

        size_t heads[buckets], shardHeads[buckets];
        Mutex locks[buckets], shardLocks[buckets];
    };
    Config cfg;
    std::atomic<uint64_t> h2d{0}, d2h{0}, backendBlocks{0}, backendShards{0};
    std::string name;
    void* mapping = nullptr;
    size_t mappedBytes = 0;
    Header* header = nullptr;
    Block* blocks = nullptr;
    Shard* shards = nullptr;
    char* data = nullptr;
    char* deviceData = nullptr;
    bool registered = false, policyOwner = false, stop = false;
    std::thread maintenance;
    std::mutex policy;
    ContextIndex index;
    struct Observation {
        uint64_t sequence;
        std::vector<Key> path;
    };
    std::map<std::string, Observation> observed;
    std::unordered_map<Key, size_t, Detail::BlockIdHasher> retained;
    std::unordered_map<Key, size_t, Detail::BlockIdHasher> resident;
    std::vector<Key> retired, unobserved;
    uint64_t now = 0;
    std::map<std::string, uint64_t> stats;

    ~Impl()
    {
        if (maintenance.joinable()) {
            {
                std::lock_guard<Mutex> l(header->allocation);
                stop = true;
                pthread_cond_broadcast(&header->changed);
            }
            maintenance.join();
        }
        if (policyOwner && header) {
            header->ownerAlive.store(false);
            pthread_cond_broadcast(&header->changed);
            shm_unlink(name.c_str());
        }
        if (registered) { Trans::Buffer::UnregisterHostBuffer(data); }
        if (mapping) { munmap(mapping, mappedBytes); }
    }
    size_t BlockOffset() const { return Align(sizeof(Header), alignof(Block)); }
    size_t ShardOffset(size_t n) const
    { return Align(BlockOffset() + sizeof(Block) * n, alignof(Shard)); }
    void Attach()
    {
        blocks = reinterpret_cast<Block*>(static_cast<char*>(mapping) + BlockOffset());
        shards = reinterpret_cast<Shard*>(static_cast<char*>(mapping) + ShardOffset(header->count));
    }
    Status Open(bool watcher)
    {
        int fd = shm_open(name.c_str(), watcher ? O_RDWR : O_RDWR | O_CREAT, 0600);
        if (fd < 0) {
            return watcher && errno == ENOENT ? Status::NotFound()
                                              : Status::Error("open context buffer");
        }
        flock(fd, LOCK_EX);
        struct stat st{};
        fstat(fd, &st);
        if (watcher && st.st_size == 0) {
            close(fd);
            return Status::NotFound();
        }
        size_t count = watcher ? 0 : cfg.bufferCapacity / cfg.shardSize;
        size_t layers = watcher ? 0 : cfg.blockSize / cfg.shardSize;
        size_t meta =
            watcher ? 0 : Align(ShardOffset(count) + sizeof(Shard) * count, sysconf(_SC_PAGESIZE));
        size_t total = meta + count * cfg.shardSize;
        uint64_t layout = 1469598103934665603ULL;
        if (!watcher) {
            for (size_t x : {count, cfg.blockSize, cfg.shardSize}) {
                layout = (layout ^ x) * 1099511628211ULL;
            }
            for (auto x : cfg.tensorSizes) { layout = (layout ^ x) * 1099511628211ULL; }
        }
        bool fresh = st.st_size == 0;
        if (fresh && posix_fallocate(fd, 0, total) != 0) {
            close(fd);
            return Status::NoSpace();
        }
        Header* probe = static_cast<Header*>(
            mmap(nullptr, sizeof(Header), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
        if (probe == MAP_FAILED) {
            close(fd);
            return Status::OutOfMemory();
        }
        if (!fresh && (probe->version != magic || (!watcher && probe->layout != layout))) {
            munmap(probe, sizeof(Header));
            close(fd);
            return Status::InvalidParam("context shared layout mismatch");
        }
        if (watcher) {
            meta = probe->metadataBytes;
            total = probe->totalBytes;
        }
        munmap(probe, sizeof(Header));
        mappedBytes = watcher ? meta : total;
        mapping =
            mmap(nullptr, mappedBytes, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd, 0);
        if (mapping == MAP_FAILED) {
            mapping = nullptr;
            close(fd);
            return Status::OutOfMemory();
        }
        header = static_cast<Header*>(mapping);
        if (fresh) {
            new (header) Header{};
            header->count = count;
            header->layers = layers;
            header->shardBytes = cfg.shardSize;
            header->metadataBytes = meta;
            header->totalBytes = total;
            header->layout = layout;
            header->allocation.Init();
            header->metadata.Init();
            header->recycled.Init();
            pthread_condattr_t a;
            pthread_condattr_init(&a);
            pthread_condattr_setpshared(&a, PTHREAD_PROCESS_SHARED);
            pthread_condattr_setclock(&a, CLOCK_MONOTONIC);
            pthread_cond_init(&header->changed, &a);
            pthread_condattr_destroy(&a);
            header->freeBlock = 0;
            header->freeCount = count;
            for (size_t i = 0; i < buckets; ++i) {
                header->heads[i] = header->shardHeads[i] = nil;
                header->locks[i].Init();
                header->shardLocks[i].Init();
            }
            Attach();
            for (size_t i = 0; i < count; ++i) {
                new (&blocks[i]) Block{};
                blocks[i].next = i + 1 == count ? nil : i + 1;
            }
            for (size_t i = 0; i < count; ++i) {
                new (&shards[i]) Shard{};
                shards[i].mutex.Init();
            }
            header->version = magic;
        } else {
            Attach();
        }
        flock(fd, LOCK_UN);
        close(fd);
        if (watcher) { return Status::OK(); }
        data = static_cast<char*>(mapping) + meta;
        Trans::Device device;
        auto s = device.Setup(cfg.deviceId);
        if (s.Failure()) { return s; }
        void* mapped = nullptr;
        s = Trans::Buffer::RegisterHostBuffer(data, total - meta, &mapped);
        if (s.Failure()) { return s; }
        registered = true;
        deviceData = static_cast<char*>(mapped);
        if (policyOwner) {
            header->ownerAlive.store(true);
            maintenance = std::thread([this] { Maintain(); });
        }
        return Status::OK();
    }
    size_t Find(const Key& key)
    {
        for (auto i = header->heads[Hash(key)]; i != nil; i = blocks[i].next) {
            if (blocks[i].key == key) { return i; }
        }
        return nil;
    }
    size_t FetchNode()
    {
        // Cache's cursor handles virgin slots. Its clock victim selection is
        // replaced by slots already released by radix/on-evict write, not a
        // scan over live shards that the policy has not permitted us to reuse.
        auto pos = header->cursor.load(std::memory_order_relaxed);
        while (pos < header->count) {
            if (header->cursor.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
                header->freeCount.fetch_sub(1, std::memory_order_relaxed);
                return pos;
            }
        }
        std::lock_guard<Mutex> pool(header->recycled);
        pos = header->freeHead;
        if (pos != nil) {
            header->freeHead = shards[pos].next;
            header->freeCount.fetch_sub(1, std::memory_order_relaxed);
        }
        return pos;
    }
    void Drain()
    {  // Only new logical blocks enter this policy admission queue.
        size_t i;
        {
            std::lock_guard<Mutex> l(header->metadata);
            i = header->pendingHead;
            header->pendingHead = nil;
        }
        // Only the policy owner retires slots, so these detached entries remain
        // stable while we register them without holding the metadata lock.
        while (i != nil) {
            auto next = blocks[i].pending;
            resident.emplace(blocks[i].key, i);
            unobserved.push_back(blocks[i].key);
            i = next;
        }
        size_t remaining = 0;
        for (const auto& key : unobserved) {
            if (index.Contains(key)) {
                index.Insert(key);
            } else {
                unobserved[remaining++] = key;
            }
        }
        unobserved.resize(remaining);
    }

    void Prune()
    {
        if (header->tasks.load() != 0) { return; }
        for (auto it = retired.rbegin(); it != retired.rend(); ++it) {
            index.Prune(*it,
                        [this](const Key& k) { return retained.count(k) || resident.count(k); });
        }
        retired.clear();
    }
    // Count occupied shard nodes, not Handle copies. Only 0->1 and 1->0
    // transitions touch this policy state; victim selection stays O(1).
    bool Available(size_t pos) { return blocks[pos].activeShards.load() == 0; }
    bool Reserve(size_t pos)
    {
        auto& block = blocks[pos];
        std::lock_guard<Mutex> bucket(header->locks[Hash(block.key)]);
        size_t expected = 0;
        return block.activeShards.compare_exchange_strong(expected, evicting);
    }
    void Unreserve(size_t pos) { blocks[pos].activeShards.store(0); }
    void Remove(size_t pos)
    {
        auto& block = blocks[pos];
        const auto key = block.key;
        const auto failed = block.failed.load();
        const auto first = block.firstShard;
        size_t tail = nil, count = 0;
        {
            std::lock_guard<Mutex> b(header->locks[Hash(key)]);
            for (auto i = block.firstShard; i != nil;) {
                auto& shard = shards[i];
                const auto next = shard.nextBlock;
                std::lock_guard<Mutex> bucket(header->shardLocks[ShardHash(key, shard.layer)]);
                auto* link = &header->shardHeads[ShardHash(key, shard.layer)];
                while (*link != i) { link = &shards[*link].next; }
                *link = shard.next;
                // The block gate excludes every handle and new shard. Once
                // unlinked under its bucket, this node needs no further lock.
                shard.block = nil;
                shard.next = next;
                tail = i;
                ++count;
                i = next;
            }
            auto* link = &header->heads[Hash(key)];
            while (*link != pos) { link = &blocks[*link].next; }
            *link = block.next;
            std::lock_guard<Mutex> m(header->metadata);
            block.next = header->freeBlock;
            header->freeBlock = pos;
            --header->blockCount;
        }
        {
            std::lock_guard<Mutex> pool(header->recycled);
            shards[tail].next = header->freeHead;
            header->freeHead = first;
            header->freeCount.fetch_add(count, std::memory_order_relaxed);
        }
        {
            std::lock_guard<Mutex> a(header->allocation);
            pthread_cond_broadcast(&header->changed);
        }
        if (index.Contains(key)) {
            index.Remove(key);
        } else {
            unobserved.erase(std::remove(unobserved.begin(), unobserved.end(), key),
                             unobserved.end());
        }
        header->failures.fetch_sub(failed);
        resident.erase(key);
        retired.push_back(key);
    }
    Status Evict(std::unique_lock<std::mutex>& policyLock)
    {
        // Failed empty fills are allocations to roll back, not policy evictions.
        if (header->failures.load())
            for (const auto& item : resident) {
                auto& block = blocks[item.second];
                if (block.ready.load() == 0 && block.failed.load() && Reserve(item.second)) {
                    Remove(item.second);
                    return Status::OK();
                }
            }
        auto begin = std::chrono::steady_clock::now();
        bool busy = false;
        auto victim = index.Select(cfg.alpha * (header->count / header->layers), cfg.evictionLimit,
                                   [this, &busy](const Key& key) {
                                       auto pos = resident.at(key);
                                       if (!Available(pos)) {
                                           busy = true;
                                           return false;
                                       }
                                       return blocks[pos].ready.load() == header->layers ||
                                              blocks[pos].loaded.load();
                                   });
        stats["decision_ns"] += std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    std::chrono::steady_clock::now() - begin)
                                    .count();
        ++stats["eviction_decisions"];
        if (victim.blocks.empty()) {
            if (busy) { return Status::Retry(); }
            ++stats["no_space"];
            return Status::NoSpace();
        }
        bool drop = cfg.retention >= 0 && now - victim.lastAccess > uint64_t(cfg.retention);
        size_t freed = 0;
        for (const auto& key : victim.blocks) {
            auto pos = resident.at(key);
            auto& block = blocks[pos];
            if (!Reserve(pos)) { continue; }
            auto s = Status::OK();
            if (!drop) {
                const auto backendKey = cfg.BackendKey(key);
                // Reserved shard nodes keep payload stable. Backend latency must
                // not hold the radix mutex across the next request's Observe.
                policyLock.unlock();
                bool written = false;
                auto found = cfg.storeBackend->Lookup(&backendKey, 1);
                if (!found) {
                    s = found.Error();
                } else if (!found.Value()[0]) {
                    if (block.ready.load() != header->layers) {
                        s = Status::NotFound();
                    } else {
                        Detail::TaskDesc task;
                        for (auto i = block.firstShard; i != nil; i = shards[i].nextBlock) {
                            task.push_back(
                                {backendKey, shards[i].layer, {data + i * header->shardBytes}});
                        }
                        std::sort(task.begin(), task.end(),
                                  [](const auto& a, const auto& b) { return a.index < b.index; });
                        auto t = cfg.storeBackend->Dump(std::move(task));
                        s = t ? cfg.storeBackend->Wait(t.Value()) : t.Error();
                        written = s.Success();
                    }
                }
                policyLock.lock();
                if (s.Success()) {
                    ++stats[written ? "backend_dump_blocks" : "backend_dump_skipped_blocks"];
                    if (written) { stats["backend_dump_bytes"] += cfg.blockSize; }
                }
            }
            if (s.Failure()) {
                Unreserve(pos);
                return s;
            }
            Remove(pos);
            ++freed;
            ++stats["evicted_blocks"];
            ++stats[drop ? "drop_blocks" : "dump_blocks"];
            Metrics::UpdateStats(NAME_TO_METRIC_ID("context_evict_blocks_total"), 1);
            Metrics::UpdateStats(drop ? NAME_TO_METRIC_ID("context_drop_blocks_total")
                                      : NAME_TO_METRIC_ID("context_dump_blocks_total"),
                                 1);
        }
        return freed ? Status::OK() : Status::Retry();
    }
    void Maintain()
    {
        for (;;) {
            size_t request, epoch;
            {
                std::unique_lock<Mutex> a(header->allocation);
                while (!stop && header->pressure == header->completed) {
                    pthread_cond_wait(&header->changed, &header->allocation.value);
                }
                if (stop) { return; }
                request = header->pressure;
                epoch = header->released.load();
            }
            // Reclaim a small reserve per pressure cycle; each Remove already
            // wakes allocators, so they need not wait for the whole batch.
            for (;;) {
                auto s = Status::OK();
                {
                    std::unique_lock<std::mutex> p(policy);
                    Drain();
                    s = Evict(p);
                    Prune();
                }
                std::lock_guard<Mutex> a(header->allocation);
                header->allocationError = s.Underlying();
                header->allocationEpoch = epoch;
                if (s.Failure() || header->freeCount >= header->count / 20) {
                    if (s.Success()) { header->pressureActive.store(false); }
                    header->completed = request;
                    pthread_cond_broadcast(&header->changed);
                    break;
                }
                epoch = header->released.load();
            }
        }
    }
};

TransBuffer::TransBuffer() : impl_(std::make_unique<Impl>()) {}
TransBuffer::~TransBuffer() = default;
Status TransBuffer::Setup(const Config& c)
{
    auto& p = *impl_;
    p.cfg = c;
    p.policyOwner = c.deviceId >= 0 && (!c.shareBufferEnable || c.tpRank == 0);
    p.name = "/ucm_context_v7_" + c.uniqueId +
             (c.shareBufferEnable ? "_mla" : "_tp" + std::to_string(c.tpRank));
    if (c.deviceId < 0) { return Status::OK(); }
    return p.Open(false);
}
Expected<TransBuffer::Handle> TransBuffer::Get(const Key& key, size_t layer)
{
    auto& p = *impl_;
    const auto bucket = ShardHash(key, layer);
    if (!p.header->ownerAlive.load()) { return Status::NotFound(); }
    {
        std::lock_guard<Mutex> b(p.header->shardLocks[bucket]);
        bool owner = false;
        const auto pos = FindAt(bucket, key, layer, owner);
        if (pos != nil) { return Handle(this, pos, owner); }
    }
    return Alloc(key, layer, false);
}
void TransBuffer::Prealloc(const Key& key, size_t layer)
{
    auto& p = *impl_;
    const auto bucket = ShardHash(key, layer);
    {
        std::lock_guard<Mutex> b(p.header->shardLocks[bucket]);
        if (ExistAt(bucket, key, layer)) { return; }
    }
    // Cache Prealloc reserves a slot, not fill ownership. Radix writeback is
    // never awaited by this speculative call; the next real Get handles pressure.
    auto result = Alloc(key, layer, true);
}
bool TransBuffer::ExistAt(size_t bucket, const Key& key, size_t layer)
{
    auto& p = *impl_;
    for (auto i = p.header->shardHeads[bucket]; i != nil; i = p.shards[i].next) {
        if (p.shards[i].key == key && p.shards[i].layer == layer) { return true; }
    }
    return false;
}
size_t TransBuffer::FindAt(size_t bucket, const Key& key, size_t layer, bool& owner)
{
    auto& p = *impl_;
    for (auto i = p.header->shardHeads[bucket]; i != nil; i = p.shards[i].next) {
        auto& node = p.shards[i];
        if (node.key != key || node.layer != layer) { continue; }
        std::lock_guard<SpinLock> n(node.mutex);
        owner = node.references == 0;
        if (owner) {
            auto& active = p.blocks[node.block].activeShards;
            auto count = active.load();
            do {
                if (count == evicting) { return nil; }
            } while (!active.compare_exchange_weak(count, count + 1));
        }
        if (owner && node.state.load(std::memory_order_relaxed) == State::FAILED) {
            node.state.store(State::LOADING, std::memory_order_relaxed);
            node.error.store(Status::OK().Underlying(), std::memory_order_relaxed);
            p.blocks[node.block].failed.fetch_sub(1);
            p.header->failures.fetch_sub(1);
        }
        ++node.references;
        return i;
    }
    return nil;
}
Expected<TransBuffer::Handle> TransBuffer::Alloc(const Key& key, size_t layer, bool prealloc)
{
    auto& p = *impl_;
    auto* h = p.header;
    const auto bucket = ShardHash(key, layer);
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(p.cfg.timeoutMs);
    for (;;) {
        bool evictingBlock = false;
        {
            std::lock_guard<Mutex> blockLock(h->locks[Hash(key)]);
            std::lock_guard<Mutex> shardLock(h->shardLocks[bucket]);
            if (prealloc) {
                if (ExistAt(bucket, key, layer)) { return Handle{}; }
            } else {
                bool owner = false;
                const auto pos = FindAt(bucket, key, layer, owner);
                if (pos != nil) { return Handle(this, pos, owner); }
            }
            auto blockPos = p.Find(key);
            evictingBlock = blockPos != nil && p.blocks[blockPos].activeShards.load() == evicting;
            // Preallocation only extends a resident block. A preceding layer
            // already created its metadata; no extra policy admission is needed.
            if (prealloc && blockPos == nil) { return Status::NotFound(); }
            if (!evictingBlock) {
                const auto pos = p.FetchNode();
                if (pos != nil) {
                    auto& node = p.shards[pos];
                    std::lock_guard<SpinLock> n(node.mutex);
                    if (blockPos == nil) {
                        std::lock_guard<Mutex> m(h->metadata);
                        blockPos = h->freeBlock;
                        auto& block = p.blocks[blockPos];
                        h->freeBlock = block.next;
                        ++h->blockCount;
                        block.key = key;
                        block.activeShards.store(0);
                        block.ready.store(0);
                        block.failed.store(0);
                        block.loaded.store(false);
                        block.firstShard = nil;
                        block.next = h->heads[Hash(key)];
                        h->heads[Hash(key)] = blockPos;
                        block.pending = h->pendingHead;
                        h->pendingHead = blockPos;
                        h->highWater.store(std::max(h->highWater.load(), h->blockCount));
                    }
                    auto& block = p.blocks[blockPos];
                    node.key = key;
                    node.block = blockPos;
                    node.layer = layer;
                    node.references = prealloc ? 0 : 1;
                    if (!prealloc) { block.activeShards.fetch_add(1); }
                    node.state.store(State::LOADING, std::memory_order_relaxed);
                    node.error.store(0, std::memory_order_relaxed);
                    node.next = h->shardHeads[bucket];
                    h->shardHeads[bucket] = pos;
                    node.nextBlock = block.firstShard;
                    block.firstShard = pos;
                    return prealloc ? Handle{} : Handle(this, pos, true);
                }
            }
        }
        if (prealloc) { return Status::NoSpace(); }
        // Only radix eviction/writeback crosses to the policy owner. Normal
        // shard allocation and reference release do not take this mutex.
        std::unique_lock<Mutex> a(h->allocation);
        timespec until;
        const auto ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(deadline.time_since_epoch())
                .count();
        until.tv_sec = ns / 1000000000;
        until.tv_nsec = ns % 1000000000;
        auto wait = [&] {
            return pthread_cond_timedwait(&h->changed, &h->allocation.value, &until);
        };
        if (evictingBlock) {
            // A writeback may have completed before we took the wait mutex.
            std::lock_guard<Mutex> b(h->locks[Hash(key)]);
            const auto pos = p.Find(key);
            evictingBlock = pos != nil && (p.blocks[pos].activeShards.load() == evicting);
        }
        if (evictingBlock) {
            if (wait() == ETIMEDOUT) { return Status::Timeout(); }
            continue;
        }
        if (h->freeCount.load()) { continue; }
        h->pressureActive.store(true);
        const auto request = h->pressure == h->completed ? ++h->pressure : h->pressure;
        pthread_cond_broadcast(&h->changed);
        while (h->completed < request && !h->freeCount.load()) {
            if (wait() == ETIMEDOUT) { return Status::Timeout(); }
        }
        // Another rank may have admitted exactly this shard while we waited.
        // Claim it before interpreting a subsequent policy allocation failure.
        {
            std::lock_guard<Mutex> b(h->shardLocks[bucket]);
            bool owner = false;
            const auto pos = FindAt(bucket, key, layer, owner);
            if (pos != nil) { return Handle(this, pos, owner); }
        }
        if (!h->freeCount.load() && h->completed >= request && h->allocationError) {
            if (h->allocationError != Status::Retry().Underlying()) {
                return Status{h->allocationError, "context allocation failed"};
            }
            while (!h->freeCount.load() && h->released.load() == h->allocationEpoch) {
                if (wait() == ETIMEDOUT) { return Status::Timeout(); }
            }
        }
    }
}
Detail::BlockId TransBuffer::BackendKey(const Key& key) const { return impl_->cfg.BackendKey(key); }
Expected<std::vector<uint8_t>> TransBuffer::Lookup(const Key* keys, size_t n)
{
    std::vector<uint8_t> result(n);
    std::vector<Key> missing;
    std::vector<size_t> offsets;
    for (size_t i = 0; i < n; ++i) {
        result[i] = Exist(keys[i]);
        if (!result[i]) {
            missing.push_back(BackendKey(keys[i]));
            offsets.push_back(i);
        }
    }
    if (!missing.empty()) {
        auto found = impl_->cfg.storeBackend->Lookup(missing.data(), missing.size());
        if (!found) { return found.Error(); }
        for (size_t i = 0; i < missing.size(); ++i) { result[offsets[i]] = found.Value()[i]; }
    }
    return result;
}
bool TransBuffer::Exist(const Key& key)
{
    auto& p = *impl_;
    if (!p.header && p.Open(true).Failure()) { return false; }
    if (!p.header->ownerAlive.load()) { return false; }
    std::lock_guard<Mutex> b(p.header->locks[Hash(key)]);
    auto pos = p.Find(key);
    return pos != nil && !(p.blocks[pos].activeShards.load() == evicting) &&
           p.blocks[pos].ready.load() == p.header->layers;
}
Status TransBuffer::Observe(const std::string& request, uint64_t observation, uint64_t time,
                            const std::vector<Key>& path)
{
    auto& p = *impl_;
    if (!p.policyOwner) { return Status::OK(); }
    auto started = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> l(p.policy);
    auto previous = p.observed.find(request);
    if (!path.empty() && previous != p.observed.end() && previous->second.sequence >= observation) {
        return Status::OK();
    }
    if (!path.empty() && time < p.now) {
        return Status::InvalidParam("context timestamps must be nondecreasing");
    }
    if (!path.empty()) {
        auto s = p.index.Observe(path, time);
        if (s.Failure()) { return s; }
        p.now = time;
    }
    bool same = previous != p.observed.end() && previous->second.path == path;
    if (same) {
        previous->second.sequence = observation;
    } else {
        for (const auto& key : path) { ++p.retained[key]; }
        if (previous != p.observed.end()) {
            for (const auto& key : previous->second.path) {
                if (--p.retained.at(key) == 0) { p.retained.erase(key); }
                p.retired.push_back(key);
            }
            p.observed.erase(previous);
        }
        if (!path.empty()) { p.observed.emplace(request, Impl::Observation{observation, path}); }
    }
    p.Drain();
    p.Prune();
    Metrics::UpdateStats(
        NAME_TO_METRIC_ID("context_observe_duration_ms"),
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started)
            .count());
    return Status::OK();
}
Status TransBuffer::BeginTask(const Detail::TaskDesc& task)
{
    auto& p = *impl_;
    if (task.empty()) { return Status::InvalidParam("empty transfer"); }
    for (const auto& s : task) {
        if (s.index >= p.header->layers || s.addrs.size() != p.cfg.tensorSizes.size() ||
            std::any_of(s.addrs.begin(), s.addrs.end(), [](void* x) { return !x; })) {
            return Status::InvalidParam("invalid transfer shard");
        }
    }
    if (task.size() > p.header->count) {
        std::set<std::pair<Key, size_t>> keys;
        for (const auto& s : task) { keys.emplace(s.owner, s.index); }
        if (keys.size() > p.header->count) { return Status::NoSpace(); }
    }
    p.header->tasks.fetch_add(1);
    return Status::OK();
}
void TransBuffer::EndTask() { impl_->header->tasks.fetch_sub(1); }
std::map<std::string, uint64_t> TransBuffer::Stats()
{
    auto& p = *impl_;
    std::lock_guard<std::mutex> l(p.policy);
    if (p.policyOwner) {
        p.Drain();
        p.Prune();
    }
    auto result = p.stats;
    if (!p.header) { return result; }
    std::lock_guard<Mutex> a(p.header->metadata);
    result["memory_blocks"] = p.header->blockCount;
    result["memory_shards"] = p.header->count - p.header->freeCount;
    result["memory_bytes"] = result["memory_shards"] * p.header->shardBytes;
    result["memory_peak_blocks"] = p.header->highWater.load();
    result["backend_load_blocks"] = p.backendBlocks.load();
    result["backend_load_shards"] = p.backendShards.load();
    result["topology_nodes"] = p.index.Size();
    result["d2h_bytes"] = p.d2h.load();
    result["h2d_bytes"] = p.h2d.load();
    return result;
}
void* TransBuffer::DataAt(Index pos)
{
    auto& p = *impl_;
    return p.data + pos * p.header->shardBytes;
}
void* TransBuffer::DeviceDataAt(Index pos)
{
    auto& p = *impl_;
    return p.deviceData + (static_cast<char*>(DataAt(pos)) - p.data);
}
void TransBuffer::Acquire(Index pos)
{
    auto& p = *impl_;
    std::lock_guard<SpinLock> l(p.shards[pos].mutex);
    ++p.shards[pos].references;
}
void TransBuffer::Release(Index pos)
{
    auto& p = *impl_;
    bool last;
    {
        std::lock_guard<SpinLock> l(p.shards[pos].mutex);
        last = --p.shards[pos].references == 0;
        if (last) { p.blocks[p.shards[pos].block].activeShards.fetch_sub(1); }
    }
    if (last && p.header->pressureActive.load()) {
        std::lock_guard<Mutex> a(p.header->allocation);
        p.header->released.fetch_add(1);
        pthread_cond_broadcast(&p.header->changed);
    }
}
bool TransBuffer::Ready(Index pos) { return GetState(pos) == State::READY; }
TransBuffer::State TransBuffer::GetState(Index pos)
{ return impl_->shards[pos].state.load(std::memory_order_acquire); }
Status TransBuffer::FailureStatus(Index pos)
{
    return Status{impl_->shards[pos].error.load(std::memory_order_acquire),
                  "context shard fill failed"};
}
void TransBuffer::MarkReady(Index pos, bool backend)
{
    auto& p = *impl_;
    if (p.shards[pos].state.exchange(State::READY, std::memory_order_acq_rel) == State::READY) {
        return;
    }
    auto& block = p.blocks[p.shards[pos].block];
    block.ready.fetch_add(1);
    if (backend) {
        p.backendShards.fetch_add(1);
        if (!block.loaded.exchange(true)) { p.backendBlocks.fetch_add(1); }
    } else {
        p.d2h.fetch_add(
            std::accumulate(p.cfg.tensorSizes.begin(), p.cfg.tensorSizes.end(), size_t(0)));
    }
}
void TransBuffer::RecordRead(size_t count)
{
    auto& p = *impl_;
    p.h2d.fetch_add(count *
                    std::accumulate(p.cfg.tensorSizes.begin(), p.cfg.tensorSizes.end(), size_t(0)));
}
void TransBuffer::MarkFailed(Index pos, const Status& s)
{
    auto& node = impl_->shards[pos];
    node.error.store(s.Underlying());
    if (node.state.exchange(State::FAILED, std::memory_order_acq_rel) != State::FAILED) {
        impl_->blocks[node.block].failed.fetch_add(1);
        impl_->header->failures.fetch_add(1);
    }
}
}  // namespace UC::Context
