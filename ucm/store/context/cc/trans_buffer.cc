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
// Buffer handles and shard fill ownership follow CacheStore's TransBuffer.
// Context owns allocation/eviction at block granularity; transfer handles never touch the tree.
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
constexpr uint64_t magic = 0x55434d4354580004ULL;
constexpr size_t buckets = 4099;
constexpr size_t evicting = size_t(1) << 63;
struct Mutex {
    pthread_mutex_t value;
    void Init()
    {
        pthread_mutexattr_t a;
        pthread_mutexattr_init(&a);
        pthread_mutexattr_setpshared(&a, PTHREAD_PROCESS_SHARED);
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
}  // namespace
struct TransBuffer::Impl {
    struct alignas(64) Block {
        Key key;
        size_t next = nil, pending = nil;
        std::atomic<size_t> references{0}, ready{0}, failed{0};
        std::atomic<bool> loaded{false};
    };
    struct alignas(64) Shard {
        SpinLock mutex;
        size_t references = 0;
        std::atomic<State> state{State::LOADING};
        std::atomic<int32_t> error{0};
    };
    struct Header {
        uint64_t version, layout;
        size_t count, layers, shardBytes, metadataBytes, totalBytes;
        Mutex allocation;
        pthread_cond_t changed;
        size_t freeHead, freeCount, pendingHead = nil;
        size_t pressure = 0, completed = 0;
        int32_t allocationError = 0;
        std::atomic<bool> ownerAlive{false};
        std::atomic<size_t> tasks{0}, highWater{0}, failures{0};

        size_t heads[buckets];
        Mutex locks[buckets];
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
        size_t count = watcher ? 0 : cfg.bufferCapacity / cfg.blockSize;
        size_t layers = watcher ? 0 : cfg.blockSize / cfg.shardSize;
        size_t meta = watcher ? 0
                              : Align(ShardOffset(count) + sizeof(Shard) * count * layers,
                                      sysconf(_SC_PAGESIZE));
        size_t total = meta + count * cfg.blockSize;
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
        mapping = mmap(nullptr, mappedBytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
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
            pthread_condattr_t a;
            pthread_condattr_init(&a);
            pthread_condattr_setpshared(&a, PTHREAD_PROCESS_SHARED);
            pthread_condattr_setclock(&a, CLOCK_MONOTONIC);
            pthread_cond_init(&header->changed, &a);
            pthread_condattr_destroy(&a);
            header->freeHead = 0;
            header->freeCount = count;
            for (size_t i = 0; i < buckets; ++i) {
                header->heads[i] = nil;
                header->locks[i].Init();
            }
            Attach();
            for (size_t i = 0; i < count; ++i) {
                new (&blocks[i]) Block{};
                blocks[i].next = i + 1 == count ? nil : i + 1;
            }
            for (size_t i = 0; i < count * layers; ++i) {
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
    void Drain()
    {  // policy -> allocation; no transfer takes policy under a buffer lock.
        size_t i;
        {
            std::lock_guard<Mutex> l(header->allocation);
            i = header->pendingHead;
            header->pendingHead = nil;
        }
        // Only the policy owner retires slots, so these detached entries remain
        // stable while we register them without holding the allocation lock.
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
    void Remove(size_t pos)
    {
        auto key = blocks[pos].key;
        auto failed = blocks[pos].failed.load();
        {
            std::lock_guard<Mutex> a(header->allocation);
            std::lock_guard<Mutex> b(header->locks[Hash(key)]);
            auto* link = &header->heads[Hash(key)];
            while (*link != pos) { link = &blocks[*link].next; }
            *link = blocks[pos].next;
            blocks[pos].next = header->freeHead;
            header->freeHead = pos;
            ++header->freeCount;
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
    Status Evict()
    {
        // Failed empty fills are allocations to roll back, not policy evictions.
        if (header->failures.load())
            for (const auto& item : resident) {
                auto& block = blocks[item.second];
                size_t ref = 0;
                if (block.ready.load() == 0 && block.failed.load() &&
                    block.references.compare_exchange_strong(ref, evicting)) {
                    Remove(item.second);
                    return Status::OK();
                }
            }
        auto begin = std::chrono::steady_clock::now();
        auto victim =
            index.Select(cfg.alpha * header->count, cfg.evictionLimit, [this](const Key& key) {
                auto pos = resident.at(key);
                return blocks[pos].references.load() == 0 &&
                       (blocks[pos].ready.load() == header->layers || blocks[pos].loaded.load());
            });
        stats["decision_ns"] += std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    std::chrono::steady_clock::now() - begin)
                                    .count();
        ++stats["eviction_decisions"];
        if (victim.blocks.empty()) {
            ++stats["no_space"];
            return Status::NoSpace();
        }
        bool drop = cfg.retention >= 0 && now - victim.lastAccess > uint64_t(cfg.retention);
        size_t freed = 0;
        for (const auto& key : victim.blocks) {
            auto pos = resident.at(key);
            auto& block = blocks[pos];
            size_t expected = 0;
            if (!block.references.compare_exchange_strong(expected, evicting)) { continue; }
            auto s = Status::OK();
            if (!drop) {
                auto backendKey = cfg.BackendKey(key);
                auto found = cfg.storeBackend->Lookup(&backendKey, 1);
                if (!found) {
                    s = found.Error();
                } else if (found.Value()[0]) {
                    ++stats["backend_dump_skipped_blocks"];
                } else {
                    if (block.ready.load() != header->layers) {
                        block.references.store(0);
                        return Status::NotFound();
                    }
                    Detail::TaskDesc task;
                    for (size_t layer = 0; layer < header->layers; ++layer) {
                        task.push_back(
                            {backendKey,
                             layer,
                             {data + (layer * header->count + pos) * header->shardBytes}});
                    }
                    auto t = cfg.storeBackend->Dump(std::move(task));
                    s = t ? cfg.storeBackend->Wait(t.Value()) : t.Error();
                    if (s.Success()) {
                        ++stats["backend_dump_blocks"];
                        stats["backend_dump_bytes"] += cfg.blockSize;
                    }
                }
            }
            if (s.Failure()) {
                block.references.store(0);
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
        return freed ? Status::OK() : Status::NoSpace();
    }
    void Maintain()
    {
        for (;;) {
            size_t request;
            {
                std::unique_lock<Mutex> a(header->allocation);
                while (!stop && header->pressure == header->completed) {
                    pthread_cond_wait(&header->changed, &header->allocation.value);
                }
                if (stop) { return; }
                request = header->pressure;
            }
            auto s = Status::OK();
            {
                std::lock_guard<std::mutex> p(policy);
                Drain();
                s = Evict();
                Prune();
            }
            {
                std::lock_guard<Mutex> a(header->allocation);
                header->allocationError = s.Underlying();
                header->completed = request;
                pthread_cond_broadcast(&header->changed);
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
    p.name = "/ucm_context_v4_" + c.uniqueId +
             (c.shareBufferEnable ? "_mla" : "_tp" + std::to_string(c.tpRank));
    if (c.deviceId < 0) { return Status::OK(); }
    return p.Open(false);
}
Expected<TransBuffer::Handle> TransBuffer::Get(const Key& key, size_t layer)
{
    auto& p = *impl_;
    auto* h = p.header;
    size_t pos = nil;
    size_t bucket = Hash(key);
    if (!h->ownerAlive.load()) { return Status::NotFound(); }
    auto pin = [&] {
        pos = p.Find(key);
        if (pos == nil) { return false; }
        auto ref = p.blocks[pos].references.load();
        while (ref != evicting) {
            if (p.blocks[pos].references.compare_exchange_weak(ref, ref + 1)) { return true; }
        }
        pos = nil;
        return false;
    };
    {
        std::lock_guard<Mutex> b(h->locks[bucket]);
        pin();
    }
    if (pos == nil) {
        if (p.policyOwner) {
            std::lock_guard<std::mutex> guard(p.policy);
            if (!p.index.Contains(key)) {
                return Status::InvalidParam("ObserveRequest required before new block allocation");
            }
        }
        std::unique_lock<Mutex> a(h->allocation);
        timespec deadline;
        clock_gettime(CLOCK_MONOTONIC, &deadline);
        deadline.tv_sec += p.cfg.timeoutMs / 1000;
        deadline.tv_nsec += (p.cfg.timeoutMs % 1000) * 1000000;
        deadline.tv_sec += deadline.tv_nsec / 1000000000;
        deadline.tv_nsec %= 1000000000;
        for (;;) {
            {
                std::lock_guard<Mutex> b(h->locks[bucket]);
                if (pin()) { break; }
                if (p.Find(key) == nil && h->freeCount) {
                    pos = h->freeHead;
                    auto& block = p.blocks[pos];
                    h->freeHead = block.next;
                    --h->freeCount;
                    block.key = key;
                    block.references.store(1);
                    block.ready.store(0);
                    block.failed.store(0);
                    block.loaded.store(false);
                    block.next = h->heads[bucket];
                    h->heads[bucket] = pos;
                    block.pending = h->pendingHead;
                    h->pendingHead = pos;
                    for (size_t i = 0; i < h->layers; ++i) {
                        auto& shard = p.shards[pos * h->layers + i];
                        shard.references = 0;
                        shard.state.store(State::LOADING);
                        shard.error.store(0);
                    }
                    h->highWater.store(std::max(h->highWater.load(), h->count - h->freeCount));
                    break;
                }
            }
            auto request = h->pressure == h->completed ? ++h->pressure : h->pressure;
            pthread_cond_broadcast(&h->changed);
            while (h->completed < request) {
                if (pthread_cond_timedwait(&h->changed, &h->allocation.value, &deadline) ==
                    ETIMEDOUT) {
                    return Status::Timeout();
                }
            }
            {
                std::lock_guard<Mutex> b(h->locks[bucket]);
                if (pin()) { break; }
            }
            if (!h->freeCount && h->allocationError) {
                return Status{h->allocationError, "context allocation failed"};
            }
        }
    }
    auto offset = pos * h->layers + layer;
    auto& node = p.shards[offset];
    std::lock_guard<SpinLock> n(node.mutex);
    bool owner = node.references == 0;
    if (owner && node.state.load() == State::FAILED) {
        node.error.store(0);
        p.blocks[pos].failed.fetch_sub(1);
        h->failures.fetch_sub(1);
        node.state.store(State::LOADING);
    }
    ++node.references;
    return Handle(this, offset, owner);
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
    return pos != nil && p.blocks[pos].references.load() != evicting &&
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
        std::set<Key> keys;
        for (const auto& s : task) { keys.insert(s.owner); }
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
    std::lock_guard<Mutex> a(p.header->allocation);
    result["memory_blocks"] = p.header->count - p.header->freeCount;
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
    return p.data + ((pos % p.header->layers) * p.header->count + pos / p.header->layers) *
                        p.header->shardBytes;
}
void* TransBuffer::DeviceDataAt(Index pos)
{
    auto& p = *impl_;
    return p.deviceData + (static_cast<char*>(DataAt(pos)) - p.data);
}
void TransBuffer::Acquire(Index pos)
{
    auto& p = *impl_;
    p.blocks[pos / p.header->layers].references.fetch_add(1);
    std::lock_guard<SpinLock> l(p.shards[pos].mutex);
    ++p.shards[pos].references;
}
void TransBuffer::Release(Index pos)
{
    auto& p = *impl_;
    {
        std::lock_guard<SpinLock> l(p.shards[pos].mutex);
        --p.shards[pos].references;
    }
    p.blocks[pos / p.header->layers].references.fetch_sub(1);
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
    auto& block = p.blocks[pos / p.header->layers];
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
        impl_->blocks[pos / impl_->header->layers].failed.fetch_add(1);
        impl_->header->failures.fetch_add(1);
    }
}
}  // namespace UC::Context
