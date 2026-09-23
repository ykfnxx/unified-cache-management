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
#include "shared_metadata.h"
#include <cerrno>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <limits>
#include <new>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace UC::Context {
struct SharedMetadata::Header {
    std::atomic<uint64_t> ready{0};
    size_t capacity = 0;
    uint64_t layout = 0;
    pthread_mutex_t lock;
    pthread_cond_t changed;
    size_t ranks = 1, batchCapacity = 0, batchUsed = 0;
    uint64_t nextBatch = 0;
    bool alive = true;
};
struct SharedMetadata::Entry {
    Key key{};
    uint8_t state = 0, copies = 0;
    size_t memory = 0, readers = 0;
    bool busy = false;
};
struct SharedMetadata::Batch {
    Key signature{};
    uint64_t id = 0, joined = 0, done = 0;
    int32_t failure = 0;
    timespec deadline{};
};
static constexpr uint64_t magic = 0x43545853544f0004ULL;
static_assert(std::atomic<uint64_t>::is_always_lock_free);
SharedMetadata::~SharedMetadata() { Close(); }
Status SharedMetadata::Setup(const std::string& name, bool owner, size_t capacity, uint64_t layout,
                             size_t ranks, size_t batches)
{
    if (name.empty() || name.find_first_not_of(
                            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-") !=
                            std::string::npos) {
        return Status::InvalidParam("context unique_id must contain only letters, digits, _ or -");
    }
    name_ = "/ucm_context_" + name;
    owner_ = owner;
    if (!owner) { return Status::OK(); }  // Scheduler may start before its worker.
    if (!ranks || ranks > 64 ||
        batches > (std::numeric_limits<size_t>::max() - sizeof(Header)) / sizeof(Batch) ||
        !capacity ||
        capacity > (std::numeric_limits<size_t>::max() - sizeof(Header) - batches * sizeof(Batch)) /
                       sizeof(Entry)) {
        return Status::InvalidParam("invalid shared metadata capacity");
    }
    int fd = shm_open(name_.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
    if (fd < 0) {
        return Status::Error("cannot create context metadata: " + std::string(strerror(errno)));
    }
    bytes_ = sizeof(Header) + capacity * sizeof(Entry) + batches * sizeof(Batch);
    if (ftruncate(fd, bytes_) != 0) {
        close(fd);
        shm_unlink(name_.c_str());
        return Status::Error("metadata ftruncate failed");
    }
    mapping_ = mmap(nullptr, bytes_, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (mapping_ == MAP_FAILED) {
        mapping_ = nullptr;
        shm_unlink(name_.c_str());
        return Status::Error("metadata mmap failed");
    }
    header_ = new (mapping_) Header;
    header_->capacity = capacity;
    header_->layout = layout;
    header_->ranks = ranks;
    header_->batchCapacity = batches;
    entries_ = reinterpret_cast<Entry*>(header_ + 1);
    for (size_t i = 0; i < capacity; ++i) { new (&entries_[i]) Entry; }
    batches_ = reinterpret_cast<Batch*>(entries_ + capacity);
    for (size_t i = 0; i < batches; ++i) { new (&batches_[i]) Batch; }
    pthread_condattr_t condAttr;
    pthread_condattr_init(&condAttr);
    pthread_condattr_setpshared(&condAttr, PTHREAD_PROCESS_SHARED);
    pthread_condattr_setclock(&condAttr, CLOCK_MONOTONIC);
    int condRc = pthread_cond_init(&header_->changed, &condAttr);
    pthread_condattr_destroy(&condAttr);
    if (condRc) { return Status::Error("metadata condition init failed"); }
    pthread_mutexattr_t attr;
    pthread_mutexattr_init(&attr);
    pthread_mutexattr_setpshared(&attr, PTHREAD_PROCESS_SHARED);
    pthread_mutexattr_setrobust(&attr, PTHREAD_MUTEX_ROBUST);
    int rc = pthread_mutex_init(&header_->lock, &attr);
    pthread_mutexattr_destroy(&attr);
    if (rc) { return Status::Error("metadata mutex init failed"); }
    header_->ready.store(magic, std::memory_order_release);
    return Status::OK();
}
Status SharedMetadata::OpenWatcher()
{
    if (mapping_) { return Status::OK(); }
    int fd = shm_open(name_.c_str(), O_RDWR, 0600);
    if (fd < 0) {
        return errno == ENOENT ? Status::NotFound() : Status::Error("metadata open failed");
    }
    struct stat info{};
    if (fstat(fd, &info) || info.st_size < off_t(sizeof(Header))) {
        close(fd);
        return Status::NotFound();
    }
    bytes_ = info.st_size;
    mapping_ = mmap(nullptr, bytes_, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (mapping_ == MAP_FAILED) {
        mapping_ = nullptr;
        return Status::Error("watcher mmap failed");
    }
    header_ = static_cast<Header*>(mapping_);
    const auto ready = header_->ready.load(std::memory_order_acquire);
    if (ready != magic || header_->capacity > (bytes_ - sizeof(Header)) / sizeof(Entry) ||
        header_->batchCapacity >
            (bytes_ - sizeof(Header) - header_->capacity * sizeof(Entry)) / sizeof(Batch)) {
        munmap(mapping_, bytes_);
        mapping_ = nullptr;
        header_ = nullptr;
        return ready == 0 ? Status::NotFound()
                          : Status::InvalidParam("incompatible context metadata layout");
    }
    entries_ = reinterpret_cast<Entry*>(header_ + 1);
    batches_ = reinterpret_cast<Batch*>(entries_ + header_->capacity);
    return Status::OK();
}
static bool Lock(pthread_mutex_t* lock, bool& alive)
{
    int rc = pthread_mutex_lock(lock);
    if (rc == EOWNERDEAD) {
        alive = false;
        pthread_mutex_consistent(lock);
        pthread_mutex_unlock(lock);
        return false;
    }
    return rc == 0;
}
Status SharedMetadata::Publish(const Key& key, uint8_t copies, size_t memory)
{
    if (!owner_ || !header_ || !Lock(&header_->lock, header_->alive)) {
        return Status::Error("metadata owner unavailable");
    }
    size_t firstFree = header_->capacity;
    auto start = Detail::BlockIdHasher{}(key) % header_->capacity;
    for (size_t n = 0; n < header_->capacity; ++n) {
        size_t pos = (start + n) % header_->capacity;
        auto& entry = entries_[pos];
        if (entry.state == 1 && entry.key == key) {
            const bool changed = entry.copies != copies || entry.memory != memory || entry.busy;
            entry.copies = copies;
            entry.memory = memory;
            entry.busy = false;
            if (!copies) {
                // Backward-shift deletion preserves probe chains without tombstones.
                size_t hole = pos;
                for (size_t next = (hole + 1) % header_->capacity;
                     next != pos && entries_[next].state; next = (next + 1) % header_->capacity) {
                    size_t home = Detail::BlockIdHasher{}(entries_[next].key) % header_->capacity;
                    if ((hole + header_->capacity - home) % header_->capacity <
                        (next + header_->capacity - home) % header_->capacity) {
                        entries_[hole] = entries_[next];
                        hole = next;
                    }
                }
                entries_[hole] = Entry{};
            }
            if (changed) { pthread_cond_broadcast(&header_->changed); }
            pthread_mutex_unlock(&header_->lock);
            return Status::OK();
        }
        if (entry.state != 1 && firstFree == header_->capacity) { firstFree = pos; }
        if (entry.state == 0) { break; }
    }
    if (copies && firstFree != header_->capacity) {
        entries_[firstFree] = Entry{key, 1, copies, memory, 0, false};
        pthread_cond_broadcast(&header_->changed);
    }
    pthread_mutex_unlock(&header_->lock);
    return copies && firstFree == header_->capacity ? Status::NoSpace() : Status::OK();
}
SharedMetadata::Entry* SharedMetadata::Find(const Key& key)
{
    auto start = Detail::BlockIdHasher{}(key) % header_->capacity;
    for (size_t n = 0; n < header_->capacity; ++n) {
        auto& entry = entries_[(start + n) % header_->capacity];
        if (entry.state == 0) { break; }
        if (entry.state == 1 && entry.key == key) { return &entry; }
    }
    return nullptr;
}
Expected<SharedMetadata::Location> SharedMetadata::Acquire(const Key& key, uint64_t layout)
{
    std::lock_guard<std::mutex> guard(mutex_);
    if (!mapping_) {
        auto status = OpenWatcher();
        if (status.Failure()) { return status; }
    }
    if (header_->ready.load(std::memory_order_acquire) != magic) { return Status::NotFound(); }
    if (header_->layout != layout) { return Status::InvalidParam("context MLA layout mismatch"); }
    if (!Lock(&header_->lock, header_->alive)) { return Status::Error("metadata writer failed"); }
    auto* entry = header_->alive ? Find(key) : nullptr;
    if (!entry || !entry->copies || entry->busy) {
        pthread_mutex_unlock(&header_->lock);
        return Status::NotFound();
    }
    ++entry->readers;
    Location location{entry->memory};
    pthread_mutex_unlock(&header_->lock);
    return location;
}
void SharedMetadata::Release(const Key& key) { Release(&key, 1); }
void SharedMetadata::Release(const Key* keys, size_t count)
{
    std::lock_guard<std::mutex> guard(mutex_);
    if (!header_ || !Lock(&header_->lock, header_->alive)) { return; }
    for (size_t i = 0; i < count; ++i) {
        auto* entry = Find(keys[i]);
        if (entry && entry->readers) { --entry->readers; }
    }
    pthread_mutex_unlock(&header_->lock);
}
bool SharedMetadata::Evictable(const Key& key)
{
    if (!header_ || !Lock(&header_->lock, header_->alive)) { return false; }
    auto* entry = Find(key);
    bool result = entry && !entry->readers && !entry->busy;
    pthread_mutex_unlock(&header_->lock);
    return result;
}
bool SharedMetadata::ReserveEviction(const std::vector<Key>& keys)
{
    if (!header_ || !Lock(&header_->lock, header_->alive)) { return false; }
    for (const auto& key : keys) {
        auto* entry = Find(key);
        if (!entry || entry->readers || entry->busy) {
            pthread_mutex_unlock(&header_->lock);
            return false;
        }
    }
    for (const auto& key : keys) { Find(key)->busy = true; }
    pthread_mutex_unlock(&header_->lock);
    return true;
}
Expected<std::vector<uint8_t>> SharedMetadata::Lookup(const Key* keys, size_t count)
{
    std::lock_guard<std::mutex> guard(mutex_);
    std::vector<uint8_t> result(count, 0);
    if (!mapping_) {
        auto status = OpenWatcher();
        if (status == Status::NotFound()) { return result; }
        if (status.Failure()) { return status; }
    }
    if (header_->ready.load(std::memory_order_acquire) != magic) { return result; }
    if (header_->capacity > (bytes_ - sizeof(Header)) / sizeof(Entry)) {
        return Status::Error("invalid metadata header");
    }
    if (!Lock(&header_->lock, header_->alive)) { return Status::Error("metadata writer failed"); }
    if (header_->alive) {
        for (size_t i = 0; i < count; ++i) {
            auto start = Detail::BlockIdHasher{}(keys[i]) % header_->capacity;
            for (size_t n = 0; n < header_->capacity; ++n) {
                auto& entry = entries_[(start + n) % header_->capacity];
                if (entry.state == 0) { break; }
                if (entry.state == 1 && entry.key == keys[i]) {
                    result[i] = entry.copies != 0 && !entry.busy;
                    break;
                }
            }
        }
    }
    pthread_mutex_unlock(&header_->lock);
    return result;
}
SharedMetadata::Batch* SharedMetadata::FindBatch(uint64_t id)
{
    if (!id || !header_->batchCapacity) { return nullptr; }
    auto& batch = batches_[id % header_->batchCapacity];
    return batch.id == id ? &batch : nullptr;
}
Expected<uint64_t> SharedMetadata::BeginLoad(const Key& signature, size_t rank, uint64_t timeoutMs)
{
    std::lock_guard<std::mutex> guard(mutex_);
    if (!mapping_) {
        auto status = OpenWatcher();
        if (status.Failure()) { return status; }
    }
    if (rank >= header_->ranks || !header_->batchCapacity) {
        return Status::InvalidParam("invalid context load rank/batch capacity");
    }
    if (!Lock(&header_->lock, header_->alive)) { return Status::Error("metadata writer failed"); }
    timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    Batch *match = nullptr, *free = nullptr;
    const uint64_t all = UINT64_MAX >> (64 - header_->ranks), bit = uint64_t(1) << rank;
    for (size_t i = 0; i < header_->batchUsed; ++i) {
        auto& b = batches_[i];
        bool expired = now.tv_sec > b.deadline.tv_sec ||
                       (now.tv_sec == b.deadline.tv_sec && now.tv_nsec >= b.deadline.tv_nsec);
        if (!b.id || b.done == all || (expired && b.joined == b.done)) {
            if (!free) { free = &b; }
        } else if (b.signature == signature && !(b.joined & bit) && (!match || b.id < match->id)) {
            match = &b;
        }
    }
    if (!match && !free && header_->batchUsed < header_->batchCapacity) {
        free = &batches_[header_->batchUsed++];
    }
    Status status = Status::OK();
    if (!header_->alive) {
        status = Status::Error("context owner stopped");
    } else if (!match && !free) {
        status = Status::NoSpace();
    } else {
        if (!match) {
            match = free;
            *match = Batch{};
            match->signature = signature;
            match->id = ++header_->nextBatch * header_->batchCapacity + size_t(match - batches_);
            match->deadline = now;
            match->deadline.tv_sec += timeoutMs / 1000;
            match->deadline.tv_nsec += (timeoutMs % 1000) * 1000000;
            match->deadline.tv_sec += match->deadline.tv_nsec / 1000000000;
            match->deadline.tv_nsec %= 1000000000;
        }
        match->joined |= bit;
    }
    uint64_t id = status.Success() ? match->id : 0;
    pthread_mutex_unlock(&header_->lock);
    return status.Success() ? Expected<uint64_t>(uint64_t(id)) : Expected<uint64_t>(status);
}
Status SharedMetadata::WaitChange(const Batch& batch)
{
    int rc = pthread_cond_timedwait(&header_->changed, &header_->lock, &batch.deadline);
    if (rc == EOWNERDEAD) {
        header_->alive = false;
        pthread_mutex_consistent(&header_->lock);
        pthread_cond_broadcast(&header_->changed);
    }
    if (!rc) { return Status::OK(); }
    return rc == ETIMEDOUT ? Status::Timeout() : Status::Error("context notification failed");
}
Expected<SharedMetadata::Location> SharedMetadata::WaitAcquire(const Key& key, uint64_t layout,
                                                               uint64_t batch)
{
    Location location{};
    auto result = WaitAcquire(&key, 1, &location, layout, batch);
    return result ? Expected<Location>(Location{location}) : Expected<Location>(result.Error());
}
Expected<size_t> SharedMetadata::WaitAcquire(const Key* keys, size_t count, Location* locations,
                                             uint64_t layout, uint64_t batch)
{
    if (!count) { return size_t(0); }
    // Mapping is established by BeginLoad, and remains live until consumers join.
    if (!Lock(&header_->lock, header_->alive)) { return Status::Error("metadata writer failed"); }
    auto* b = FindBatch(batch);
    Status status = Status::OK();
    size_t acquired = 0;
    for (;;) {
        if (!b || header_->layout != layout) {
            status = Status::InvalidParam("context load layout/batch mismatch");
            break;
        }
        if (!header_->alive) {
            status = Status::Error("context owner stopped");
            break;
        }
        if (b->failure) {
            status = Status{b->failure, "context peer load failed"};
            break;
        }
        auto* entry = Find(keys[0]);
        if (entry && entry->copies && !entry->busy) {
            do {
                ++entry->readers;
                locations[acquired++].slot = entry->memory;
                if (acquired == count) { break; }
                entry = Find(keys[acquired]);
            } while (entry && entry->copies && !entry->busy);
            break;
        }
        status = WaitChange(*b);
        if (status.Failure()) { break; }
    }
    if (status == Status::Timeout() && b) {
        status =
            Status{status.Underlying(),
                   fmt::format("context WaitAcquire timeout: batch={}, joined={:#x}, done={:#x}",
                               batch, b->joined, b->done)};
    }
    pthread_mutex_unlock(&header_->lock);
    return status.Success() ? Expected<size_t>(size_t(acquired)) : Expected<size_t>(status);
}
void SharedMetadata::FailLoad(uint64_t batch, const Status& status)
{
    if (!batch || status.Success() || !Lock(&header_->lock, header_->alive)) { return; }
    if (auto* b = FindBatch(batch); b && !b->failure) { b->failure = status.Underlying(); }
    pthread_cond_broadcast(&header_->changed);
    pthread_mutex_unlock(&header_->lock);
}
Status SharedMetadata::WaitReaders(uint64_t batch, bool wait)
{
    if (!Lock(&header_->lock, header_->alive)) { return Status::Error("metadata writer failed"); }
    auto* b = FindBatch(batch);
    Status status = Status::OK();
    const auto readers = (UINT64_MAX >> (64 - header_->ranks)) & ~uint64_t(1);
    while (b && header_->alive && !b->failure && (b->done & readers) != readers) {
        if (!wait) {
            timespec now;
            clock_gettime(CLOCK_MONOTONIC, &now);
            bool expired = now.tv_sec > b->deadline.tv_sec ||
                           (now.tv_sec == b->deadline.tv_sec && now.tv_nsec >= b->deadline.tv_nsec);
            status = expired ? Status::Timeout() : Status::Retry();
            break;
        }
        status = WaitChange(*b);
        if (status.Failure()) { break; }
    }
    if (!b || !header_->alive) {
        status = Status::Error("context owner stopped or batch missing");
    } else if (b->failure) {
        status = Status{b->failure, "context peer load failed"};
    }
    if (status == Status::Timeout() && b) {
        status = Status{status.Underlying(),
                        fmt::format("context WaitReaders timeout: batch={}, joined={:#x}, "
                                    "done={:#x}, expected_readers={:#x}",
                                    batch, b->joined, b->done, readers)};
    }
    pthread_mutex_unlock(&header_->lock);
    return status;
}
void SharedMetadata::EndLoad(uint64_t batch, size_t rank)
{
    if (!batch || !Lock(&header_->lock, header_->alive)) { return; }
    if (auto* b = FindBatch(batch)) { b->done |= uint64_t(1) << rank; }
    // Only fully completed trailing slots leave the matching scan. Retain their
    // IDs until reuse, so completion inspection remains valid after EndLoad.
    const auto all = UINT64_MAX >> (64 - header_->ranks);
    while (header_->batchUsed && batches_[header_->batchUsed - 1].done == all) {
        --header_->batchUsed;
    }
    pthread_cond_broadcast(&header_->changed);
    pthread_mutex_unlock(&header_->lock);
}
void SharedMetadata::Deactivate()
{
    std::lock_guard<std::mutex> guard(mutex_);
    if (owner_ && header_ && header_->ready.load(std::memory_order_acquire) == magic &&
        Lock(&header_->lock, header_->alive)) {
        header_->alive = false;
        pthread_cond_broadcast(&header_->changed);
        pthread_mutex_unlock(&header_->lock);
    }
}
void SharedMetadata::Close()
{
    std::lock_guard<std::mutex> guard(mutex_);
    if (!mapping_) { return; }
    if (owner_) {
        if (header_->ready.load(std::memory_order_acquire) == magic &&
            Lock(&header_->lock, header_->alive)) {
            header_->alive = false;
            pthread_cond_broadcast(&header_->changed);
            pthread_mutex_unlock(&header_->lock);
        }
        shm_unlink(name_.c_str());
    }
    munmap(mapping_, bytes_);
    mapping_ = nullptr;
    header_ = nullptr;
    entries_ = nullptr;
}
}  // namespace UC::Context
