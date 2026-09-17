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
    pthread_mutex_t lock;
    bool alive = true;
};
struct SharedMetadata::Entry {
    Key key{};
    uint8_t state = 0, copies = 0;
};
static constexpr uint64_t magic = 0x43545853544f0001ULL;
static_assert(std::atomic<uint64_t>::is_always_lock_free);
SharedMetadata::~SharedMetadata() { Close(); }
Status SharedMetadata::Setup(const std::string& name, bool owner, size_t capacity)
{
    if (name.empty() || name.find_first_not_of(
                            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-") !=
                            std::string::npos) {
        return Status::InvalidParam("context unique_id must contain only letters, digits, _ or -");
    }
    name_ = "/ucm_context_" + name;
    owner_ = owner;
    if (!owner) { return Status::OK(); }  // Scheduler may start before its worker.
    if (!capacity ||
        capacity > (std::numeric_limits<size_t>::max() - sizeof(Header)) / sizeof(Entry)) {
        return Status::InvalidParam("invalid shared metadata capacity");
    }
    int fd = shm_open(name_.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
    if (fd < 0) {
        return Status::Error("cannot create context metadata: " + std::string(strerror(errno)));
    }
    bytes_ = sizeof(Header) + capacity * sizeof(Entry);
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
    entries_ = reinterpret_cast<Entry*>(header_ + 1);
    for (size_t i = 0; i < capacity; ++i) { new (&entries_[i]) Entry; }
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
    entries_ = reinterpret_cast<Entry*>(header_ + 1);
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
Status SharedMetadata::Publish(const Key& key, uint8_t copies)
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
            entry.copies = copies;
            if (!copies) { entry.state = 2; }
            pthread_mutex_unlock(&header_->lock);
            return Status::OK();
        }
        if (entry.state != 1 && firstFree == header_->capacity) { firstFree = pos; }
        if (entry.state == 0) { break; }
    }
    if (copies && firstFree != header_->capacity) { entries_[firstFree] = Entry{key, 1, copies}; }
    pthread_mutex_unlock(&header_->lock);
    return copies && firstFree == header_->capacity ? Status::NoSpace() : Status::OK();
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
                    result[i] = entry.copies != 0;
                    break;
                }
            }
        }
    }
    pthread_mutex_unlock(&header_->lock);
    return result;
}
void SharedMetadata::Deactivate()
{
    std::lock_guard<std::mutex> guard(mutex_);
    if (owner_ && header_ && header_->ready.load(std::memory_order_acquire) == magic &&
        Lock(&header_->lock, header_->alive)) {
        header_->alive = false;
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
