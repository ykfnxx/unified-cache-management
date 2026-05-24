# Memory Store Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a first-version `MemoryStore` backend with cache-store-compatible standard APIs plus batched per-token-per-layer APIs.

**Architecture:** Add token-layer API defaults to `StoreV1`, implement a new `ucm/store/memory` shared library, and expose the APIs through pipeline pybind and Python `UcmPipelineStore`. The first implementation uses CPU memory copies, a single mutex-protected LRU pool, typed token chunks, and full-shard backend interaction.

**Tech Stack:** C++17, CMake, pybind11, Python connector code, GoogleTest/GMock.

---

### Task 1: Add StoreV1 Token-Layer API Surface

**Files:**
- Modify: `ucm/store/detail/type/types.h`
- Modify: `ucm/store/ucmstore_v1.h`
- Test: `ucm/store/test/case/memory/memory_store_test.cc`

- [ ] **Step 1: Add a failing test that references the new token-layer API**

Create `ucm/store/test/case/memory/memory_store_test.cc` with a minimal compile-time test:

```cpp
#include <gtest/gtest.h>
#include "ucmstore_v1.h"

TEST(UCMTokenLayerApiTest, DefaultStoreV1TokenLayerMethodsAreUnsupported)
{
    class MinimalStore : public UC::StoreV1 {
    public:
        UC::Status Setup(const UC::Detail::Dictionary&) override { return UC::Status::OK(); }
        std::string Readme() const override { return "MinimalStore"; }
        UC::Expected<std::vector<uint8_t>> Lookup(const UC::Detail::BlockId*, size_t num) override
        {
            return std::vector<uint8_t>(num, false);
        }
        UC::Expected<ssize_t> LookupOnPrefix(const UC::Detail::BlockId*, size_t) override
        {
            return -1;
        }
        void Prefetch(const UC::Detail::BlockId*, size_t) override {}
        UC::Expected<UC::Detail::TaskHandle> Load(UC::Detail::TaskDesc) override { return 1; }
        UC::Expected<UC::Detail::TaskHandle> Dump(UC::Detail::TaskDesc) override { return 2; }
        UC::Expected<bool> Check(UC::Detail::TaskHandle) override { return true; }
        UC::Status Wait(UC::Detail::TaskHandle) override { return UC::Status::OK(); }
    };

    MinimalStore store;
    UC::Detail::TokenLayerTaskDesc task;
    auto lookup = store.LookupTokens(task);
    auto load = store.LoadTokens(task);
    auto dump = store.DumpTokens(task);
    EXPECT_FALSE(lookup.HasValue());
    EXPECT_FALSE(load.HasValue());
    EXPECT_FALSE(dump.HasValue());
}
```

- [ ] **Step 2: Run test build and verify it fails because token-layer types are missing**

Run: `cmake --build build --target ucmstore.test -j`

Expected: compile failure mentioning `TokenLayerTaskDesc` or `LookupTokens`.

- [ ] **Step 3: Add token-layer descriptor and StoreV1 default methods**

In `ucm/store/detail/type/types.h`, add:

```cpp
using TensorType = std::size_t;

struct TokenLayerShard {
    BlockId owner;
    std::size_t layer;
    std::size_t tokenOffset;
    TensorType tensorType;
    std::vector<void*> addrs;
};

struct TokenLayerTaskDesc : std::vector<TokenLayerShard> {
    using vector::vector;
    std::string brief;
    uintptr_t prerequisiteHandle{0};
};
```

In `ucm/store/ucmstore_v1.h`, add default methods:

```cpp
virtual Expected<std::vector<uint8_t>> LookupTokens(const Detail::TokenLayerTaskDesc& task)
{
    return Status::NotSupport();
}

virtual Expected<Detail::TaskHandle> LoadTokens(Detail::TokenLayerTaskDesc task)
{
    return Status::NotSupport();
}

virtual Expected<Detail::TaskHandle> DumpTokens(Detail::TokenLayerTaskDesc task)
{
    return Status::NotSupport();
}
```

If `Status::NotSupport()` does not exist, use `Status::Error("not support")`.

- [ ] **Step 4: Run test build and verify it passes**

Run: `cmake --build build --target ucmstore.test -j`

Expected: build succeeds.

### Task 2: Implement MemoryStore Core

**Files:**
- Create: `ucm/store/memory/CMakeLists.txt`
- Create: `ucm/store/memory/__init__.py`
- Create: `ucm/store/memory/cc/memory_store.cc`
- Modify: `ucm/store/CMakeLists.txt`
- Modify: `ucm/store/test/CMakeLists.txt`
- Test: `ucm/store/test/case/memory/memory_store_test.cc`

- [ ] **Step 1: Add failing MemoryStore behavior tests**

Extend `ucm/store/test/case/memory/memory_store_test.cc` with tests for setup, standard dump/load, token dump/load, and partial token lookup:

```cpp
#include "detail/types_helper.h"

TEST(UCMMemoryStoreTest, TokenDumpLoadRoundTripUsesTensorType)
{
    auto store = std::unique_ptr<UC::StoreV1>(MakeMemoryStore());
    UC::Detail::Dictionary config;
    config.Set<UC::StoreV1*>("store_backend", nullptr);
    config.SetNumber("shard_size", 16);
    config.SetNumber("block_size", 16);
    config.SetNumber("memory_token_chunk_size", 2);
    config.SetNumber("memory_buffer_capacity_gb", 1);
    config.Set("memory_required_tensor_types", std::vector<ssize_t>{0, 1});
    config.Set("memory_tensor_size_by_type_0", std::vector<ssize_t>{4});
    config.Set("memory_tensor_size_by_type_1", std::vector<ssize_t>{4});
    ASSERT_EQ(store->Setup(config), UC::Status::OK());

    auto block = UC::Test::Detail::TypesHelper::MakeBlockIdRandomly();
    std::array<std::byte, 4> src{std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4}};
    std::array<std::byte, 4> dst{};
    UC::Detail::TokenLayerTaskDesc dump;
    dump.push_back({block, 0, 0, 0, {src.data()}});
    auto dumpTask = store->DumpTokens(dump);
    ASSERT_TRUE(dumpTask.HasValue());
    ASSERT_EQ(store->Wait(dumpTask.Value()), UC::Status::OK());

    UC::Detail::TokenLayerTaskDesc query;
    query.push_back({block, 0, 0, 0, {dst.data()}});
    auto lookup = store->LookupTokens(query);
    ASSERT_TRUE(lookup.HasValue());
    ASSERT_EQ(lookup.Value(), std::vector<uint8_t>{1});

    auto loadTask = store->LoadTokens(query);
    ASSERT_TRUE(loadTask.HasValue());
    ASSERT_EQ(store->Wait(loadTask.Value()), UC::Status::OK());
    EXPECT_EQ(std::memcmp(src.data(), dst.data(), src.size()), 0);
}
```

- [ ] **Step 2: Run and verify tests fail because MemoryStore does not exist**

Run: `cmake --build build --target ucmstore.test -j`

Expected: compile or link failure mentioning `MakeMemoryStore` or missing `memorystore`.

- [ ] **Step 3: Create MemoryStore with minimal synchronous task completion**

Implement `MemoryStore` as a `StoreV1` subclass. Use a mutex, monotonically increasing task ids, `std::unordered_map` metadata, `std::list` LRU order, and `std::vector<std::byte>` chunk payloads. Token dump copies from source addrs into typed token entries. Token load copies from typed token entries into destination addrs. Standard dump splits mixed full shard into typed token entries when type layout is derivable. Standard load assembles mixed full shard from typed entries when full shard is ready.

- [ ] **Step 4: Add CMake wiring**

Create `ucm/store/memory/CMakeLists.txt`:

```cmake
file(GLOB_RECURSE UCM_MEMORY_STORE_CC_SOURCE_FILES "./cc/*.cc")
add_library(memorystore SHARED ${UCM_MEMORY_STORE_CC_SOURCE_FILES})
target_include_directories(memorystore PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cc)
target_link_libraries(memorystore PUBLIC storeintf infra_logger)
file(RELATIVE_PATH INSTALL_REL_PATH ${UCM_ROOT_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
install(TARGETS memorystore LIBRARY DESTINATION ${INSTALL_REL_PATH} COMPONENT ucm)
```

Add `add_subdirectory(memory)` in `ucm/store/CMakeLists.txt`.

Add `memorystore` to `ucm/store/test/CMakeLists.txt` target libraries.

- [ ] **Step 5: Run and verify tests pass**

Run: `cmake --build build --target ucmstore.test -j && ./build/ucm/store/test/ucmstore.test --gtest_filter='UCMTokenLayerApiTest.*:UCMMemoryStoreTest.*'`

Expected: tests pass.

### Task 3: Expose Token APIs Through Pipeline

**Files:**
- Modify: `ucm/store/pipeline/cpy/pipeline_store.py.cc`
- Modify: `ucm/store/pipeline/connector.py`

- [ ] **Step 1: Add pybind methods**

Add C++ pybind wrappers that parse `ids`, `layers`, `token_offsets`, `tensor_types`, and `addrs` buffers into `TokenLayerTaskDesc`, then call `LookupTokens`, `LoadTokens`, and `DumpTokens` on `StoreBack()`.

- [ ] **Step 2: Add Python connector methods**

Add `lookup_tokens_on_layer`, `load_tokens_on_layer`, and `dump_tokens_on_layer` to `UcmPipelineStore`. Each method flattens block ids like existing `lookup`, converts numeric arrays to `array.array("Q", ...)`, normalizes tensor pointers or raw addresses, and calls the pybind method.

- [ ] **Step 3: Register memory pipelines**

Add `_memory_empty_pipeline_builder` and `_memory_posix_pipeline_builder`, then register `Memory|Empty` and `Memory|Posix`.

- [ ] **Step 4: Build pipeline module**

Run: `cmake --build build --target ucmpipelinestore -j`

Expected: build succeeds.

### Task 4: Verification

**Files:**
- All modified files

- [ ] **Step 1: Run focused C++ tests**

Run: `cmake --build build --target ucmstore.test -j && ./build/ucm/store/test/ucmstore.test --gtest_filter='UCMTokenLayerApiTest.*:UCMMemoryStoreTest.*'`

Expected: all focused tests pass.

- [ ] **Step 2: Run cache tests to catch compatibility regressions**

Run: `./build/ucm/store/test/ucmstore.test --gtest_filter='UCCache*'`

Expected: cache tests pass or only fail for pre-existing environment/device reasons. Report any environment/device failures explicitly.

- [ ] **Step 3: Inspect git diff**

Run: `git diff --stat && git status --short`

Expected: only memory-store implementation, interface additions, pipeline bindings, tests, and design/plan docs changed.

