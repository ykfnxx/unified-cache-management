# Memory Store Design

## Goal

Add a memory-side store backend for pipeline store mode. The new backend provides
per-token-per-layer access through new APIs while keeping the existing cache-store
compatible APIs and backend storage format unchanged.

The first implementation should stay simple:

- Add an independent `MemoryStore` backend under `ucm/store/memory`.
- Keep existing `StoreV1` methods compatible with `CacheStore`.
- Add new token-layer methods through `StoreV1`, pipeline pybind, and Python
  `UcmPipelineStore`. Token-layer lookup is synchronous like existing lookup;
  token-layer load and dump are asynchronous like existing load and dump.
- Use a single in-memory pool with a simple LRU policy.
- Use token chunk management internally to reduce metadata and allocation
  overhead.
- Interact with the downstream backend using the same full-shard format that
  `CacheStore` uses.

## Existing Context

Pipeline store currently stacks stores in reverse data-flow order. For
`Cache|Posix`, the builder stacks `Posix` first and `Cache` second. Public calls
hit the top store, and the top store talks to the downstream backend through the
`store_backend` pointer.

`CacheStore` exposes the standard `StoreV1` methods:

- `Lookup`
- `LookupOnPrefix`
- `Prefetch`
- `Load`
- `Dump`
- `Check`
- `Wait`

`CacheStore` stores and transfers data by `BlockId` plus `shard_index`. In
layerwise mode, `shard_index` is the layer id, and backend stores see full
`Shard{owner = block_id, index = layer_id}` payloads.

The new `MemoryStore` must preserve this downstream backend format.

## Current Implementation vs. CacheStore

This comparison is based on the current code, not only on the original design
intent. The main implementation anchors are:

- `ucm/store/cache/cc/cache_store.cc`
- `ucm/store/cache/cc/buffer_manager.h`
- `ucm/store/cache/cc/load_queue.cc`
- `ucm/store/cache/cc/dump_queue.cc`
- `ucm/store/memory/cc/memory_store.cc`
- `ucm/store/memory/cc/trans_buffer.cc`
- `ucm/store/memory/cc/load_queue.cc`
- `ucm/store/memory/cc/dump_queue.cc`

| Aspect | Current CacheStore implementation | Current MemoryStore implementation |
| --- | --- | --- |
| API surface | Implements only the standard `StoreV1` methods. Token-layer methods still use the default unsupported behavior on `StoreV1`. | Implements `LookupTokens`, `LoadTokens`, and `DumpTokens` in addition to the standard methods. |
| Internal storage unit | Stores full `(block_id, shard_index)` shards in buffer slots. | Stores token chunks keyed by `(block_id, layer_id, chunk_id, tensor_type)` and separately tracks `(block_id, layer_id)` in a `fullReady` set. |
| Lookup path | `BufferManager` checks the local buffer first, then falls back to backend lookup on miss. `cacheLoadBackendOnly` can explicitly bypass local hits. | `Lookup`, `LookupOnPrefix`, and `LookupTokens` are local-memory checks only. Backend fetch on miss happens only through `Load` or `LoadTokens`. |
| Transfer model | Explicit host/device transfer pipeline with `device_id`, copy streams, optional GDR, dispatch threads, and transfer/backend stages. | Reuses the existing store-side `cache/cc/copy_stream.h` helper and splits transfer into staged queues. Load uses dispatch + transfer stages; dump uses dispatch + transfer + backend-host stages. User-address transfer still goes through `Trans::Stream` `HostToDeviceAsync` / `DeviceToHostAsync`. |
| Backend interaction | Backend shard load/dump is submitted asynchronously and waited in queue stages. | Token misses degrade to full-shard backend `Load`, and `TransBuffer` waits synchronously inside `LoadFullFromBackend`. Full-shard dump also waits synchronously inside `DumpFullToBackend`. |
| Config surface | Depends on `device_id`, `share_buffer_enable`, `cache_buffer_capacity_gb`, `running_queue_depth`, `stream_number`, `use_gdr`, `cpu_affinity_cores`, and related cache-transfer settings. | Currently uses `device_id`, `cache_stream_number` / `memory_stream_number`, `use_gdr`, `cpu_affinity_cores`, `shard_size`, `block_size`, `tensor_size(_list)`, `memory_token_chunk_size`, `memory_buffer_capacity_gb`, `memory_required_tensor_types`, `memory_tensor_size_by_type_*`, `waiting_queue_depth`, `running_queue_depth`, and `timeout_ms`. |

### Practical Notes About the Current Code

1. `MemoryStore` is already a separate data model, not a minor variation of
   `CacheStore`.
   `CacheStore` is centered on full-shard buffers, while `MemoryStore` is
   centered on typed token chunks and derives full-shard readiness from those
   chunks.

2. Standard `Lookup` semantics are currently more local in `MemoryStore`.
   `CacheStore::Lookup` can continue into backend lookup after a local miss.
   `MemoryStore::Lookup` only checks local `fullReady_` state, so the two stores
   are not fully equivalent on standard lookup behavior.

3. `MemoryStore` now aligns more closely with `CacheStore` on transfer
   ownership and queue staging, but it still does not match `CacheStore` on
   backend execution flow.
   The load and dump queues own caller-buffer/device transfers through the
   existing store-side `CopyStream` helper and `Trans::Stream`, while `TransBuffer`
   stays on the host side for chunk assembly, full-shard assembly, and backend
   interaction.
   `CacheStore` still keeps backend wait inside its queue-driven transfer
   pipeline, while `MemoryStore` calls backend `Load` or `Dump` and then
   synchronously waits inside `LoadFullFromBackend` or `DumpFullToBackend`
   before continuing token/full-shard conversion.

4. The current `MemoryStore` implementation does not implement the explicit
   `memory_full_shard_layout` configuration described later in this design doc.
   `SplitFullShard` and `AssembleFullShard` currently assume the full-shard byte
   layout is token-major, with `memory_required_tensor_types` concatenated in
   order inside each token. This is a stricter assumption than `CacheStore`
   makes.

5. The concurrency model in `MemoryStore` is still narrower than in
   `CacheStore`, but it is no longer single-stage.
   Today load is split into dispatch + transfer stages, and dump is split into
   dispatch + transfer + backend-host stages, while `TransBuffer` remains
   protected by a single mutex. `CacheStore` still relies on buffer handles and
   asynchronous backend wait at queue level, so the two stores have not fully
   converged.

6. Eviction happens at different physical units.
   `MemoryStore` evicts token chunks. `CacheStore` evicts shard buffer slots.
   This difference directly affects hit semantics, invalidation granularity, and
   readiness tracking even when both stores still expose a standard shard view.

This comparison should be treated as the baseline for future work: the current
`MemoryStore` preserves backend full-shard compatibility, keeps queue-owned
`Trans::Stream` transfer flow, and no longer stores the whole `Config` object
inside `TransBuffer`, but it still does not replicate `CacheStore` in backend
wait flow, concurrency staging, or lookup behavior.

## Architecture

Create a new shared library:

- Source: `ucm/store/memory`
- Exported factory: `MakeMemoryStore`
- Pipeline registrations:
  - `Memory|Empty`
  - `Memory|Posix`

The store has two public views over one physical memory pool:

1. Standard store view:
   Existing `StoreV1` APIs keep `CacheStore` semantics. A standard load or dump
   works with a full `(block_id, shard_index)` payload.

2. Token-layer view:
   New APIs operate on batches of token-layer items. Each item is identified by
   `(block_id, layer_id, token_offset)` and has its own tensor address array.
   Token-layer load and dump return one normal task handle for the whole batch.

The physical pool is managed by one LRU. The LRU eviction unit is a token chunk,
not an individual token. A token chunk represents a small fixed number of tokens
for a layer. The chunk size is configurable.

## Storage Semantics

The store must keep three semantic layers distinct.

### Physical Data

Physical memory stores token chunks. A token chunk contains token-layer payloads
for a fixed token range.

The standard store view still uses the existing `tensor_size` or
`tensor_size_list` rules for full-shard payloads. The token-layer view does not
reuse those settings. Token-layer payload layout is selected by a tensor type
passed with each token-layer item. Each tensor type has its own configured tensor
size list. That type-specific tensor size list defines how the item's address
array is gathered into one logical token-layer payload and scattered back out.

For the first version, tensor types must at least support separate K cache and V
cache payloads. K and V are stored independently in the token-layer view, so
callers may lookup, load, and dump K without V or V without K.

### Token-Layer State

Token-layer state is precise to:

```text
(block_id, layer_id, token_offset, tensor_type)
```

A token-layer dump marks only the corresponding token-layer item ready. A
token-layer load can hit only if that exact item is ready.

### Full-Shard State

Standard store state is precise to:

```text
(block_id, layer_id)
```

Only a complete layer shard can make the standard view ready. Partial token
writes must not cause standard `Lookup`, `LookupOnPrefix`, or `Load` to behave
as if a full shard is present.

This rule keeps the standard API compatible with `CacheStore`.

The standard full-shard view is derived from typed token chunks. It does not
require a second copy of the full mixed K/V shard in memory. A full shard is
ready when all tokens and all `memory_required_tensor_types` for that
`(block_id, layer_id)` are ready in the unified chunk pool.

## Backend Format

The downstream backend always receives cache-store compatible shards:

```text
Shard{owner = block_id, index = layer_id, addrs = {full_shard_buffer}}
```

The backend must never receive token-granular keys or encoded token offsets.
This avoids changing the storage format in `PosixStore`, `Ds3fsStore`,
`EmptyStore`, or other existing downstream stores.

Backend shards remain mixed K/V full-shard payloads. `MemoryStore` converts at
the boundary:

- Standard or backend full-shard load: split the mixed K/V shard into typed token
  chunks.
- Standard or backend full-shard dump: assemble the mixed K/V shard from typed
  token chunks.

This conversion requires a configured full-shard layout that maps each tensor
type to its byte positions inside the cache-store compatible mixed shard.

## Data Flow

### Standard Dump

For standard `Dump`, `MemoryStore` behaves like `CacheStore` from the caller's
perspective:

1. Copy the full shard payload into memory.
2. Mark the full `(block_id, layer_id)` shard ready.
3. Populate token-layer ready bits for tokens covered by the full shard when
   token geometry is configured.
4. Submit a full-shard `Dump` to the downstream backend.
5. `Wait` reports failure if local copy, backend submit, or backend wait fails.

### Standard Load

For standard `Load`:

1. If the full `(block_id, layer_id)` shard is ready in memory, scatter it to
   destination addresses.
2. Otherwise, submit a full-shard `Load` to the downstream backend.
3. After backend load completes, copy the full shard into memory, mark the full
   shard ready, populate token-layer ready bits, and scatter to destination
   addresses.

### Token-Layer Dump

For token-layer `Dump`, one call may contain many token-layer items:

1. Use each item's `tensor_type` to select the token-layer tensor size list.
2. Copy each requested `(block_id, layer_id, token_offset, tensor_type)` payload
   into the corresponding token chunk.
3. Mark each token-layer item ready.
4. If all required token/type items for a `(block_id, layer_id)` are ready, assemble the complete
   layer shard and submit a normal full-shard `Dump` to the downstream backend.
5. If only part of a shard is ready, do not write to the backend.

The first version does not aggregate input tokens by `(block_id, layer_id)` as an
optimization. It may still maintain enough metadata to detect when a full shard
has become ready.

### Token-Layer Load

For token-layer `Load`, one call may contain many token-layer items:

1. Use each item's `tensor_type` to select the token-layer tensor size list.
2. If a requested token-layer item is ready in memory, scatter it directly.
3. If it is missing, submit a full-shard `Load` for `(block_id, layer_id)` to
   the downstream backend.
4. When the full shard is loaded, populate memory token chunks for the shard,
   mark the full shard and token/type ready bits, and scatter requested token
   payloads to destination addresses.

This keeps backend interaction compatible with `CacheStore`, even though the
public token-layer API is more granular.

## Eviction

Eviction is unified. There is one physical memory pool and one LRU. The two
views do not maintain independent eviction policies.

When a token chunk is evicted:

1. Clear all token ready bits owned by the chunk.
2. Identify affected `(block_id, layer_id)` entries.
3. Mark affected full-shard state as not ready.

This prevents the standard view from reporting a full-shard hit after one of its
token chunks has been evicted.

## New APIs

Add default unsupported methods to `StoreV1` so existing stores do not need to
implement token-layer support immediately.

The token-layer lookup descriptor is a batch descriptor. It contains parallel arrays:

- `block_ids`: original backend block ids, one per token-layer item.
- `layer_ids`: layer ids, one per token-layer item.
- `token_offsets`: token offsets inside the original block, one per item.
- `tensor_types`: token-layer tensor type ids, one per item.

Token-layer load and dump descriptors include the same arrays plus:

- `addrs`: tensor address arrays, one row per item.

The pipeline pybind layer exposes equivalent methods. Python
`UcmPipelineStore` exposes:

- `lookup_tokens_on_layer`: synchronous, returns per-item hit booleans.
- `load_tokens_on_layer`: asynchronous, returns the existing task type.
- `dump_tokens_on_layer`: asynchronous, returns the existing task type.

The load and dump methods reuse existing `wait` and `check`.

## Configuration

Add memory-store specific config with conservative defaults:

- `memory_token_chunk_size`: number of tokens per memory-managed chunk. This is
  an internal LRU granularity and is independent from the standard backend block
  size. It may be set to the derived tokens per block to make the memory layer
  manage one backend block/layer as one chunk.
- `memory_buffer_capacity_gb`: memory pool capacity in GiB.
- `memory_tensor_size_by_type`: mapping from token-layer tensor type id to the
  tensor size list used for that type.
- `memory_required_tensor_types`: tensor type ids that must be ready for every
  token before a `(block_id, layer_id)` can be assembled and dumped as a complete
  backend-compatible shard.
- `memory_full_shard_layout`: layout mapping from token-layer tensor types to
  byte positions inside the cache-store compatible mixed K/V full shard.

Reuse existing config where applicable:

- `store_backend`
- `unique_id`
- `device_id`
- `tensor_size`
- `tensor_size_list`
- `shard_size`
- `block_size`
- `waiting_queue_depth`
- `running_queue_depth`
- `timeout_ms`
- `cpu_affinity_cores`

`tokens_per_block` is not configured directly. `MemoryStore::Setup` derives it
from the standard full-shard size and the token-layer typed payload layout. Setup
fails with invalid config if the full-shard payload size is not divisible by the
combined per-token size of `memory_required_tensor_types`. `tensor_size` and
`tensor_size_list` remain standard-view settings; token-layer methods use
`memory_tensor_size_by_type`.

## Error Handling

Failure behavior should match `CacheStore`:

- Queue submission failure marks the task failed.
- Local copy failure marks the task failed.
- Backend submit failure marks the task failed.
- Backend wait failure marks the task failed.
- `Check` reports whether a task has finished.
- `Wait` returns the final success, error, or timeout status.

The downstream backend is not required to implement token-layer APIs because
`MemoryStore` talks to it only through standard full-shard `Load` and `Dump`.

## Testing

Add focused unit tests and pipeline tests:

- Token-layer dump followed by token-layer load returns identical data.
- Token-layer dump/load select payload size by `tensor_type`, not by the standard
  `tensor_size_list`.
- Token-layer load miss fetches a full shard from the backend and fills token
  state.
- Partial token-layer dump does not make the standard view report a full-shard
  hit.
- Full token coverage for a `(block_id, layer_id)` triggers a backend full-shard
  dump.
- Evicting a token chunk clears token ready state and invalidates affected
  full-shard state.
- Standard `Load`, `Dump`, `Lookup`, `LookupOnPrefix`, `Check`, and `Wait`
  remain compatible with cache-store behavior.
- Pipeline builders for `Memory|Empty` and `Memory|Posix` can stack and execute
  basic operations.

## Non-Goals For First Version

- Do not optimize by grouping requested tokens by `(block_id, layer_id)` during
  task execution.
- Do not change downstream backend key format.
- Do not require existing backend stores to implement token-layer APIs.
- Do not refactor `CacheStore` internals into shared components.
- Do not implement a separate LRU for the standard view and token-layer view.
