# OnEvictCache real KV offload

`OnEvictCache` is a single storage backend. It saves real KV in host shared
memory and uses UCM's device stream and task completion facilities for D2H/H2D.
It does not stack CacheStore, FakeStore, or a disk backend.

```yaml
ucm_connectors:
  - ucm_connector_name: "UcmPipelineStore"
    ucm_connector_config:
      store_pipeline: "OnEvictCache"
      fake_res_cap: 64
      on_evict_cache_policy: "radix_lru"
      on_evict_cache_dump_max_idle_s: 3600
```

`fake_res_cap` is a positive integer in GiB, matching UCM's existing capacity
configuration. The resident block limit is
`floor(fake_res_cap * 2^30 / block_size)`, using the worker's physical block size
(including UCM's shard alignment). This limit applies to each writing Store
instance, not the sum across TP workers. The connector supplies the block,
shard, tensor layout, device ID, and shared `unique_id`.

## KV lifetime

- A new Dump copies device tensors into one host allocation for the block.
  The transfer stream waits for the producer's prerequisite event first.
- Layerwise Dumps fill individual shards. Lookup reports a hit only after all
  shards have completed and the block has been admitted and published.
- Resident blocks count against `fake_res_cap`. Capacity pressure invokes the
  existing LRU or Radix leaf-LRU policy, with the current request paths protected.
- An eviction victim whose idle time reaches the configured threshold is
  dropped: its shared-memory name is removed and its storage is released after
  existing readers finish. New Lookup/Load calls miss.
- Other victims become Dumped. Their payload stays in the same allocation and
  stops counting against the resident limit. No disk I/O or second copy occurs.
- Resident and Dumped blocks both support H2D Load. Reading a Dumped block does
  not promote it. Repeated Dumps of an already published immutable block also
  do not promote it.

The actual host memory footprint includes Resident, Dumped, and incomplete
blocks; `fake_res_cap` is not a physical memory limit. The backend does not
simulate disk latency or garbage-collect Dumped blocks.

## Processes and completion

Payloads use POSIX shared memory under `/dev/shm`, with names scoped by UCM's
`unique_id` and block ID. A completed block is published by renaming its pending
shared-memory object. Scheduler instances query published names; local workers
can open the same payload for Load. This is local-host storage, not a cross-host
P/D service. The normal connector rank/hash ownership rules still apply.

Transfers run in submission order on one UCM stream. `Check` reports actual task
completion and `Wait` returns transfer/admission errors. A protected request
that cannot fit returns `NoSpace`. Its unpublished payload is not a hit; earlier
successful admissions in a batch are not rolled back.

Normal writer destruction drains outstanding transfers and unlinks its own
shared-memory objects. An abrupt process exit can leave objects behind; their
names start with `uc_on_evict_<unique_id>_`. A new engine must use a fresh
`unique_id`. Existing POSIX shared-memory size and host-registration limits apply.

The existing `on_evict_backend_write_*` counters now represent simulated disk
writes for this backend. Eviction/drop counters retain their existing meaning.
The legacy `OnEvictCache|Fake` trace replay remains a metadata-only simulator,
selected separately with `on_evict_cache_capacity_gb`.

## Validation

CPU tests built with `RUNTIME_ENVIRONMENT=simu` exercise actual payload copying,
layer completion, eviction/drop, recovery, process visibility, and task errors.
They do not establish Ascend/CUDA host-registration compatibility or measured
hardware offload performance.

Run the focused CPU suite from the repository root:

```bash
cmake -S . -B /tmp/ucm-real-build \
  -DRUNTIME_ENVIRONMENT=simu -DBUILD_UNIT_TESTS=ON
cmake --build /tmp/ucm-real-build --target ucmstore.test -j4
/tmp/ucm-real-build/ucm/store/test/ucmstore.test \
  --gtest_filter='UCOnEvictCacheStoreTest.*:OnEvictRealTest.*:OnEvictTransferTest.*'
```
