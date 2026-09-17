# ContextStore

`ContextStore` is an independent local KV store. Device saves enter the Memory pool.
Only a context_lru **Dump eviction** writes a block into the second host-memory
pool, which represents SSD. Drop evictions do not write; an existing SSD copy
suppresses another write. SSD hits load directly to the device without promotion.

The implementation includes fixed retention and fixed eviction size. Adaptive
retention/eviction parameters are not implemented. SSD capacity exhaustion returns
NoSpace and preserves the Memory victim. Both pools contain real KV bytes; their
latencies are host-memory transfer latencies, not SSD performance predictions.

## Configuration

Use [ucm_context_config.yaml](ucm_context_config.yaml) through the normal
`UCM_CONFIG_FILE` setting. The current vLLM integration supports direct/layerwise
non-MLA attention with single-host TP and PP=CP=1. Each TP worker owns its data
and policy. The scheduler lazily attaches to all rank lookup tables and intersects
availability before prefix/reverse lookup. Each rank may serve a block from either
Memory or simulated SSD. Payload and context indexes stay private to each worker.
DP groups are isolated. All TP processes must share the same POSIX shm namespace.

All ranks use the scheduler's canonical block IDs for context and I/O. The native
store appends `_tp<rank>` to the DP-scoped namespace. The connector supplies
`context_tp_size` and `context_tp_rank` (defaults for direct Store callers: 1 and 0).
Eviction is independent on each rank; ordinary load-miss/recompute handling still
covers eviction between lookup and load. MLA, PP, CP and multi-host TP are excluded.

Capacity can be given in integer GiB (`context_memory_capacity_gb`,
`context_simulated_ssd_capacity_gb`) or exact bytes (`*_capacity_bytes`) for small
experiments, but not both for the same pool. Capacity rounds down to full blocks.
These are **per-rank** capacities: TP=4 with 8 GiB Memory and 32 GiB simulated
SSD per rank uses 32 GiB + 128 GiB of host payload memory. Compare against the
baseline at the same total Memory capacity. The simulated SSD must be sized so it
does not fill during the experiment; there is no SSD reclamation policy. Its pool
must have at least one slot. Metadata is allocated separately.

`context_retention_ns: null` or `-1` means every eviction is Dump. A nonnegative
value selects Drop only when idle time is strictly greater than that value.
`context_alpha` is a floating-point value in `(0, 1]` and
`context_max_eviction_blocks` is a positive integer. No background TTL expiration
or end-of-run flush occurs.

The normal device/block/shard/tensor layout is supplied by the connector. Direct
Store callers must provide `unique_id`, `device_id`, `block_size`, `shard_size`
and `tensor_size_list` (or uniform `tensor_size`). The unique ID accepts letters,
digits, `_`, and `-`; only one worker may own that namespace at a time. Watchers
omit `device_id` (or set it to -1). A clean owner exit invalidates the table and
unlinks its name. A hard crash requires removing that namespace's stale
`/dev/shm/ucm_context_<unique_id>` only after its old processes have stopped.

Transfer options are copied from CacheStore: `cache_stream_number`, `use_gdr`,
`cache_sdma_direct`, `cache_io_aggregation`. SDMA direct and aggregation are
mutually exclusive and cannot be combined with GDR. They require the corresponding
runtime build. Both the Memory and simulated-SSD pools satisfy the selected
host-buffer registration requirements.

## Request context and completion

Before Load/Dump, the worker receives the complete prefix through:

```python
store.observe_request(request_id, observation, timestamp_ns, ordered_block_ids)
```

Block IDs are 16-byte values. Timestamps must be nondecreasing; duplicate/older
observations of the same live request are ignored. Live connector calls use a
monotonic clock. Trace replay supplies its own times. Each engine dispatch updates
request recency once, independently of the number of layers/streams.

After all intended submissions for a request, retire its context with:

```python
store.observe_request(request_id, 0, 0, [])
```

Queued transfers retain their topology until completion. Retirement removes only
request bookkeeping and unused topology; stored KV is not deleted. A block becomes
lookup-visible only after every required shard has finished. `Wait` drains actual
copies before returning, including on timeout. A Lookup hit is not a reservation;
Load may report NotFound if eviction intervenes, and the connector handles that as
a failed load.

`store.context_stats()` exposes completed D2H/H2D bytes, Memory/SSD load shards,
writeback/drop/duplicate-write counts, pool occupancy/peaks, task/queue time,
eviction decision time, NoSpace and failed tasks. SSD-write bytes include the
configured block layout; device-copy bytes include actual tensor payload only.

## CPU validation

A `simu` build uses CPU buffers as device memory but performs real byte copies.
It does not validate accelerator APIs or accelerator bandwidth.

```bash
cmake -S . -B /tmp/ucm-context-build \
  -DRUNTIME_ENVIRONMENT=simu -DCMAKE_BUILD_TYPE=Debug -DBUILD_UNIT_TESTS=ON
cmake --build /tmp/ucm-context-build --target contextstore.test ucmpipelinestore -j 6
ctest --test-dir /tmp/ucm-context-build -R '^Context(Store|Index)Test\.' --output-on-failure
# In a Python environment with pytest, numpy and wrapt:
UCM_CONTEXT_BUILD=/tmp/ucm-context-build python -m pytest --noconftest -o addopts='' \
  -q ucm/store/test/e2e/context_store_test.py
```

C++ tests cover byte-preserving SSD readback, on-evict-only writes, Drop,
duplicate-write suppression, partial block readiness, capacity pressure, request
retirement, process-separated Watcher access and index selection against an
independent full-tree oracle. Python tests exercise the native pipeline binding
including TP=2 in separate processes with distinct payloads, partial readiness,
independent Dump/Drop and mixed-tier loads, plus connector prefix/key propagation
with engine fixtures; they are not a vLLM model
execution test.

Build/install the UCM package with the desired accelerator runtime before using
it in vLLM. StoreV1 gains optional ObserveRequest/ContextStats methods: rebuild
native stores and the pipeline extension together; do not mix old and new binaries.
For a bandwidth baseline use Cache|Empty with a resident working set and the same
transfer options. Report the simulated SSD capacity separately for tiered-cache
experiments.
