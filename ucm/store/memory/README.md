# MemoryStore 接口说明

MemoryStore 是一个带 host memory token chunk 缓存的 `StoreV1` 实现。它同时提供两类接口：

- 标准 shard 接口：`Lookup`、`LookupOnPrefix`、`Load`、`Dump`、`Check`、`Wait`，用于和 CacheStore 相同的 block/shard 粒度读写。
- token-layer 接口：`LookupTokens`、`LoadTokens`、`DumpTokens`，用于按 `(block_id, layer_id, token_offset, tensor_type)` 访问 per-token per-layer 的 K/V cache。

标准接口面向 CacheStore 兼容；token-layer 接口是 MemoryStore 独有能力，其他 `StoreV1` 默认返回 unsupported。

## 核心概念

### Block、Shard、Tensor、Token

- `block_id`：16 字节 block hash。
- `shard_index`：标准接口中的物理 shard index。
- `layer_id`：token-layer 接口中的逻辑模型层号。
- `token_offset`：一个 block 内的 token 偏移。
- `tensor_type`：逻辑 tensor 类型，例如约定 `0 = K cache`、`1 = V cache`。实际含义由调用方定义，但 `memory_required_tensor_types` 的顺序会影响布局映射。

### 运行模式

MemoryStore 根据 `device_id` 分成两类运行形态：

- `device_id >= 0`：worker 模式，启用本地 host memory 缓存和异步 H2D/D2H transfer。
- `device_id == -1`：scheduler 模式，不启用本地 buffer/transfer，只通过 `store_backend` 代理标准 `Lookup` / `LookupOnPrefix`。

`Load`、`Dump`、`LoadTokens`、`DumpTokens` 都需要 worker 模式；scheduler 模式下 transfer 未启用。

### 普通模式和 layerwise 模式

MemoryStore 通过 `physical_shard_number = block_size / shard_size` 判断布局：

- 普通模式：`physical_shard_number == 1`，一个 block 只有物理 shard 0。所有层的 K/V 都打包在这一个 shard 里。
- layerwise 模式：`physical_shard_number > 1`，每个物理 shard 表示一个逻辑层上的一个 tensor type。

如果有两个 tensor type `[0, 1]`，表示 K/V，则 layerwise 模式的物理 shard 映射为：

```text
K layer i -> shard_index = i
V layer i -> shard_index = num_layers + i
```

更一般地：

```text
physical_shard = rank(tensor_type) * logical_layer_number + layer_id
```

其中 `rank(tensor_type)` 是该 type 在 `memory_required_tensor_types` 中的位置。

## 配置字段

### C++ Config

`ucm/store/memory/cc/global_config.h` 中的 `UC::MemoryStore::Config`：

| 字段 | 说明 |
| --- | --- |
| `storeBackend` | 后端 `StoreV1*` 的地址。worker 模式可为空，表示只使用本地 memory；scheduler 模式必须非空。 |
| `uniqueId` | 实例标识，当前 MemoryStore 不依赖它做共享内存命名。 |
| `deviceId` | `>=0` 为 worker 模式；`-1` 为 scheduler 模式。 |
| `shardSize` | 一个物理 shard 的字节数。 |
| `blockSize` | 一个完整 block 的字节数，必须是 `shardSize` 的整数倍。 |
| `tensorSizeList` | 标准 `Load` / `Dump` 中一行地址对应的 tensor 字节大小列表。为空时默认 `{shardSize}`。 |
| `memoryTokenChunkSize` | 每个 host memory chunk 包含多少 token。 |
| `memoryBufferCapacity` | host memory token chunk 缓存容量上限，单位字节。 |
| `waitingQueueDepth` | transfer 等待队列深度。 |
| `runningQueueDepth` | transfer 运行队列深度。 |
| `timeoutMs` | task wrapper 等待超时时间。 |
| `requiredTensorTypes` | token-layer 需要管理的 tensor type 顺序，例如 `{0, 1}` 表示 K/V。 |
| `tensorSizesByType` | 每种 tensor type 的单 token payload 拆分大小，例如 `{0: {k_bytes}, 1: {v_bytes}}`。 |
| `tokensPerBlock` | 由 `Setup` 根据布局自动推导，调用方通常不需要设置。 |

### Python connector 配置名

`ucm/store/memory/connector.py` 使用 snake_case 配置：

| Python key | C++ 字段 |
| --- | --- |
| `store_backend` | `storeBackend` |
| `unique_id` | `uniqueId` |
| `device_id` | `deviceId` |
| `shard_size` | `shardSize` |
| `block_size` | `blockSize` |
| `tensor_size_list` | `tensorSizeList` |
| `memory_token_chunk_size` | `memoryTokenChunkSize` |
| `memory_buffer_capacity` | `memoryBufferCapacity` |
| `waiting_queue_depth` | `waitingQueueDepth` |
| `running_queue_depth` | `runningQueueDepth` |
| `timeout_ms` | `timeoutMs` |
| `memory_required_tensor_types` | `requiredTensorTypes` |
| `memory_tensor_sizes_by_type` | `tensorSizesByType` |
| `memory_tokens_per_block` | `tokensPerBlock` |

注意：MemoryStore connector 不会把 CacheStore 的 `tensor_size` 自动转换成 `tensor_size_list` 或 `memory_tensor_sizes_by_type`。使用 MemoryStore 时需要显式传入 memory 专属配置。

## 标准接口

### Lookup

C++:

```cpp
Expected<std::vector<uint8_t>> Lookup(const Detail::BlockId* blocks, size_t num);
```

Python:

```python
hits = store.lookup(block_ids)
```

功能：

- worker 模式先查本地 full-ready 状态。
- 本地 miss 且配置了 `storeBackend` 时，会向 backend 查询缺失 block，并把结果合并回原始顺序。
- scheduler 模式直接代理 backend `Lookup`。

返回值为与输入 block 顺序一致的 bool 数组。

### LookupOnPrefix

C++:

```cpp
Expected<ssize_t> LookupOnPrefix(const Detail::BlockId* blocks, size_t num);
```

Python:

```python
prefix = store.lookup_on_prefix(block_ids)
```

功能：

- 返回连续命中的最大下标。
- 如果第一个 block 未命中，返回 `-1`。
- worker 模式会合并本地和 backend 的 prefix 结果。
- scheduler 模式直接代理 backend。

### Prefetch

C++:

```cpp
void Prefetch(const Detail::BlockId* blocks, size_t num);
```

Python:

```python
store.prefetch(block_ids)
```

当前 MemoryStore 中该接口为空实现，不会触发实际预取。

### Load

C++:

```cpp
Expected<Detail::TaskHandle> Load(Detail::TaskDesc task);
Status Wait(Detail::TaskHandle task_id);
Expected<bool> Check(Detail::TaskHandle task_id);
```

Python tensor 版本：

```python
task = store.load(block_ids, shard_indexes, dst_tensors)
store.wait(task)
```

Python raw-address 版本：

```python
task = store.load_data(block_ids, shard_indexes, dst_addrs)
store.wait(task)
```

功能：

- 从 MemoryStore/backend 加载完整物理 shard 到调用方地址。
- 本地 full-ready 命中时，直接从 host memory 组装完整 shard，再 H2D 到目标地址。
- 本地 miss 且存在 backend 时，先向 backend 发起完整 shard `Load`，backend 数据回填本地 token chunk 后，再 H2D 到目标地址。

参数约束：

- `block_ids`、`shard_indexes`、`dst_addrs` 行数必须一致。
- `dst_addrs` 每行地址个数必须等于 `tensor_size_list` 长度。
- 每行按 `tensor_size_list` 顺序 scatter。

普通模式下，通常所有 block 的 `shard_indexes` 都是 `0`。

### Dump

C++:

```cpp
Expected<Detail::TaskHandle> Dump(Detail::TaskDesc task);
```

Python tensor 版本：

```python
task = store.dump(block_ids, shard_indexes, src_tensors)
store.wait(task)
```

Python raw-address 版本：

```python
task = store.dump_data(block_ids, shard_indexes, src_addrs, prerequisite_handle=0)
store.wait(task)
```

功能：

- 从调用方地址 D2H 读取完整物理 shard。
- 写入本地 token chunk 缓存，并标记该物理 shard full-ready。
- 如果存在 backend，则继续向 backend 发起完整 shard `Dump`。

`prerequisite_handle` 当前必须为 `0`。MemoryStore 的 dump queue 目前对非零 prerequisite handle 返回 unsupported。

## Token-layer 接口

token-layer 接口以 `(block_id, layer_id, token_offset, tensor_type)` 为最小访问单位，用于读取或写入单个 token 在单个逻辑层上的 K/V payload。

### LookupTokens

C++:

```cpp
Expected<std::vector<uint8_t>> LookupTokens(const Detail::TokenLayerTaskDesc& task);
```

Python:

```python
hits = store.lookup_tokens_on_layer(
    block_ids=[block_id],
    layer_ids=[layer_id],
    token_offsets=[token_offset],
    tensor_types=[tensor_type],
)
```

功能：

- 只检查本地 token chunk 是否 ready。
- 不访问 backend。
- 返回值与输入 token item 顺序一致。

### LoadTokens

C++:

```cpp
Expected<Detail::TaskHandle> LoadTokens(Detail::TokenLayerTaskDesc task);
```

Python:

```python
task = store.load_tokens_on_layer(
    block_ids=[block_id],
    layer_ids=[layer_id],
    token_offsets=[token_offset],
    tensor_types=[tensor_type],
    dst_addr=dst_addrs,
)
store.wait(task)
```

功能：

- 本地 token ready 时，直接把该 token payload H2D 到目标地址。
- 本地 token miss 且存在 backend 时，加载该 token 所在的物理 shard，回填本地 token chunk，然后抽取目标 token H2D。
- 普通模式下，任意逻辑层和 K/V type 都映射到物理 shard `0`，但会按逻辑层和 tensor type 解析 shard 内偏移。
- layerwise 模式下，会按 `physical_shard = rank(tensor_type) * logical_layer_number + layer_id` 选择 backend shard。

参数约束：

- `layer_id < logical_layer_number`。
- `token_offset < tokens_per_block`。
- `tensor_type` 必须在 `memory_required_tensor_types` 中。
- `dst_addr` 每行地址个数必须等于该 `tensor_type` 在 `memory_tensor_sizes_by_type` 中的 size 列表长度。

### DumpTokens

C++:

```cpp
Expected<Detail::TaskHandle> DumpTokens(Detail::TokenLayerTaskDesc task);
```

Python:

```python
task = store.dump_tokens_on_layer(
    block_ids=[block_id],
    layer_ids=[layer_id],
    token_offsets=[token_offset],
    tensor_types=[tensor_type],
    src_addr=src_addrs,
    prerequisite_handle=0,
)
store.wait(task)
```

功能：

- 从调用方地址 D2H 读取单个 token payload。
- 写入本地 token chunk，并标记该 token ready。
- 只有当该 token 所属的物理 shard 已经 full-ready 时，才组装完整 shard 并向 backend `Dump`。
- 因此不会向 backend 写入半个物理 shard。

`prerequisite_handle` 当前必须为 `0`；非零值会返回 unsupported。

## 布局示例

### 普通模式：2 层 K/V 打包在 shard 0

配置示例：

```python
config = {
    "device_id": 0,
    "store_backend": backend.cc_store(),
    "shard_size": 16,
    "block_size": 16,
    "tensor_size_list": [4, 4, 4, 4],
    "memory_required_tensor_types": [0, 1],
    "memory_tensor_sizes_by_type": {
        0: [2],
        1: [2],
    },
    "memory_token_chunk_size": 2,
}
```

含义：

- `block_size / shard_size == 1`，所以是普通模式。
- `tensor_size_list` 有 4 项，`required_tensor_types` 有 2 项，因此逻辑层数为 `4 / 2 = 2`。
- 每层每个 K token 是 2 字节，每层每个 V token 是 2 字节。
- `tokens_per_block = 16 / (2 layers * (2 K bytes + 2 V bytes)) = 2`。

标准 `Dump` / `Load` 的地址列顺序：

```text
[K_layer0, K_layer1, V_layer0, V_layer1]
```

完整 shard 内部字节布局：

```text
K_layer0_token0
K_layer0_token1
K_layer1_token0
K_layer1_token1
V_layer0_token0
V_layer0_token1
V_layer1_token0
V_layer1_token1
```

读取 layer 1 的 V token 1：

```python
task = store.load_tokens_on_layer(
    block_ids=[block_id],
    layer_ids=[1],
    token_offsets=[1],
    tensor_types=[1],
    dst_addr=np.array([[dst_ptr]], dtype=np.uint64),
)
store.wait(task)
```

### Layerwise 模式：K/V 分布在多个物理 shard

配置示例：

```python
config = {
    "device_id": 0,
    "store_backend": backend.cc_store(),
    "shard_size": 4,
    "block_size": 16,
    "tensor_size_list": [4],
    "memory_required_tensor_types": [0, 1],
    "memory_tensor_sizes_by_type": {
        0: [2],
        1: [2],
    },
    "memory_token_chunk_size": 2,
}
```

含义：

- `block_size / shard_size == 4`，所以是 layerwise 模式。
- 两个 tensor type `[K, V]`，因此逻辑层数为 `4 / 2 = 2`。
- 物理 shard 映射：

```text
shard 0 -> K layer 0
shard 1 -> K layer 1
shard 2 -> V layer 0
shard 3 -> V layer 1
```

读取 V layer 1 token 1 时，token-layer API 仍传逻辑参数：

```python
task = store.load_tokens_on_layer(
    block_ids=[block_id],
    layer_ids=[1],
    token_offsets=[1],
    tensor_types=[1],
    dst_addr=np.array([[dst_ptr]], dtype=np.uint64),
)
store.wait(task)
```

MemoryStore 内部会把它映射到物理 shard 3。

## Host memory LRU 缓存

MemoryStore 的本地缓存粒度不是完整 block，也不是完整 shard，而是 token chunk：

```text
(block_id, layer_id, chunk_id, tensor_type)
```

其中：

```text
chunk_id = token_offset / memory_token_chunk_size
chunk 内偏移 = token_offset % memory_token_chunk_size
```

每个 chunk 保存：

- `memory_token_chunk_size` 个 token 的 payload bytes。
- 对应 token ready bitmap。
- LRU 链表位置。

`memory_buffer_capacity` 会换算成最多可保存的 chunk 数。估算方式为：

```text
max_chunks = memory_buffer_capacity /
             (max_tensor_type_payload_size * memory_token_chunk_size)
```

达到上限后，MemoryStore 会按 LRU 淘汰 chunk，并清理该 chunk 所属物理 shard 的 full-ready 状态。

## 与 CacheStore 的关系

相同点：

- 标准 `Load` / `Dump` 都使用 `TaskDesc`，返回 `TaskHandle`，通过 `Wait` / `Check` 等待完成。
- 标准接口都以 block + shard 为单位和 backend 交互。
- 普通模式下，一个 block 的所有 K/V tensor 可以在一个 shard 中按地址列打包。
- layerwise 模式下，K/V layer 可以拆成多个物理 shard。

不同点：

- CacheStore 的本地缓存粒度是物理 shard buffer。
- MemoryStore 的本地缓存粒度是 token chunk，并用 token chunk 组装/拆分完整物理 shard。
- CacheStore 不实现 token-layer API，默认 unsupported。
- MemoryStore 的 `LookupTokens` 只查本地 token chunk，不 fallback backend。
- MemoryStore 当前不实现 CacheStore 的 shared-buffer watcher 语义。

## 常见错误

- `invalid layer`：`layer_id` 超过逻辑层数。普通模式下逻辑层数由 `tensor_size_list.size() / memory_required_tensor_types.size()` 推导。
- `invalid token offset`：`token_offset >= tokens_per_block`。
- `invalid tensor type`：`tensor_type` 不在 `memory_required_tensor_types` 中。
- `invalid addr number`：地址列数量与 `tensor_size_list` 或 `memory_tensor_sizes_by_type[tensor_type]` 不一致。
- `transfer is not enable`：使用了 `device_id == -1` 的 scheduler 配置调用 transfer 接口。
- `unsupported`：传入了非零 `prerequisite_handle`，或通过非 MemoryStore 的 `StoreV1` 调用 token-layer API。

## 测试用例参考

MemoryStore 的接口和布局行为由以下测试覆盖：

- `UCMMemoryStoreLayoutCrossTest.StandardDumpThenTokenLoadUsesLayerMajorKvLayout`
- `UCMMemoryStoreLayoutCrossTest.OrdinaryStandardDumpThenTokenLoadUsesLogicalLayerAndTensorType`
- `UCMMemoryStoreLayoutCrossTest.LayerwiseStandardDumpThenTokenLoadMapsTensorTypeToShard`
- `UCMMemoryStoreLayoutCrossTest.TokenDumpThenStandardLoadAssemblesLayerMajorKvLayout`
- `UCMMemoryStoreCacheStoreContractTest.StandardDumpLoadPreservesOrdinaryMultiTensorLayout`
- `UCMMemoryStoreCacheStoreContractTest.StandardDumpLoadPreservesLayerwiseShardLayout`
