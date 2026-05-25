# Memory Store 设计文档

## 目标

在 pipeline store 模式下新增一个内存侧存储后端。新后端通过新增 API
提供 per-token-per-layer 粒度访问，同时保持现有 cache-store 兼容接口和后置
backend 的存储格式不变。

第一版实现保持简单：

- 新增独立 `MemoryStore` 后端，路径为 `ucm/store/memory`。
- 现有 `StoreV1` 方法保持与 `CacheStore` 兼容。
- 通过 `StoreV1`、pipeline pybind 和 Python `UcmPipelineStore` 增加
  token-layer 方法。token-layer lookup 像现有 lookup 一样同步返回；
  token-layer load 和 dump 像现有 load/dump 一样异步返回 task handle。
- 使用单一内存池和简单 LRU 策略。
- 内部使用 token chunk 管理，降低 metadata 和小对象分配开销。
- 与后置 backend 交互时使用和 `CacheStore` 相同的完整 shard 格式。

## 现有上下文

pipeline store 当前按反向数据流顺序 stack store。以 `Cache|Posix` 为例，
builder 会先 stack `Posix`，再 stack `Cache`。外部公开调用落到栈顶 store，
栈顶 store 再通过 `store_backend` 指针与后置 backend 交互。

`CacheStore` 暴露标准 `StoreV1` 方法：

- `Lookup`
- `LookupOnPrefix`
- `Prefetch`
- `Load`
- `Dump`
- `Check`
- `Wait`

`CacheStore` 按 `BlockId` 和 `shard_index` 存储、传输数据。在 layerwise 模式下，
`shard_index` 就是 layer id，后置 backend 看到的是完整
`Shard{owner = block_id, index = layer_id}` payload。

新的 `MemoryStore` 必须保持这个后置 backend 格式。

## 当前实现与 CacheStore 的差异

下面的对比基于当前代码，而不是只基于最初方案。对应实现主要在：

- `ucm/store/cache/cc/cache_store.cc`
- `ucm/store/cache/cc/buffer_manager.h`
- `ucm/store/cache/cc/load_queue.cc`
- `ucm/store/cache/cc/dump_queue.cc`
- `ucm/store/memory/cc/memory_store.cc`
- `ucm/store/memory/cc/trans_buffer.cc`
- `ucm/store/memory/cc/load_queue.cc`
- `ucm/store/memory/cc/dump_queue.cc`

| 维度 | CacheStore 当前实现 | MemoryStore 当前实现 |
| --- | --- | --- |
| API 面 | 只实现标准 `StoreV1` 接口。token-layer 方法仍然走 `StoreV1` 默认 unsupported。 | 在标准接口之外，额外实现 `LookupTokens`、`LoadTokens`、`DumpTokens`。 |
| 内部数据粒度 | 以内存 buffer slot 保存完整 `(block_id, shard_index)` shard。 | 以 `(block_id, layer_id, chunk_id, tensor_type)` 为 key 保存 token chunk，并额外维护 `(block_id, layer_id)` 的 `fullReady` 集合。 |
| Lookup 路径 | `BufferManager` 先查本地 buffer；miss 时继续查 backend，`cacheLoadBackendOnly` 还可以显式绕过本地命中。 | `Lookup` / `LookupOnPrefix` / `LookupTokens` 都只查本地内存状态，不访问 backend。backend miss 补齐只发生在 `Load` / `LoadTokens`。 |
| 传输模型 | 明确区分 host/device 传输，依赖 `device_id`、copy stream、可选 GDR、dispatch 线程和 transfer/backend 线程。 | 复用共享的 `trans/copy_stream.h` helper，并把传输拆成 staged queue。load 走 dispatch + transfer 两阶段，dump 走 dispatch + transfer + backend-host 三阶段。用户地址与 memory buffer 之间的数据搬运仍然通过 `Trans::Stream` 的 `HostToDeviceAsync` / `DeviceToHostAsync` 完成。 |
| backend 交互 | backend load/dump 以 shard 为单位异步提交，`Wait` 发生在 queue stage。 | token miss 会退化成整 shard 的 backend `Load`，并在 `TransBuffer` 内同步 `Wait`；full shard dump 也是在 `TransBuffer` 内同步 `Wait`。 |
| 配置面 | 依赖 `device_id`、`share_buffer_enable`、`cache_buffer_capacity_gb`、`running_queue_depth`、`stream_number`、`use_gdr`、`cpu_affinity_cores` 等。 | 当前使用 `device_id`、`cache_stream_number` / `memory_stream_number`、`use_gdr`、`cpu_affinity_cores`、`shard_size`、`block_size`、`tensor_size(_list)`、`memory_token_chunk_size`、`memory_buffer_capacity_gb`、`memory_required_tensor_types`、`memory_tensor_size_by_type_*`、`waiting_queue_depth`、`running_queue_depth`、`timeout_ms`。 |

### 当前实现上的具体说明

1. `MemoryStore` 已经不是 `CacheStore` 的“轻微变体”，而是独立的数据模型。
   `CacheStore` 的核心对象是完整 shard buffer；`MemoryStore` 的核心对象是 typed token
   chunk，再从 chunk 派生 full-shard ready 状态。

2. `MemoryStore` 的标准 `Lookup` 语义目前比 `CacheStore` 更“本地化”。
   `CacheStore::Lookup` 会在本地 miss 后继续问 backend；`MemoryStore::Lookup`
   只看本地 `fullReady_`。因此标准 `Lookup` 在两个 store 间并不完全等价。

3. `MemoryStore` 现在在“谁负责传输”和“queue 怎么分阶段”这两件事上更接近 `CacheStore`，
   但 backend 执行流仍然不相同。
   当前 load/dump queue 负责通过共享的 `Trans::CopyStream` helper 和 `Trans::Stream`
   处理调用方地址与设备/host buffer 之间的搬运；`TransBuffer` 退回到 host 侧，只负责
   chunk/full-shard 拼装与 backend 交互。
   `CacheStore` 仍然把 backend wait 放在线程队列阶段，围绕异步 shard transfer 工作；
   `MemoryStore` 则在 `LoadFullFromBackend` / `DumpFullToBackend` 中直接同步等待 backend
   完成，然后再继续 token/full-shard 转换。

4. `MemoryStore` 当前没有实现设计文档里提到的显式 `memory_full_shard_layout`
   配置。现有 `SplitFullShard` / `AssembleFullShard` 直接假设 full shard 的字节布局是：
   按 token 顺序遍历，每个 token 内再按 `memory_required_tensor_types` 顺序拼接。
   这意味着它并不是像 `CacheStore` 一样对任意 shard layout 配置开放。

5. `MemoryStore` 的并发模型仍然比 `CacheStore` 更保守，但已经不是单阶段 worker。
   当前 load 已拆成 dispatch + transfer 两阶段，dump 已拆成 dispatch + transfer +
   backend-host 三阶段；不过底层 `TransBuffer` 仍由单 mutex 串行保护。`CacheStore`
   还依赖 buffer handle 和 queue 层异步 backend wait，因此两者并没有完全收敛。

6. `MemoryStore` 的淘汰单位是 token chunk，而 `CacheStore` 的淘汰单位是 shard buffer slot。
   这意味着 `MemoryStore` 在命中、驱逐和 ready 判定上都天然带有 token/type 粒度，
   即使它对外还保留了标准 shard 视图。

这部分对比应该作为后续继续演进 `MemoryStore` 时的基线：当前实现已经保证了
backend full-shard 兼容，调用方数据搬运已经收敛到 queue 持有的 `Trans::Stream`
流程里，`TransBuffer` 也不再保留整个 `Config`；但它仍没有在 backend wait 流程、
并发拆分或 lookup 语义上完全复制 `CacheStore`。

## 架构

新增一个共享库：

- 源码路径：`ucm/store/memory`
- 导出 factory：`MakeMemoryStore`
- pipeline 注册：
  - `Memory|Empty`
  - `Memory|Posix`

这个 store 在同一个物理内存池上提供两个公开视图：

1. 标准 store 视图：
   现有 `StoreV1` API 保持 `CacheStore` 语义。标准 load/dump 操作的是完整
   `(block_id, shard_index)` payload。

2. Token-layer 视图：
   新增 API 操作一批 token-layer items。每个 item 由
   `(block_id, layer_id, token_offset)` 标识，并拥有自己的 tensor 地址数组。
   token-layer load/dump 为整个 batch 返回一个普通 task handle。

物理内存池由一个 LRU 管理。LRU 淘汰单位是 token chunk，而不是单个 token。
一个 token chunk 表示某一层上的一小段固定 token range。chunk size 可配置。

## 存储语义

store 必须区分三层语义。

### 物理数据

物理内存存储 token chunk。一个 token chunk 包含固定 token range 的 token-layer
payload。

标准 store 视图继续使用现有 `tensor_size` 或 `tensor_size_list` 规则描述完整
shard payload。token-layer 视图不复用这些设置。token-layer payload 的布局由
每个 token-layer item 调用时传入的 tensor type 决定。每个 tensor type 都有独立
配置的 tensor size list，用于定义该类型如何从地址数组 gather 成一个逻辑
token-layer payload，以及如何 scatter 回地址数组。

第一版 tensor type 至少需要支持 K cache 和 V cache 分离。K 和 V 在 token-layer
视图中独立存储，因此调用方可以只 lookup/load/dump K，也可以只处理 V。

### Token-Layer 状态

token-layer 状态精确到：

```text
(block_id, layer_id, token_offset, tensor_type)
```

token-layer dump 只会把对应的 token-layer item 标记为 ready。token-layer load
只有在精确 item ready 时才算命中。

### Full-Shard 状态

标准 store 状态精确到：

```text
(block_id, layer_id)
```

只有完整 layer shard 才能让标准视图 ready。部分 token 写入不能让标准
`Lookup`、`LookupOnPrefix` 或 `Load` 表现得像完整 shard 已存在。

这个规则保证标准 API 与 `CacheStore` 兼容。

标准 full-shard 视图由 typed token chunks 派生，不要求在内存里额外保存一份完整
KV 混合 shard。只有某个 `(block_id, layer_id)` 下所有 token、所有
`memory_required_tensor_types` 都在统一 chunk pool 中 ready 时，该 full shard
才 ready。

## 后置 Backend 格式

后置 backend 永远只接收 cache-store 兼容 shard：

```text
Shard{owner = block_id, index = layer_id, addrs = {full_shard_buffer}}
```

backend 绝不接收 token 粒度 key，也不接收编码后的 token offset。这样不会改变
`PosixStore`、`Ds3fsStore`、`EmptyStore` 或其他现有后置 store 的存储格式。

backend shard 仍然是 K/V 混合的完整 full-shard payload。`MemoryStore` 在边界处做转换：

- 标准或 backend full-shard load：将混合 K/V shard 拆分成 typed token chunks。
- 标准或 backend full-shard dump：从 typed token chunks 组装混合 K/V shard。

这个转换需要配置 full-shard layout，用于描述每个 tensor type 在 cache-store
兼容混合 shard 中的字节位置。

## 数据流

### 标准 Dump

对标准 `Dump`，`MemoryStore` 从调用方视角表现得像 `CacheStore`：

1. 将完整 shard payload copy 到内存。
2. 将完整 `(block_id, layer_id)` shard 标记为 ready。
3. 如果配置了 token geometry，则填充这个完整 shard 覆盖的 token-layer ready bit。
4. 向后置 backend 提交完整 shard `Dump`。
5. 如果本地 copy、backend submit 或 backend wait 失败，`Wait` 返回失败。

### 标准 Load

对标准 `Load`：

1. 如果完整 `(block_id, layer_id)` shard 在内存中 ready，直接 scatter 到目标地址。
2. 否则，向后置 backend 提交完整 shard `Load`。
3. backend load 完成后，将完整 shard copy 到内存，标记 full shard ready，
   填充 token-layer ready bit，并 scatter 到目标地址。

### Token-Layer Dump

对 token-layer `Dump`，一次调用可以包含多个 token-layer items：

1. 使用每个 item 的 `tensor_type` 选择 token-layer tensor size list。
2. 将每个请求的 `(block_id, layer_id, token_offset, tensor_type)` payload copy
   到对应 token chunk。
3. 将每个 token-layer item 标记为 ready。
4. 如果某个 `(block_id, layer_id)` 的所有必需 token/type item 都 ready，则组装完整 layer shard，
   并向后置 backend 提交普通完整 shard `Dump`。
5. 如果只有部分 token ready，不写后置 backend。

第一版不把输入 token 按 `(block_id, layer_id)` 聚合成优化路径。实现仍然需要维护
足够的 metadata，用于判断某个 full shard 是否已经 ready。

### Token-Layer Load

对 token-layer `Load`，一次调用可以包含多个 token-layer items：

1. 使用每个 item 的 `tensor_type` 选择 token-layer tensor size list。
2. 如果请求的 token-layer item 在内存中 ready，直接 scatter。
3. 如果 miss，则向后置 backend 提交该 `(block_id, layer_id)` 的完整 shard `Load`。
4. 完整 shard load 完成后，将 shard 填充到内存 token chunks，标记 full shard
   和 token/type ready bit，再把请求的 token payload scatter 到目标地址。

这样即使公开 token-layer API 粒度更细，backend 交互仍然与 `CacheStore` 兼容。

## 淘汰

淘汰统一处理。系统只有一个物理内存池和一个 LRU。标准视图和 token-layer 视图
不维护各自独立的淘汰策略。

当一个 token chunk 被淘汰时：

1. 清除该 chunk 内所有 token ready bit。
2. 找到受影响的 `(block_id, layer_id)` entry。
3. 将受影响的 full-shard 状态标记为 not ready。

这可以防止某个 token chunk 已被淘汰后，标准视图仍然报告 full-shard hit。

## 新增 API

在 `StoreV1` 中增加默认 unsupported 的方法，使现有 store 不需要立刻实现
token-layer 支持。

token-layer lookup descriptor 是一个 batch descriptor，包含并行数组：

- `block_ids`：原始后置 backend block ids，每个 token-layer item 一个。
- `layer_ids`：layer ids，每个 token-layer item 一个。
- `token_offsets`：token 在原始 block 内的 offset，每个 item 一个。
- `tensor_types`：token-layer tensor type ids，每个 item 一个。

token-layer load 和 dump descriptor 包含同样的数组，另外包含：

- `addrs`：tensor 地址数组，每个 item 一行。

pipeline pybind 层暴露等价方法。Python `UcmPipelineStore` 暴露：

- `lookup_tokens_on_layer`：同步返回每个 item 的命中 bool。
- `load_tokens_on_layer`：异步返回现有 task 类型。
- `dump_tokens_on_layer`：异步返回现有 task 类型。

load 和 dump 方法复用现有 `wait` 和 `check`。

## 配置

新增 memory-store 专属配置，并使用保守默认值：

- `memory_token_chunk_size`：每个内存管理 chunk 中的 token 数量。这是内部 LRU
  粒度，独立于标准 backend block size。它也可以设置为推导出的 tokens per block，
  让内存层退化为按一个 backend block/layer 一个 chunk 管理。
- `memory_buffer_capacity_gb`：内存池容量，单位 GiB。
- `memory_tensor_size_by_type`：从 token-layer tensor type id 到该类型 tensor
  size list 的映射。
- `memory_required_tensor_types`：每个 token 都必须 ready 的 tensor type ids。
  只有这些类型都 ready 后，某个 `(block_id, layer_id)` 才能组装并 dump 成
  backend 兼容的完整 shard。
- `memory_full_shard_layout`：从 token-layer tensor types 到 cache-store 兼容
  K/V 混合 full shard 内字节位置的 layout 映射。

复用已有配置：

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

`tokens_per_block` 不直接配置。`MemoryStore::Setup` 会根据标准 full-shard size
和 token-layer typed payload layout 推导它。如果 full-shard payload size 不能被
`memory_required_tensor_types` 的单 token 总 payload size 整除，Setup 返回
invalid config。`tensor_size` 和 `tensor_size_list` 仍然是标准视图配置；
token-layer 方法使用 `memory_tensor_size_by_type`。

## 错误处理

失败行为与 `CacheStore` 保持一致：

- 队列提交失败会将 task 标记为失败。
- 本地 copy 失败会将 task 标记为失败。
- backend submit 失败会将 task 标记为失败。
- backend wait 失败会将 task 标记为失败。
- `Check` 只报告 task 是否结束。
- `Wait` 返回最终 success、error 或 timeout 状态。

后置 backend 不需要实现 token-layer API，因为 `MemoryStore` 只通过标准完整 shard
`Load` 和 `Dump` 与其交互。

## 测试

添加聚焦的单测和 pipeline 测试：

- token-layer dump 后 token-layer load 能读回相同数据。
- token-layer dump/load 根据 `tensor_type` 选择 payload size，而不是使用标准
  `tensor_size_list`。
- token-layer load miss 时从 backend 拉取完整 shard，并填充 token 状态。
- 部分 token-layer dump 不会让标准视图误报 full-shard hit。
- 某个 `(block_id, layer_id)` 的 token 全部 ready 后，会触发 backend 完整 shard dump。
- 淘汰 token chunk 后，会清理 token ready 状态并让受影响的 full-shard 状态失效。
- 标准 `Load`、`Dump`、`Lookup`、`LookupOnPrefix`、`Check` 和 `Wait` 继续保持
  cache-store 兼容行为。
- `Memory|Empty` 和 `Memory|Posix` pipeline builder 可以正常 stack 并执行基本操作。

## 第一版非目标

- 不在任务执行阶段按 `(block_id, layer_id)` 聚合请求 token 做优化。
- 不改变后置 backend key 格式。
- 不要求现有 backend store 实现 token-layer API。
- 不把 `CacheStore` 内部重构成共享组件。
- 不为标准视图和 token-layer 视图实现独立 LRU。
