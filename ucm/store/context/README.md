# ContextStore|Fake

独立的 ContextStore，传输代码从 CacheStore 复制后修改，不继承或调用 CacheStore 的管理类。radix/context_lru 管理完整 block；shard handle 管理填充所有权、READY 和传输引用。

## 数据流

- **保存**：Dump 队列等待设备事件，使用多 stream 做 D2H；同步后发布对应 shard READY，只留在 Memory，不写 Fake。
- **读取**：各 rank 按 Cache 的错序方式取得 shard handle。哪个 rank 取得未就绪 shard 的填充权，哪个 rank 提交 Fake.Load；传输线程 Wait、发布 READY，再做本 rank H2D。调度线程可继续提交后续 shard，不等待整个 block 回填。
- **命中**：只有全部 shard READY 才报告 Memory block 命中；否则查询 Fake。Fake 命中的内容进入 Memory，参与后续策略淘汰。layerwise 仅回填当前层，不预读全部层。
- **淘汰**：Memory 无空闲 shard 槽时唤醒策略线程。它按 context_lru 选择 victim，通过 block 的使用中 shard 计数预留 victim，随后执行 Drop 或 Dump；释放该 block 当前驻留的全部 shards。Dump 只在后端没有该 block 时写回所有 shard，成功后释放；失败保留 victim。Drop 直接释放。

所有 rank 独立完成 Load，无跨 rank 批次屏障或等待后来 reader 的 pin。晚到的 rank 若遇到已经淘汰的数据，会重新查询/回填；Lookup 本身不是预留。MLA 的设备保存与策略管理由 rank0 负责，但后端 Load 不固定在 rank0。

Fake 不保存或恢复 payload。Fake.Load 后 H2D 复制的是 slot 中的现有字节，与 Cache|Fake 一样仅用于模拟实验，不能用于验证模型输出正确性。没有模拟 SSD 数据池或 SSD 容量配置。

## 配置

完整示例：[通用配置](../../../examples/ucm_context_config.yaml)、[GLM-5.1 配置](../../../examples/ucm_context_glm51_config.yaml)。

```yaml
use_layerwise: true
use_lite: false
enable_event_sync: true
ucm_connectors:
  - ucm_connector_name: UcmPipelineStore
    ucm_connector_config:
      store_pipeline: ContextStore|Fake
      context_memory_capacity_gb: 64
      buffer_number: 1048576
      context_alpha: 0.01
      context_retention_ns: 60000000000
      context_max_eviction_blocks: 64
      cache_stream_number: 4
      waiting_queue_depth: 8192
      running_queue_depth: 524288
      timeout_ms: 30000
      use_gdr: false
      cache_sdma_direct: false
      cache_io_aggregation: false
```

| 参数 | 含义 |
|---|---|
| `context_memory_capacity_gb` | 必填 Memory 容量，GiB；MLA 每 TP 组一份，GQA 每 rank 一份 |
| `context_memory_capacity_bytes` | 小容量实验可用 bytes，与 gb 二选一，按 shard 向下取整；容量至少容纳一个完整 block |
| `buffer_number` | Fake 元数据条目数，默认 1048576，至少 1024；应覆盖实验后端 ID 总数 |
| `context_alpha` | 冷候选预算系数，默认 0.01，范围 `(0,1]` |
| `context_retention_ns` | 候选空闲时间严格超过此值才 Drop；默认 `-1`，`null` 同义，始终 Dump |
| `context_max_eviction_blocks` | 单次选择最多淘汰的 block 数，默认 64 |
| `cache_stream_number` | 每个方向的 stream 数，默认 4；SDMA direct 使用 1 |
| `waiting_queue_depth` | 每个方向的任务队列容量，默认 8192，至少 2 |
| `running_queue_depth` | Load 的 shard 传输队列容量，默认 524288，至少 2 |
| `local_rank_size` | connector 自动传入：MLA 为 vLLM 的 TP size，非 MLA 为 1；无需在 YAML 设置，手写值会被覆盖。绕过 connector 直接调用 native Store 时，未传入的默认值为 8 |
| `timeout_ms` | 等待阈值，默认 30000；返回前排空本任务已提交的后端写入和 DMA |

两种 SPSC 队列实际可用容量为配置值减 1。retention 只决定淘汰时的 Drop/Dump，不是驻留 TTL；没有定时清除或退出时全量刷写。SDMA direct、IO aggregation 互斥，也不能与 GDR 同开。各 rank 必须使用相同版本 native 库及相同的共享容量、张量布局配置。

支持同机 TP、MLA/GQA、layerwise、MTP。非 MLA 的后端 key 按 rank 隔离，scheduler 取各 rank 可用性的交集。MLA 共享 payload 和 shard 状态；每个 rank 独立映射和注册共享内存。共享文件初始化不等待 rank0 完成设备注册。

## 结构与生命周期

`context_store.cc` 提供 StoreV1 接口；`trans_manager`、`load_queue`、`dump_queue`、`copy_stream` 是 Context 目录内的 Cache 副本。`trans_buffer` 按 Cache 的形式通过 `(block, layer)` 查找独立 shard slot，维护填充状态和引用；block 元数据只记录已分配 shard 的关联，不预留未加载层的 payload；`context_index` 维护 radix 策略。

同一 block 的首个 shard 分配时登记一次入驻事件，策略线程批量消费。纯 Memory 命中的提交和 H2D 不访问 radix tree。各层按需分配独立 payload 槽，Load 提交完当前层后预分配下一层槽位，不发起下一层后端读取；预分配无空间时跳过。命中只获取 bucket 锁和 shard 锁，Handle 引用只计在 shard 上；新增 shard 通过共享游标取得未使用槽，或从 radix 已释放槽的链表领取；链表锁只保护 O(1) 领取/归还，不覆盖 shard 分配全过程。只有创建 block 元数据时使用 metadata 锁。block 仅记录使用中的 shard 数，在 shard 引用从 0→1、1→0 时更新，Handle 复制不更新它。淘汰只锁定一次 block bucket，并用 CAS 将零计数置为淘汰中，不逐层扫描锁；任意一层仍在传输都会阻止预留。一次容量压力触发后，策略线程持续回收到约 5% 的 shard 槽空闲，或本轮没有可淘汰候选；每释放一个 block 就唤醒分配者，不要求等整批结束。暂时被传输引用占用的候选等待引用释放后重试，不直接报告 NoSpace。这是压力触发的批量回收，没有增加定时扫描或新的配置项。

后端 Lookup/Dump/Wait 在 radix 锁外执行；写回期间已预留的 shard 保持稳定，新的 Observe 可以继续。

完整 SHM 映射使用与 Cache 相同的 MAP_POPULATE，启动时完成元数据初始化和各 worker 的 payload 注册。共享布局已更新，所有进程必须使用同一版本并一起重启。

connector 在新请求/恢复和有保存的分块 prefill 时更新前缀。纯 decode 不重复广播前缀或刷新 radix recency；已有请求引用保留到 get_finished，再显式释放。

失败的后端填充不会发布 READY；重试可重新取得填充权。空的失败分配可以回收，不计入策略 evict/dump/drop。已回源的部分层可在无引用时释放，其余层仍由后端提供；未保存完整、也没有后端副本的设备写入 block 不参与正常淘汰。

## Metrics

开启现有 UCM metrics 后导出：

- `ucm:context_evict_blocks_total`：策略成功淘汰的逻辑 block 次数。
- `ucm:context_dump_blocks_total`：选择 Dump 并释放的次数，含后端已有副本而跳过写入的情况。
- `ucm:context_drop_blocks_total`：选择 Drop 并释放的次数。

`evict = dump + drop`，不按历史 block ID 去重，不包含普通 D2H 保存；MLA 只在策略 owner 上计数。

传输指标使用 `context_transfer_*`，可与对应的 `cache_*` 比较。例如 `context_transfer_load_duration_ms`、`context_transfer_load_queue_wait_duration_ms`、`context_transfer_load_backend_submit_duration_ms`、`context_transfer_shard_backend_wait_ms`、`context_transfer_h2d_sync_ms`。`context_observe_duration_ms` 记录非重复 Observe 的耗时。完整列表见默认 metrics 配置。它们测量任务或 shard 阶段，不能把均值直接乘层数或 TP rank 数当作 TTFT。

`ContextStats()` 提供 Memory block 数/峰值、实际驻留的 `memory_shards` / `memory_bytes`、拓扑节点数、成功传输的 H2D/D2H payload 字节数、回源 block/shard 次数、后端写回 block/字节数、策略淘汰计数与决策耗时。回源计数归实际执行填充的 rank，MLA 不再假定全部出现在 rank0。
