# ContextStore|Fake

独立拷贝 `1686bef4` 的 CacheStore，保留它的 shard 哈希表、CLOCK 回收、引用计数、共享内存、队列、多 stream 和 layerwise 传输。不使用 radix tree，也不向 connector 传递额外的前缀元数据。

唯一的策略差异是 **on-evict write**：

- 普通保存只执行 D2H，数据留在 DRAM。
- 容量压力下按 Cache 的 CLOCK 规则回收单个无引用 shard。
- 距离 shard 最近一次记录的访问时间未超过 `context_retention_ns`：Dump；已从后端加载的干净 shard 无须重复写回。
- 严格超过阈值：Drop，不写后端。`-1` 或 `null` 表示始终 Dump。
- 写回成功后才复用槽位；写回失败保留原 shard，并向当前任务返回错误。写回在分配线程内同步完成，没有专门的淘汰线程。

时间阈值不决定 CLOCK 的候选顺序，也不触发定时淘汰。淘汰和时间戳都是 **shard 粒度**，不要求一个 block 的所有层一起淘汰。MLA 沿用 Cache：rank0 保存，各 rank 读取共享数据，任一分配线程均可回收 shard。

普通保存沿用 Cache 的完成顺序：D2H 同步完成后立即发布 READY，再将 buffer 引用交给后台完成队列释放，最后记录耗时指标。该队列不调用后端 Dump；后端写入仍只发生在淘汰时。每个 load/dump task 只读取一次单调时钟，该批 shard 复用这个访问时间，逐 shard 获取引用和提交传输的流水线保持不变。因此访问时间对应批次开始处理的时刻，与逐 shard 采样相差批内处理耗时。释放 metadata 锁后通过原子 max（CAS）更新，避免并发 load/dump 的旧批次覆盖新时间；槽位复用会清空旧时间。

共享 MLA 场景下，vLLM connector 默认只让 TP 首 rank 更新时间戳；标准连续 rank 布局下，跨节点 TP 还让每台主机的 local rank0 更新本机共享 buffer。其他 rank 仍正常获取引用、读取数据，但不读时钟、不刷新访问时间。非 MLA 或非共享路径继续各自更新。此优化依赖共享 MLA 的各 rank 访问同一批 KV。

Fake 只保存命中元数据，不保存或恢复 KV payload；Fake.Load 不填充数据。其后的 H2D 会复制槽位现有字节。因此本配置用于与 Cache|Fake 对照性能，不能验证模型输出正确性。

## 配置

容量和时间阈值沿用 ContextStore 原有字段名，其余传输参数与 Cache 一致：

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
      context_retention_ns: 60000000000  # 60 秒
      cache_stream_number: 4
      cache_load_exclusive_buffer_number: 1024
      waiting_queue_depth: 8192
      running_queue_depth: 524288
      timeout_ms: 30000
      use_gdr: false
      cache_sdma_direct: false
      cache_io_aggregation: false
```

`context_memory_capacity_gb` 单位为 GiB；MLA 为 TP 组共享容量，非共享模式为每 rank 容量。`local_rank_size` 由 vLLM connector 提供。多 stream、SDMA direct、IO aggregation 的配置与 Cache 相同；SDMA direct 有效时使用一个 stream。

容量字段保持原名：`context_memory_capacity_gb`（GiB）与 `context_memory_capacity_bytes`（bytes）二选一，仍受 Cache 的最少 shard 槽位限制。无需改写成 Cache 的容量字段。

`context_update_access_time` 是每个 store 实例的更新开关。vLLM connector 自动按上述规则设置；显式配置 `true` 可恢复所有 rank 更新时间戳（仍然每批取时），用于对照测试。独立使用原生 store 时默认为 `true`；非共享 buffer 始终更新。`retention=-1` 时完全跳过访问时间采样和刷新。

`context_retention_ns` 保持不变；`context_alpha`、`context_max_eviction_blocks`、`context_tp_size`、`context_tp_rank` 已随旧 radix/block 管理移除，不再生效。没有模拟 SSD、radix、Observe 或 ContextStats 接口。

完整示例：[通用配置](../../../examples/ucm_context_config.yaml)、[GLM-5.1 配置](../../../examples/ucm_context_glm51_config.yaml)。

## Metrics

以下为累计计数，按执行回收的 worker/rank 上报：

- `ucm:context_evict_shards_total`：成功淘汰的 READY shard 次数。
- `ucm:context_dump_shards_total`：选择 Dump 的次数，包括已有后端副本而免写的 shard。
- `ucm:context_drop_shards_total`：选择 Drop 的次数。
- `ucm:context_writeback_shards_total`：实际成功写回后端的 shard 次数。

`evict = dump + drop`。不计空槽、失败填充、预分配的未就绪槽，不按历史 ID 去重。原 `*_blocks_total` 淘汰指标已删除，不能沿用旧 Grafana 查询。

传输指标 `context_transfer_*` 对应 Cache 的 `cache_*`。例如 `context_transfer_load_queue_wait_duration_ms`、`context_transfer_shard_backend_wait_ms`、`context_transfer_h2d_sync_ms`；`posix_load_success_shards_total` 沿用 Cache 的名字，表示等待共享 shard 就绪后完成的读取，不等于实际后端 Load 次数。
