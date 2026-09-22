# ContextStore|Fake 使用说明

ContextStore 独立维护 context_lru 策略和本机 Memory，设备传输沿用 CacheStore 的多 stream、event、D2H/H2D 方式。后端串接现有 FakeStore，只记录 block ID，不保存 KV 数据。用于与 `Cache|Fake` 做策略与传输实验。

## 数据流

| 操作 | 行为 |
| --- | --- |
| 设备 Dump | D2H 写入 Memory；全部 shard 完成后发布可用状态，不写 Fake |
| 淘汰 Drop | 释放 Memory，不写 Fake；已有 Fake 记录保留 |
| 淘汰 Dump | 查询 Fake；未命中则提交完整 block 的所有 shard 并等待完成，再释放 Memory；已命中则跳过重复后端 Dump |
| Memory 命中 Load | 从 Memory 做真实 H2D |
| Fake 命中 Load | 分配 Memory，必要时执行 context 淘汰；调用 Fake.Load/Wait 后发布整个 block，再做真实 H2D。此 block 进入策略索引，可再次命中或被淘汰 |
| 两层都未命中 | Lookup 返回 miss；直接 Load 返回 NotFound |

Fake.Load 不填充 Memory，因此 Fake 回源后 H2D 搬运的是该 slot 当前内容，不能保证 KV 正确。与 `Cache|Fake` 一样，这个部署不能用于验证模型输出正确性。后端入驻按完整逻辑 block 处理；layerwise H2D 仍只传输本次请求的层。

没有模拟 SSD 数据池、SSD 容量配置或 Memory→模拟 SSD 的 memcpy。Fake 的 `buffer_number` 是元数据记录数，不是 payload 容量；应覆盖实验的后端 ID 数，避免 Fake 循环替换记录影响命中率。

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
      context_memory_capacity_gb: 8
      buffer_number: 1048576
      context_alpha: 0.01
      context_retention_ns: 60000000000
      context_max_eviction_blocks: 64
      cache_stream_number: 4
      waiting_queue_depth: 8192
      timeout_ms: 30000
      use_gdr: false
      cache_sdma_direct: false
      cache_io_aggregation: false
```

| 参数 | 含义 |
| --- | --- |
| `context_memory_capacity_gb` | Memory 容量，GiB；GQA 每 TP rank 一份，MLA 每 TP 组一份 |
| `context_memory_capacity_bytes` | 小容量实验可使用 bytes；与 gb 二选一，向下取整为完整 block，至少一个 block |
| `buffer_number` | Fake 元数据容量，默认 1048576，至少 1024；同一 DP 组各进程必须一致，GQA 不同 TP rank 的同一逻辑 block 分别占记录 |
| `context_alpha` | 冷候选扫描预算系数，默认 0.01，范围 `(0, 1]`；预算为系数乘 Memory block 容量 |
| `context_retention_ns` | 淘汰候选空闲时间严格大于此值才 Drop；`null` / `-1` 表示始终 Dump，默认 -1 |
| `context_max_eviction_blocks` | 一次策略选择最多淘汰的 block 数，默认 64 |
| `cache_stream_number` | 传输 stream 数，默认 4，范围 1–32 |
| `waiting_queue_depth` | 每个传输等待队列深度，默认 8192 |
| `timeout_ms` | Wait 超时阈值，默认 30000；返回前仍排空传输，保证地址生命周期；MLA reader 等待 rank 0 入驻也使用此阈值 |

retention 只控制淘汰时的 Drop/Dump，不是驻留 TTL；没有后台到期删除或结束时刷写。SDMA direct 与 IO aggregation 互斥，二者不能与 GDR 同时启用，须使用相应设备运行时。

## TP、MLA 和 layerwise

支持单机 GQA/MLA TP，PP=CP=1，direct 或 layerwise。MLA 包括普通 DSA KV 的多个 tensor component 以及已注册的 MTP 层；不包含 sparse C8、CP 或 hybrid attention 适配。

- GQA：各 rank 独立 Memory 和策略；策略使用原始逻辑 hash，后端 ID 区分 TP rank。scheduler 对每个 rank 的 `Memory 或 Fake` 可用性取交集。
- MLA：rank 0 独占 Dump、淘汰和 Fake 回源入驻；其余 rank 等待共享 Memory 发布，再持有跨进程读引用执行 H2D。读引用释放前不能淘汰该 slot。
- 全部进程使用相同 DP 组 unique_id 和共享内存命名空间。connector 自动设置 `context_tp_size`、`context_tp_rank`、`share_buffer_enable`、设备和 KV 布局参数。

配置 8 GiB Memory 时，TP=4 GQA 占 32 GiB payload，MLA 占 8 GiB payload；另有 Fake 和策略元数据。MLA 的 `/dev/shm` 必须容纳共享 Memory。Fake 只在共享内存中存元数据。

与原有 Store 一样，Lookup 不是预留。Load 时容量不足、block 正在被其他任务占用或 MLA owner 未及时入驻，可能返回失败。单个任务需要同时持有的逻辑 block 不应超过 Memory 容量。

## 构建和启动

在已有可运行目标模型的 vLLM / vLLM-Ascend 环境中，加载设备环境后安装：

```bash
export PLATFORM=ascend  # 按设备选择 ascend / ascend-a3 / cuda
python -m pip install -v -e . --no-build-isolation
```

必须一起重建 ContextStore、FakeStore 和 pipeline 扩展。CPU simu 构建只用于测试，不能用于设备部署。

保留现有 TP/MTP 模型参数，添加：

```bash
--kv-transfer-config '{"kv_connector":"UCMConnector","kv_connector_module_path":"ucm.integration.vllm.ucm_connector","kv_role":"kv_both","kv_connector_extra_config":{"UCM_CONFIG_FILE":"/absolute/path/ucm_context_glm51_config.yaml"}}'
```

配置入口的环境变量/启动方式参照[仓库 Ascend 启动说明](../../../docs/source/getting-started/quickstart_vllm_ascend.md)。实验之间使用新的 unique_id；不支持进程热重启或硬崩溃后的读引用恢复。清理共享内存前必须停止该实验所有进程。

## 统计和直接调用

直接调用者先通过 `ObserveRequest(request_id, observation, timestamp_ns, ordered_block_ids)` 提交完整前缀；请求结束传空 IDs 退休上下文。block ID 为 16 字节，时间非递减。Load 回源不会更新访问时间，访问时间由 ObserveRequest 决定。

`context_stats()` / native `ContextStats()` 返回 worker 本地计数，未发生的项用 `stats.get(name, 0)`：

| 计数 | 口径 |
| --- | --- |
| `memory_blocks` / `memory_peak_blocks` | 当前 / 峰值驻留逻辑 block 数 |
| `d2h_bytes` / `h2d_bytes` | 成功任务实际提交的 tensor payload 字节数累计 |
| `backend_dump_blocks` / `backend_dump_bytes` | 淘汰时后端 Dump+Wait 成功的 block 次数 / 布局字节数；Fake 不实际写 payload |
| `backend_dump_skipped_blocks` | 淘汰 Dump 因 Fake 已存在而跳过的 block 次数 |
| `backend_load_blocks` / `backend_load_shards` | 后端 Load+Wait 后成功入驻的 block / shard 次数 |
| `memory_load_shards` / `backend_h2d_shards` | 成功 H2D 任务中已驻留 / 本次新入驻的 shard 次数；MLA reader 计入前者 |
| `drop_blocks` / `dump_blocks` / `evicted_blocks` | 策略 Drop / Dump / 总淘汰次数 |
| `decision_ns` / `queue_wait_ns` / `failed_tasks` / `no_space` | 策略耗时、队列等待、失败任务、无可用淘汰空间 |

以上均不是累计 unique block 计数，也不是已接入 Prometheus 的指标。GQA 汇总后端次数会包含各 rank；MLA 后端计数只来自 rank 0，H2D 字节数仍需按各 rank 求和。
