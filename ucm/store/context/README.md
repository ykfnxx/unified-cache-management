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
      running_queue_depth: 524288
      local_rank_size: 8
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
| `waiting_queue_depth` | 任务等待队列、MLA 批次表和后台释放队列容量，默认 8192 |
| `running_queue_depth` | 与 Cache 相同的 SPSC H2D 队列容量，默认 524288，至少 2，实际可容纳容量减 1 个条目 |
| `local_rank_size` | 与 Cache 相同的 shard 错序分片数，默认 8；按 device_id 偏移 |
| `timeout_ms` | Wait 超时阈值，默认 30000；返回前仍排空传输，保证地址生命周期；MLA 批次从首次提交起也使用此截止时间，超时会通知参与者 |

retention 只控制淘汰时的 Drop/Dump，不是驻留 TTL；没有后台到期删除或结束时刷写。SDMA direct 与 IO aggregation 互斥，二者不能与 GDR 同时启用，须使用相应设备运行时。

## TP、MLA 和 layerwise

Memory 按 shard 分区：layerwise 下同一层各 block 的 slot 连续排列，地址为 `(layer * slot_count + slot) * shard_size`。策略仍按整个 block 管理引用与淘汰，后端写回和回填使用相同布局，不增加配置项。所有 TP rank 需使用同一版本的 native 库。

H2D 按 CacheStore 的 rank 顺序和多 stream 方式逐 shard 提交；成功时最后一个 shard 直接触发 stream 同步与任务完成，准备失败时通过单独结束标记排空此前的传输。

支持单机 GQA/MLA TP，PP=CP=1，direct 或 layerwise。MLA 包括普通 DSA KV 的多个 tensor component 以及已注册的 MTP 层；不包含 sparse C8、CP 或 hybrid attention 适配。

- GQA：各 rank 独立 Memory 和策略；策略使用原始逻辑 hash，后端 ID 区分 TP rank。scheduler 对每个 rank 的 `Memory 或 Fake` 可用性取交集。
- MLA：rank 0 独占 Dump、淘汰和 Fake 回源入驻；其余 rank 通过共享条件变量等待 READY 或失败，再持有跨进程读引用执行 H2D。rank 0 本地 H2D 完成即返回，通过独立、有界的后台队列等待 reader，保留整个 Load 的 block 引用；H2D 线程可继续提交后续层；晚到的 reader 不会错过已经发布的数据。
- MLA 的一次 Load 必须由整个 TP 组共同提交，最多支持 64 ranks。相同 block/shard 描述的重复 Load 按各 rank 的提交顺序匹配，设备地址不参与匹配。每个 rank 的 Wait 只等待本地 H2D，owner 可先返回；所有 rank 仍须在批次期限内提交并完成。批次表有界，容量复用 `waiting_queue_depth`。
- 全部进程使用相同 DP 组 unique_id 和共享内存命名空间。connector 自动设置 `context_tp_size`、`context_tp_rank`、`share_buffer_enable`、设备和 KV 布局参数。

配置 8 GiB Memory 时，TP=4 GQA 占 32 GiB payload，MLA 占 8 GiB payload；另有 Fake 和策略元数据。MLA 的 `/dev/shm` 必须容纳共享 Memory。Fake 只在共享内存中存元数据。

与原有 Store 一样，Lookup 不是预留。Load 时容量不足、block 正在被其他任务占用或 MLA owner 未及时入驻，可能返回失败。单个 Load 任务的不同逻辑 block 数不能超过 Memory 容量；超过时在分配和淘汰前返回 NoSpace，并报告需要数和可容纳数。未实现跨容量的滑动窗口读取。

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

上述 ContextStats 项均不是累计 unique block 计数。GQA 汇总后端次数会包含各 rank；MLA 后端计数只来自 rank 0，H2D 字节数仍需按各 rank 求和。


### Prometheus 淘汰计数

以下三个 Counter 默认通过 UCM metrics 接入 vLLM 的 `/metrics`：

| 指标 | 口径 |
| --- | --- |
| `ucm:context_evict_blocks_total` | 成功释放 Memory 的逻辑 block 次数，等于 Dump + Drop |
| `ucm:context_dump_blocks_total` | 淘汰策略选择 Dump 并成功释放 Memory 的 block 次数；包括后端已有记录而跳过写入的情况，不包含设备写入 Memory |
| `ucm:context_drop_blocks_total` | 淘汰策略选择 Drop 并成功释放 Memory 的 block 次数 |

按完整 block 计数，不按 layer/shard 计数，不对历史 block ID 去重。失败且未释放的 victim 不计数。GQA 各 rank 分别计数；MLA 只有 rank 0 执行淘汰，不会因其他 rank 读取而重复计数。

默认启用 metrics。若使用自定义 `metrics_config_path` 或内联 `metrics_config`，需将这三个指标加入其 `counter` 列表，并启用 `consumers.vllm_connector: true`。定义见 [metrics 配置](../../../examples/metrics/metrics_configs.yaml)。修改后需重建 ContextStore 并更新 Python 配置。

例如查询最近 5 分钟的总淘汰次数：

```promql
sum(increase(ucm:context_evict_blocks_total[5m]))
```

多实验共用 Prometheus 时，加上实际的模型、engine 等标签过滤；按 rank 查看可用 `sum by (worker_rank) (...)`。原生采集接口返回区间增量，由现有 exporter 累加为 Prometheus Counter。


### 读流水线与耗时

Load 分为 prepare 和 H2D 两个线程，使用 CacheStore 的 SPSC 队列及消费循环，逐 shard 提交。shard 顺序与 Cache 一致：按 device_id 和 local_rank_size 错开，不按 block key 排序。准备阶段仍按 block 管理入驻，按指定 shard 顺序将就绪地址交给传输线程。准备中途失败也会排空已经提交的 DMA。每个 rank 的 Load/Wait 在本地 H2D 同步后完成；MLA owner 的后台线程等待其他 rank 结束后释放 block 引用，等待期间禁止淘汰，后台失败单独记录日志，不改写已返回的本地成功。

首次 Memory miss 时批量 Lookup 剩余 block，每最多 32 个 block 合并一次后端 Load/Wait，并预留 shard 描述容量；全 Memory 命中不做额外预扫描或后端 Lookup，直接进入传输队列。只有整个 block 就绪才发布 READY，失败保留原有 shard 并释放新分配槽位。

节点、驻留项与引用计数使用哈希查询，淘汰顺序仍由原有有序索引决定。传输条目数不超过 Memory slot 数时直接通过容量上限检查；只有可能超容量时才构造唯一 block 集合，避免逐层重复去重。策略索引合并入驻/淘汰产生的祖先增量，在选择 victim 或删除拓扑节点前刷新；ObserveRequest 保留逐 block 的访问顺序和时间，但同一 segment 的冷排序索引只刷新一次。

共享元数据删除时修复哈希探测链，不保留 tombstone。MLA READY、失败和退出会通知等待者；所有 worker 在初始化阶段完成共享 Memory 映射和设备注册，首次 Load 不再承担该开销。任意 rank 均可先创建共享 payload；文件锁仅保护容量检查和空间预分配，各 rank 独立注册，不等待 owner 元数据。

以下 Histogram 默认导出为 `ucm:<名称>`，单位毫秒。使用自定义 metrics 配置时需同步添加定义：

| 名称 | 口径 |
| --- | --- |
| `context_observe_duration_ms` | 实际更新状态的 ObserveRequest 耗时，含主锁等待；去重回调不记录 |
| `context_load_prepare_queue_wait_ms` | 提交到 prepare 开始 |
| `context_load_prepare_duration_ms` | prepare 开始到全部 block 交给传输队列；包含查找、淘汰、入驻、MLA READY 等待及队列反压 |
| `context_load_mla_ready_wait_ms` | reader 一个 Load 内等待 READY 的累计时间 |
| `context_load_mla_completion_wait_ms` | rank 0 本地 H2D 同步到完成线程确认 reader 结束的时间，含后台队列等待，不代表本地 Load/Wait 耗时 |
| `context_load_first_h2d_ms` | 提交到首次 H2D 成功提交；没有提交 H2D 的失败任务不产生此样本 |
| `context_load_h2d_sync_ms` | 最终 stream 同步等待，不等于完整 DMA 用时 |

各阶段可重叠，不能将它们直接相加作为端到端时间。CPU-simu 用例验证流水线重叠和引用生命周期，实际 NPU 吞吐仍需实测。

评估 TTFT 时，应按 connector 的实际顺序统计：ObserveRequest → 提交首层 → 等待本层 → 提交下一层 → 计算本层。关注同步提交耗时及未被计算掩盖的等待；首个 H2D 或单层 Load 加速比不等于 TTFT 加速比。单请求通常只有一层预取，MLA 完成队列主要解除已排队的其他任务被 reader 等待阻塞的问题。

升级后需重新构建并重启所有 TP 进程，不能混用旧版共享元数据映射。
