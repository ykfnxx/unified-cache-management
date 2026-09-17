# ContextStore 使用说明

ContextStore 是独立的本机 KV Store，将 DWPDSim 的 context_lru 思路用于真实 KV 数据。设备与主机之间的传输沿用 CacheStore 的多 stream、event 同步方式；策略索引和数据池由 ContextStore 自己管理。

当前用于简单对比实验：Memory 和模拟 SSD 都使用本机 DRAM，预设模拟 SSD 在实验期间不会写满。实现固定 retention、固定淘汰粒度，支持同机非 MLA 多 TP，要求 PP=CP=1。支持 standard direct/layerwise connector，不支持 hybrid attention、MLA 或跨机 TP。

## 数据流

```text
vLLM 完整请求前缀 → ObserveRequest → context_lru 索引

设备 KV ── Store::Dump / D2H ──→ Memory
                                  │
                             容量不足时淘汰
                                  │
                    ┌─────────────┴─────────────┐
                 策略 Drop                    策略 Dump
                    │                           │
              释放 Memory              无 SSD 副本时复制到模拟 SSD
                                                │
                                           释放 Memory

Memory 或模拟 SSD ── Store::Load / H2D ──→ 设备 KV
```

- **Store 的 Dump 是设备保存操作，只写 Memory。** 只有淘汰策略选择 Dump，才写模拟 SSD。
- 策略 Drop 只释放 Memory，不写 SSD，也不删除已有 SSD 副本。
- 已有 SSD 副本时，策略 Dump 跳过重复写入。
- SSD 命中直接加载到设备，不提升到 Memory。
- 一个 block 的全部 shard 完成后才发布可用状态；layerwise 模式下不能只写完一层就报命中。
- retention 用于淘汰时判断 Drop/Dump，不是保证驻留时长，也没有后台 TTL 清理或结束时自动刷写。

## 构建安装

在运行模型的 Python 环境中，从仓库根目录安装：

```bash
# 按实际设备选择 ascend / ascend-a3 / cuda。
export PLATFORM=ascend
python -m pip install -v -e . --no-build-isolation
```

Ascend 环境需先加载已有 CANN 环境配置。请使用已经能正常运行目标模型的 vLLM / vLLM-Ascend 环境；引擎适配步骤见[仓库 Ascend 启动说明](../../../docs/source/getting-started/quickstart_vllm_ascend.md)。

必须一起重建 native stores 和 pipeline Python 扩展。本分支增加了 StoreV1 接口，不能混用旧版二进制。CPU `simu` 构建仅用于测试，不能用于真实设备传输。

## 配置

完整示例：[examples/ucm_context_config.yaml](../../../examples/ucm_context_config.yaml)。

```yaml
use_layerwise: true
use_lite: false
enable_event_sync: true
ucm_connectors:
  - ucm_connector_name: UcmPipelineStore
    ucm_connector_config:
      store_pipeline: ContextStore
      context_memory_capacity_gb: 8
      context_simulated_ssd_capacity_gb: 32
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
| `context_memory_capacity_gb` | 每 rank 的 Memory 容量，单位 GiB |
| `context_simulated_ssd_capacity_gb` | 每 rank 的模拟 SSD 容量，单位 GiB，需覆盖实验写入量 |
| `context_alpha` | 冷候选扫描预算系数，预算为系数乘 Memory block 容量，取值 `(0, 1]` |
| `context_retention_ns` | 淘汰时，候选空闲时间严格大于此值才 Drop；单位 ns，示例为 60 秒；`null` / `-1` 表示始终选择 Dump |
| `context_max_eviction_blocks` | 一次策略选择最多淘汰的 block 数，正整数 |
| `cache_stream_number` | 传输 stream 数，范围 1–32 |
| `waiting_queue_depth` | 传输等待队列深度 |
| `timeout_ms` | Wait 超时阈值；超时后仍等待实际传输结束，避免设备地址提前复用 |

小容量实验可使用 `context_memory_capacity_bytes` 和 `context_simulated_ssd_capacity_bytes`；同一个池不要同时指定 bytes 与 gb。容量向下取整为完整 block，两个池均需至少容纳一个 block。

**上述容量都是每 rank 容量。** TP=4 配置 8 GiB Memory 和 32 GiB 模拟 SSD，将分配总计 32 GiB + 128 GiB 的主机 payload 空间，此外还有元数据和引擎自身占用。模拟 SSD 不实现回收，写满会返回 `NoSpace` 并保留 Memory victim，不会自动改成 Drop。

GDR、SDMA direct 和 IO aggregation 需要对应运行时支持。SDMA direct 与 aggregation 互斥，二者也不能与 GDR 同时开启。初次运行可保持示例中的关闭设置。

## 接入 vLLM

保留现有模型启动命令，在其中设置 TP 数并加入以下参数。将配置路径替换为本机绝对路径：

```bash
--tensor-parallel-size 4 \
--kv-transfer-config '{
  "kv_connector": "UCMConnector",
  "kv_connector_module_path": "ucm.integration.vllm.ucm_connector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "UCM_CONFIG_FILE": "/path/to/unified-cache-management/examples/ucm_context_config.yaml"
  }
}'
```

`UCM_CONFIG_FILE` 是 `kv_connector_extra_config` 中的字段，仅 export 同名环境变量不能替代上述配置。`use_layerwise: false` 可选择 direct 路径。

TP rank、设备编号和 block/shard/tensor 布局由 connector 自动提供，不需要在 YAML 手动填写。所有 TP worker 和 scheduler 必须能够访问同一个 POSIX shared-memory namespace。

## 多 TP 的行为

- 每个 TP rank 独立拥有 Memory、模拟 SSD 和 context_lru 索引，独立执行淘汰。
- connector 使用 DP 隔离的 `unique_id`，native store 再追加 `_tp<rank>`。
- 所有 rank 在上下文和 Load/Dump 中使用 scheduler 的逻辑 block ID，依靠命名空间区分各 rank 的 payload。
- scheduler 只映射各 rank 的可用性元数据，不映射 payload 或策略索引。
- 每个逻辑 block 只有在所有 rank 都有完整副本时才算命中；各 rank 的副本可以分别位于 Memory 或模拟 SSD。
- Lookup 不预留数据。查询后若发生淘汰导致 Load 缺失，仍通过 connector 原有失败重算路径处理。

正常退出会撤销可用性并删除 owner 的共享内存名称。硬崩溃后可能留下 `/dev/shm/ucm_context_<unique_id>_tp<rank>`，只在确认对应旧进程全部退出后清理该实验的残留对象。

## 实验指标

worker 的 pipeline store 提供：

```python
stats = store.context_stats()
```

可读取 `ssd_write_blocks`、`ssd_write_bytes`、`ssd_read_bytes`、`ssd_write_skipped_blocks`、`memory_blocks`、`ssd_blocks`、`eviction_decisions`、`decision_ns`、`queue_wait_ns`、`no_space` 和 `failed_tasks` 等统计。尚未发生的计数可能没有对应 key，读取时可使用 `stats.get(name, 0)`。

这是 worker Store 的统计接口，不是现成的 HTTP / Prometheus 指标。多 TP 实验需从各 worker 收集；字节数按 rank 求和，耗时需明确是累计工作时间还是请求端到端时间。`ssd_write_bytes` 包含 block 布局空间，设备传输字节数按实际 tensor payload 计数。

对比时固定总 Memory 容量、TP 数、工作负载、prefix caching 设置和传输选项，单独报告模拟 SSD 容量。模拟 SSD 的 DRAM 复制耗时不能解释为真实 SSD 延迟或带宽。

## CPU 验证

以下命令从仓库根目录执行。Python 测试环境需安装 pytest、numpy 和 wrapt：

```bash
cmake -S . -B /tmp/ucm-context-build \
  -DRUNTIME_ENVIRONMENT=simu -DCMAKE_BUILD_TYPE=Debug -DBUILD_UNIT_TESTS=ON
cmake --build /tmp/ucm-context-build --target contextstore.test ucmpipelinestore -j 6
ctest --test-dir /tmp/ucm-context-build \
  -R '^Context(Store|Index)Test\.' --output-on-failure
UCM_CONTEXT_BUILD=/tmp/ucm-context-build \
  python -m pytest --noconftest -o addopts='' -q \
  ucm/store/test/e2e/context_store_test.py
```

已验证 11 个 C++ 测试和 8 个 Python 测试，C++ ASan/UBSan 通过。测试包含真实 CPU 字节拷贝、独立进程 TP=2、部分 shard 就绪、不同 rank 的独立淘汰和跨 Memory/模拟 SSD 回读；connector 配置和 key 传播使用引擎替身验证。

尚未完成目标模型的真实设备多 TP 推理验证。自适应 retention / 淘汰粒度未实现。

详细设计见 [context_store_design.md](../../../docs/source/developer-guide/context_store_design.md)。
