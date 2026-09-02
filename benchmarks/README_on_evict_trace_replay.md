# OnEvict Trace Replay 使用说明

`on_evict_trace_replay.py` 直接回放 `{timestamp, hash_ids}` JSONL，驱动真实的 C++
`OnEvictCacheStore | FakeStore`。它不启动 vLLM，不执行模型推理，也不需要 GPU/NPU。

## 1. 本机构建

在 UCM 仓库根目录执行：

```bash
sudo apt-get install -y cmake g++ python3-dev python3-wrapt

cmake -S . -B /tmp/ucm-replay-build \
  -DRUNTIME_ENVIRONMENT=simu \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/ucm-replay-build -j8
cmake --install /tmp/ucm-replay-build \
  --prefix "$PWD" \
  --component ucm
```

安装步骤会把 `ucmpipelinestore`、`libonevictcachestore.so`、`libfakestore.so` 和
metrics 库放到源码目录下对应的 `ucm/` 包中。已经完成过本机构建时可以跳过本节。

`simu` 只选择 CPU 模拟设备运行时；本工具的 OnEvict、Radix Tree、Fake backend 和
metrics 都运行真实 C++ 实现。

## 2. Trace 格式

输入文件为 JSONL，一行一个请求：

```json
{"timestamp": 0, "hash_ids": [1, 2]}
{"timestamp": 500, "hash_ids": [1, 3]}
{"timestamp": 2000, "hash_ids": [4]}
```

要求：

- `timestamp` 非负且按文件顺序非递减；
- `hash_ids` 是整数或字符串数组；
- `hash_ids` 必须是一次 prefill 请求完整、有序的 block 路径，而不是新增后缀；
- timestamp 相同的请求按文件顺序执行；
- 其他 Mooncake trace 字段会被忽略；
- 不根据 `output_length` 生成 decode block。

每个 hash ID 使用 `BLAKE2b-128` 稳定映射成 UCM 的 16-byte `BlockId`。工具不会重新
tokenize 或改变原 trace 的 block 粒度。

## 3. 运行

从仓库根目录执行：

```bash
python3 benchmarks/on_evict_trace_replay.py \
  --trace-path /path/to/trace.jsonl \
  --timestamp-unit ms \
  --capacity-gb 64 \
  --block-size-bytes 56000000 \
  --dump-max-idle-s 3600 \
  --output /tmp/ucm-replay-report.json
```

参数：

| 参数 | 含义 |
| --- | --- |
| `--trace-path` | 输入 JSONL 路径 |
| `--timestamp-unit` | trace timestamp 单位，只支持 `s` 或 `ms` |
| `--capacity-gb` | OnEvict 逻辑容量，单位 GiB |
| `--block-size-bytes` | 一个 hash ID 代表的逻辑 KV block 字节数 |
| `--dump-max-idle-s` | 淘汰时允许写后端的最大空闲时间；默认 `0`，表示关闭超时丢弃 |
| `--output` | 可选，保存完整 JSON 报告 |

实际容量为：

```text
capacity_blocks = floor(capacity_gb * 2^30 / block_size_bytes)
```

例如 `capacity-gb=1`、`block-size-bytes=536870912` 时，OnEvict 容量为 2 blocks。

工具不会按照 timestamp 真实等待。第一条记录的时间作为零点，后续请求间隔转换成
逻辑纳秒传入 C++ Store，因此长时间 trace 仍按本机 CPU 处理速度运行。

## 4. 请求语义

每条请求依次执行：

```text
prefix lookup
  -> 首个 miss 后的完整后缀执行 admission
  -> 使用完整 hash_ids 更新 Radix Tree
  -> 容量不足时执行 leaf-LRU 路径淘汰
```

未超时的 victim 会 Dump 到 Fake backend，之后可以产生 backend hit。最近访问时间超过
`--dump-max-idle-s` 的 victim 会直接丢弃，不写 Fake backend。timeout 只在容量淘汰发生
时判断，不执行主动扫描。

## 5. 输出

终端输出示例：

```text
requests=3 blocks=5 prefix_hit_ratio=0.200000
eviction_paths=2 evicted_blocks=3 backend_writes=1 discarded_blocks=2
wall_time_s=0.000465 requests_per_second=6448.42
```

JSON 报告包含：

- 请求数、空请求数和输入 block occurrence 数；
- prefix hit、admission 和 block hit ratio；
- eviction path、evicted block、backend write 和 timeout discard；
- backend write/discard 字节数；
- 回放耗时和请求处理速度。

`prefix_block_hit_ratio` 的分母是输入中的 block occurrence，不是 token 数：

```text
prefix_block_hit_ratio = prefix_hit_blocks_total / block_occurrences_total
```

## 6. 快速自测

将第 2 节的三行样例保存为 `/tmp/ucm-replay-trace.jsonl`，然后执行：

```bash
python3 benchmarks/on_evict_trace_replay.py \
  --trace-path /tmp/ucm-replay-trace.jsonl \
  --timestamp-unit ms \
  --capacity-gb 1 \
  --block-size-bytes 536870912 \
  --dump-max-idle-s 1
```

预期核心结果：

```text
requests=3
blocks=5
prefix_hit_ratio=0.2
eviction_paths=2
evicted_blocks=3
backend_writes=1
discarded_blocks=2
```

本工具验证缓存策略、命中关系、淘汰范围和写回统计，不验证真实 KV payload、设备搬运
带宽、vLLM 时延或后端实际 I/O 时延。详细设计位于 G35 工作目录下的
`docs/ucm_replay/ucm_store_trace_replay_design.md`，不属于本仓库。
