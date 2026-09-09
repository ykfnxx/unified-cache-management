# OnEvictCache 真实 KV offload 使用说明

本文对应 `g35-leaf-lru-real` 分支的单后端实现，配置文件为
[ucm_on_evict_real.yaml](ucm_on_evict_real.yaml)。

OnEvictCache 把真实 KV 保存到主机共享内存，使用 UCM 的设备传输和异步任务接口完成
D2H/H2D。Resident 与 Dumped 是同一份 KV 的逻辑状态：淘汰为 Dumped 时保留原内存，
淘汰为 drop 时释放数据。`store_pipeline` 只配置 `OnEvictCache`。

## 1. 部署前准备

在已经能运行目标模型的 vLLM 环境中安装本分支；Ascend 环境还需要对应的
vLLM-Ascend、torch_npu 和 CANN。本文不更换现有模型环境的版本。

- Scheduler 和 Worker 必须能够访问同一个主机的 `/dev/shm`，并使用相同 UCM
  `unique_id` 命名空间；正常启动时由 Connector 生成和传递。
- 当前共享内存后端用于本机进程间复用，不提供跨主机 P/D 传输。
- 实际主机内存包括 Resident、Dumped 和正在写入的 KV；`fake_res_cap` 不是物理内存上限。
- 本分支已通过 CPU `simu` 关键路径测试；Ascend/CUDA 的主机内存注册、实际设备搬运
  和完整模型推理仍需在部署机验收。

先检查主机与容器内的资源：

```bash
free -h
df -h /dev/shm
ulimit -l
```

容器部署时，在**现有已验证的模型容器启动命令**中配置共享内存与锁页额度，例如追加：

```text
--shm-size=128g --ulimit memlock=-1
```

这里的 128 GiB 是容器 tmpfs 容量示例，须按主机可用内存和实验总 KV 量调整。
即使 `fake_res_cap=64`，运行过程也可能超过 64 GiB，因为 Dumped 数据仍在内存。
保留原有 GPU/NPU 设备映射、驱动/CANN 挂载及网络配置。多个容器中的进程若需要复用 KV，
还必须共享同一个 IPC/共享内存环境；分别设置 `--shm-size` 不会让两个容器共享数据。

## 2. 获取分支并安装

以下命令用于在部署机建立独立 checkout：

```bash
git clone --branch g35-leaf-lru-real --single-branch \
  git@github.com:ykfnxx/unified-cache-management.git ucm-onevict-real
cd ucm-onevict-real
```

仓库也已推送到 `ndsf/unified-cache-management` 的同名分支。
后续命令都在仓库根目录、模型服务使用的 Python 环境中执行。

### 2.1 从源码编译安装

先加载部署机已有的 CUDA 或 CANN 环境。Ascend 的 `set_env.sh` 路径以本机安装位置为准。
安装构建工具，然后按设备选择一个 `PLATFORM`：

```bash
python -m pip install 'setuptools>=64' wheel 'cmake>=3.18'

# 普通 Ascend 构建；Atlas A3 改为 ascend-a3，NVIDIA GPU 改为 cuda。
export PLATFORM=ascend
python -m pip install -v -e . --no-build-isolation
```

本仓库 `setup.py` 根据 `PLATFORM` 选择 C++ runtime。**不要省略该变量**：未匹配的平台
会构建成 `simu`，不能用它验证真实设备 offload。仅修改 Python 文件或安装 PyPI 的
`uc-manager` 不能代替编译本分支的 `libonevictcachestore.so`。

editable 安装会把 C++ 动态库安装到当前源码包中。每个新终端启动服务前设置：

```bash
export LD_LIBRARY_PATH="$PWD/ucm/shared/metrics${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

python - <<'PY'
import ctypes
from pathlib import Path
import ucm
from ucm.shared.metrics import ucmmetrics
from ucm.store.pipeline import ucmpipelinestore
from ucm.store.pipeline.connector import UcmPipelineStoreBuilder

root = Path(ucm.__file__).resolve().parent
library = root / 'store/on_evict_cache/libonevictcachestore.so'
ctypes.CDLL(str(library))
assert UcmPipelineStoreBuilder.get('OnEvictCache') is not None
print('UCM package:', root)
print('OnEvict library:', library)
print('OnEvict import and registration OK')
PY
```

这些检查验证包路径、动态库加载和注册，不代表设备传输已经通过。

### 2.2 UCM 与 vLLM 的集成补丁

沿用当前模型环境的 UCM 集成方式。本仓库对支持的 vLLM 版本提供运行时补丁：

```bash
export ENABLE_UCM_PATCH=1
```

仓库的安装文档说明该方式适用于 vLLM ≥ 0.11.0；旧版 0.9.2 使用手动补丁。
具体版本和补丁操作见仓库的 [Ascend 快速入门](../docs/source/getting-started/quickstart_vllm_ascend.md)
或 [CUDA 快速入门](../docs/source/getting-started/quickstart_vllm.md)。

## 3. 配置文件

直接使用 [ucm_on_evict_real.yaml](ucm_on_evict_real.yaml)：

```yaml
use_lite: false
use_layerwise: true
enable_metrics: true

ucm_connectors:
  - ucm_connector_name: "UcmPipelineStore"
    ucm_connector_config:
      store_pipeline: "OnEvictCache"
      fake_res_cap: 64
      on_evict_cache_policy: "radix_lru"
      on_evict_cache_dump_max_idle_s: 3600
```

| 参数 | 单位 / 默认值 | 含义 |
| --- | --- | --- |
| `fake_res_cap` | 正整数 GiB，真实模式必填 | 每个写入 Store 实例的伪驻留容量 |
| `on_evict_cache_policy` | 真实模式默认 `radix_lru` | 支持 `radix_lru` 或 `lru` |
| `on_evict_cache_dump_max_idle_s` | 秒，默认 `0` | 淘汰时空闲时间达到阈值则 drop；`0` 关闭超时丢弃 |
| `timeout_ms` | 毫秒，默认 `30000` | 可选，Store 异步任务等待超时 |
| `use_layerwise` | 顶层，默认 `true` | 分层保存/恢复；普通非 PP 模型可按现有配置使用 `false` |
| `use_lite` | 顶层，默认 `false` | 必须为 `false` 才进入真实 Store 路径 |
| `enable_metrics` | 顶层，默认 `true` | 启用 UCM 内置指标 |

容量换算：

```text
resident_block_limit = floor(fake_res_cap × 2^30 / block_size_bytes)
```

`block_size_bytes` 是 Connector 根据当前模型、分片与 I/O 对齐得出的物理 block 字节数，
不是 vLLM `--block-size` 的 token 数。`block_size`、`shard_size`、`tensor_size_list`、
`device_id` 和 `unique_id` 由 Connector 提供，通常不在 YAML 中手填。

TP 等并行配置下，`fake_res_cap` 按写入实例生效，不是所有 Worker 共享的总额度；
MLA 的写入 rank 选择及其他模型的 rank hash 规则沿用现有 Connector。

真实模式不配置 `storage_backends`、`cache_buffer_capacity_gb` 或
`on_evict_cache_capacity_gb`。最后一个参数属于旧的 `OnEvictCache|Fake` trace 模拟入口。

## 4. 启动模型服务

下面是单机、TP=1 的接入模板。替换模型目录；模型所需的其他参数沿用已验证的启动命令。
首次验收使用 eager 模式，便于定位搬运和模型问题。

```bash
export ONEVICT_MODEL=/path/to/model
export ONEVICT_CONFIG="$PWD/examples/ucm_on_evict_real.yaml"
export UCM_LOG_PATH="$PWD/log-on-evict"
export UCM_LOG_LEVEL=info

ONEVICT_KV_CONFIG=$(python - <<'PY'
import json
import os
print(json.dumps({
    'kv_connector': 'UCMConnector',
    'kv_connector_module_path': 'ucm.integration.vllm.ucm_connector',
    'kv_role': 'kv_both',
    'kv_connector_extra_config': {'UCM_CONFIG_FILE': os.environ['ONEVICT_CONFIG']},
}))
PY
)

vllm serve "$ONEVICT_MODEL" \
  --served-model-name onevict-model \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --enforce-eager \
  --port 8000 \
  --kv-transfer-config "$ONEVICT_KV_CONFIG"
```

`use_layerwise: true` 需要相应模型的 layerwise KV layout 支持。若现有模型配置使用
`use_layerwise: false`，保留该设置；PP 路径要求 layerwise。

服务启动后确认日志使用的 `store_pipeline` 为 `OnEvictCache`、`fake_res_cap` 正确，
且没有动态库、共享内存分配或 host registration 错误。

## 5. 验证真实 KV 复用

先发送一个包含足够多完整 KV block 的请求，再发送相同前缀的请求。例如：

```bash
python - <<'PY'
import json
import urllib.request

payload = json.dumps({
    'model': 'onevict-model',
    'prompt': 'Please summarize this passage. ' + 'The cache stores reusable context. ' * 128,
    'max_tokens': 16,
    'temperature': 0,
}).encode()
for attempt in range(2):
    request = urllib.request.Request(
        'http://127.0.0.1:8000/v1/completions', data=payload,
        headers={'Content-Type': 'application/json'},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        print(attempt + 1, response.read().decode())
PY
```

按模型 tokenizer 调整 prompt 长度，避免超过 `--max-model-len`。若启用了服务鉴权，
同时给请求增加相应 Authorization header。

两次请求都成功只说明服务可用。默认 HBM prefix cache 可能直接命中；要专门检查
OnEvict 的恢复路径，可在一次独立验收运行中给启动命令追加
`--no-enable-prefix-caching`，并结合 UCM lookup/load 日志和接口指标确认发生了恢复。
这不是正常部署必须关闭的选项。

要触发 eviction，需要让不受保护的 Resident KV 达到 `fake_res_cap`，然后继续写入
不同前缀。等待超时本身不会执行 drop；只有后续容量淘汰时才判断空闲时间。

可以查看服务指标：

```bash
curl -s http://127.0.0.1:8000/metrics | rg 'on_evict_|ucm.*(load|save|lookup)'
```

| 指标 | 含义 |
| --- | --- |
| `on_evict_eviction_paths_total` | 已提交的淘汰路径数量 |
| `on_evict_evicted_blocks_total` | 淘汰的 Resident block 数 |
| `on_evict_backend_write_requests_total` | 模拟下盘写入次数，当前逐 block 计数 |
| `on_evict_backend_write_bytes_total` | 模拟下盘字节数，不是实际磁盘 I/O |
| `on_evict_discarded_blocks_total` | 超时直接丢弃的 block 数 |
| `on_evict_discarded_bytes_total` | 超时直接丢弃的字节数 |

单独 OnEvictCache 不经过 CacheStore，因此不要用 `cache_d2h_*`、`cache_h2d_*`
等 CacheStore 专属指标判断本后端的传输。自定义 `metrics_config_path` 会替换指标启用列表，
需要包含上表中的指标；不配置该路径时使用内置集合。

## 6. 查看 DEBUG 日志

重新编译安装本分支的 C++ 动态库后，在**启动服务之前**设置：

```bash
export UCM_LOG_LEVEL=debug
export UCM_LOG_PATH="$PWD/log-on-evict"
export UCM_LOG_TO_FILE=true

# 排查时需要看到每个任务，可临时关闭按日志位置的限流。
export UCM_LOG_RATE_LIMIT_ENABLE=false
```

然后用第 4 节命令启动服务；不要再次执行其中的 `export UCM_LOG_LEVEL=info`。
这些变量需要传入实际启动 Worker 的容器/进程。日志级别在 logger 初始化时读取，
修改另一个终端的变量不会改变已经运行的服务。Release 构建也可以输出 DEBUG 日志。

日志同时输出到控制台和 `$UCM_LOG_PATH/ucm-<pid>.log`。只显示 OnEvict 日志可使用：

```bash
tail -F "$UCM_LOG_PATH"/ucm-*.log | rg --line-buffered 'OnEvict'
```

| 级别 | 内容 |
| --- | --- |
| INFO | 初始化角色/设备、模式、命名空间、策略、GiB 容量、block 上限和 idle 阈值 |
| DEBUG | Lookup 命中/未命中数；任务 ID、load/dump、shard 数、逻辑字节数、排队/执行耗时 |
| DEBUG | Dump 实际复制 shard/有效 payload 字节数、新发布 block 数、驻留量和保留 payload 数 |
| DEBUG | 每次 eviction 的淘汰、dump、drop 数量，以及淘汰后的驻留量 |
| ERROR | 任务失败、共享内存分配/注册、设备拷贝、事件等待、发布失败及 NoSpace 状态 |

`task=... op=dump/load start` 与 `complete` 用于关联同一个任务。任务日志的 `bytes`
按 `shards × shard_size` 计算，可能包含对齐空间或已保存的 shard；
`copied_payload_bytes` 才是本次 Dump 提交复制的有效 tensor 字节数。
`retained_payload_blocks` 包括 Resident、Dumped 和未完成的 block。

UCM 默认会对同一日志位置限流（10 秒内最多 3 条），因此只设置 DEBUG 不一定能看到
每次操作。`UCM_LOG_RATE_LIMIT_ENABLE=false` 可关闭该限流，但高流量下异步日志队列
仍可能丢弃旧消息，日志不能作为精确计数来源。恢复常规部署时可设置
`UCM_LOG_LEVEL=info`、`UCM_LOG_RATE_LIMIT_ENABLE=true`，精确累计量看 metrics。

## 7. 停止服务与常见问题

正常停止 Worker 会等待已提交传输并释放它持有的共享内存。异常退出可能留下
`/dev/shm/uc_on_evict_<unique_id>_*` 对象。清理前先停止对应实例，并根据该实例日志中的
`unique_id` 确认对象归属；不要批量删除其他运行中实例的共享内存。

| 现象 | 排查方向 |
| --- | --- |
| `libmetrics.so` 找不到 | 在启动终端设置第 2 节的 `LD_LIBRARY_PATH`，保留原 CUDA/CANN 路径 |
| 未知 pipeline 或库缺失 | 确认 import 到本分支，已重新编译并安装 C++ 动态库 |
| `allocate KV shared memory` 失败 | 检查容器 `/dev/shm` 和主机内存；增大伪驻留容量不能解决物理内存不足 |
| host registration / 锁页失败 | 检查当前设备 runtime、驱动/CANN、共享内存注册支持和锁页额度 |
| `NoSpace` / `-50009` | 当前请求保护了全部可淘汰叶子；检查伪驻留容量是否能容纳工作集，缩短 idle 阈值不能解除路径保护 |
| 第二次请求看不到 offload Load | 排除 HBM prefix hit、短于完整 block、异步 Dump 尚未完成及配置了 Lite 模式 |
| 运行越久主机内存越多 | Dumped KV 仍保存在内存，当前没有 Dumped GC，也不模拟磁盘容量/时延 |

算法状态、生命周期和 CPU 测试命令详见
[OnEvictCache 实现说明](../docs/source/user-guide/on_evict_cache.md)。
