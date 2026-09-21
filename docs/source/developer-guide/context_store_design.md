# ContextStore：context_lru 与 DRAM 模拟 SSD 实现方案

状态：固定 retention / 固定淘汰粒度版本已实现。CPU simu 验证覆盖真实字节搬运和共享元数据；尚未完成 NPU 或真实 vLLM 推理验证。支持同机 GQA 独立多 TP 池，以及 MLA 单 rank 写入、多 rank 共享读取；自适应参数未实现。

运行配置、接口和验证命令见 [ContextStore 使用说明](../../../examples/README_context_store.md)。

源码基线：UCM `93014e5b48b4e69fc402c5379a1d8d8d226ef913`；DWPDSim `f2c2021` 的 C++ 策略实现。

## 1. 目标与已确认约束

实现一个独立的 UCM Store，暂名 `ContextStore`：

- 设备 KV 与本机 DRAM 之间进行真实数据传输。
- Memory 层使用 DWPDSim 的 `context_lru` 管理容量。
- 额外划出一份 DRAM 作为模拟 SSD，保存真实 KV payload。
- **仅在 Memory 淘汰决策为 Dump 时，才向模拟 SSD 写入。**
- 从 CacheStore 直接复制并裁剪多 stream、D2H/H2D、event、异步队列等代码。不修改 CacheStore 来承载新策略，不要求抽取共用框架。
- 前缀关系由 UCM connector 从 vLLM 获取，传给新 Store。
- 一个逻辑 block 在每层对应一个完整 DRAM slot，各 shard 使用固定偏移。
- worker 侧 Store 管理策略、数据池和传输；scheduler 侧 Store 通过共享元数据查询缓存可用状态。

对外配置为 `store_pipeline: ContextStore`，不组合 Cache、Empty 或 Posix。没有真实文件 I/O、远端服务或 HIXL。

## 2. 核心语义：on-evict write

必须区分两个不同层次的 Dump：

| 名称 | 含义 | 是否写模拟 SSD |
| --- | --- | --- |
| `Store::Dump(task)` | 将设备 KV 保存到 Memory | 否 |
| context_lru 的 `Dump` 决策 | 淘汰 Memory victim，必要时写入模拟 SSD | 是，仅无现有 SSD 副本时 |
| context_lru 的 `Drop` 决策 | 丢弃 Memory victim | 否 |

内部淘汰写入函数使用 `WriteBack` 命名，避免与 Store API 混淆。

```text
Store::Dump
    设备 ──D2H──→ Memory
                  不同时写模拟 SSD

Memory 分配空间不足
    context_lru 选择 segment 和动作
        ├─ Drop：释放所选 Memory 副本
        └─ Dump：
             已有 SSD 副本 → 直接释放 Memory 副本
             没有 SSD 副本 → Memory 复制到 SSD 池
                            → 发布 SSD 副本
                            → 释放 Memory 副本

Store::Load
    Memory 命中 → Memory ──H2D──→ 设备
    仅 SSD 命中 → SSD 池 ──H2D──→ 设备
    两层均缺失 → 返回未命中/加载失败
```

Load、Lookup、请求结束和 Store 销毁均不主动新增 SSD 副本。没有定期 flush。测试结束时留在 Memory 的数据不计作 SSD 写入。

`Drop` 只删除 Memory 副本；已有 SSD 副本保持有效。

## 3. 代码组织与复用范围

代码位于 `ucm/store/context/`：

| 文件 | 职责 |
| --- | --- |
| `context_store.cc` | StoreV1 接口、配置、任务与内存管理的协调 |
| `context_index.h/.cc` | block 前缀树、segment、访问索引、context_lru 决策 |
| `buffer_pool.h` | 两个固定 slot 池及空闲槽位管理 |
| `shared_metadata.h/.cc` | worker 发布 block 可用状态，scheduler 以 Watcher 方式查询 |
| `context_store.cc` 内的两个 Queue | 独立 Load/Dump 工作线程、任务引用及完成管理 |
| `copy_stream.h` | 从 CacheStore 复制的设备传输代码 |

复用 CacheStore 的 pinned/mapped host buffer 分配能力、CopyStream、event 等待、多 stream 提交和同步、任务 handle/Wait 组织方式。保留已有低层 `trans` 库依赖。

不复制 Cache 的 CLOCK、backend 回源、backend Dump 等待线程、shard 自动覆盖逻辑。引用保护概念保留，但引用绑定 ContextStore 自己的 block/slot。

第一版不增加通用 EvictionPolicy 工厂、可插拔 SSD 后端、RPC 服务或独立写回线程池。写回先由 Dump 工作线程同步完成 Host 内存复制；后续只有实测证明需要，才增加复制流水线。

## 4. 数据布局与所有权

每个存储命名空间由 worker 侧拥有一棵前缀树和两个池；scheduler 的查询实例不创建这些结构。容量分别计算：

```text
memory_slots = floor(memory_capacity_bytes / block_bytes)
ssd_slots    = floor(simulated_ssd_capacity_bytes / block_bytes)

block B
    memory_slot：可选
    ssd_slot：可选
    shard i 地址 = 对应层 slot 基址 + shard_offset[i]
```

这里的 block 是该命名空间内的逻辑 block。TP rank 独立存储时，block_bytes 是该 rank 应保存的数据量，不是全模型各 rank 的总量。

worker 私有拓扑节点保存 parent、children 和全局 depth；缓存条目保存两个 slot、写入就绪信息、引用和策略访问状态。共享元数据发布 BlockId 和完整副本可用标志；MLA 还共享 slot 索引、布局签名、读引用和淘汰占用状态。不共享树、segment 或进程私有指针。

一个 shard 内多个 tensor 按固定 offset 布局，由初始化配置计算。一个 block 的所需 shard 全部成功完成 D2H 后，Memory 副本才 READY。

拓扑存在不代表数据驻留。无两层副本但还有子节点的节点保留，以维持 parent/depth；无副本、无子节点且无在途用途时剪枝。

### 状态与引用

- 正在填充的 block 不参与淘汰、不报告完整命中。
- Load 在查找数据时同时取得 slot 引用，H2D 真正完成后释放。
- WriteBack 开始时保护源 Memory slot 和目标 SSD slot；复制完成后才发布 SSD 副本。
- 淘汰选取和条目标记在同一临界区内完成，防止选取后被另一个线程取得引用。
- 删除或复用 slot 前先确保引用为零，不能让索引继续指向已复用地址。

失败的 D2H 不发布 READY；涉及同一 block 的在途传输全部结束后才清理其空间。无法成功完成 WriteBack 时保留 Memory 数据，返回容量/任务错误，不隐式改成 Drop。

## 5. context_lru 策略

移植 DWPDSim 的策略语义和必要索引，不引入整个 Simulator 或 Python 运行时。

1. 按 segment 最近访问序号从老到新遍历合法 leaf segment，即 endpoint 下没有其他 Memory 驻留后代段。
2. 累计候选段的 Memory 驻留 block 数，达到 `alpha × memory_slots` 时停止；至少一段，越过预算的最后一段完整纳入。
3. 按 `(endpoint depth, segment 驻留数, recency, endpoint ID)` 升序选择 victim。
4. 从选中段尾部选择最多 `max_eviction_blocks` 个 Memory block，提交回收时按前到后顺序处理，不继续向父段回收。
5. segment 的最近访问取其最近访问的驻留成员。拓扑分叉/剪枝时更新 segment 拆分和合并。

容量预算只包含 Memory 池，不包含模拟 SSD。UCM 使用完整 16-byte BlockId；不截断成 DWPDSim 的 uint64。平局按完整 ID 的稳定顺序比较；对照测试需保持 ID 排序一致。

### Drop / Dump 判定

沿用 DWPDSim：

```text
retention 未配置：Dump
idle > effective_retention：Drop
否则：Dump
```

idle 使用选中段最近访问的 Memory 成员的时间。retention 是淘汰时的动作判定阈值，不是保证驻留时间，也不是后台 TTL 删除条件。

自适应选项保留原含义：

- `effective_retention = retention + beta × growth_blocks`；beta 单位为秒/block。
- growth 使用 session 请求长度增量，发布时机遵循 DWPDSim 的请求事件语义。
- 动态回收量依据不同请求之间的历史访问间隔，限制在配置的最小/最大范围。

当前仅实现固定 retention 和固定粒度；上述自适应规则保留为后续扩展语义，尚无配置入口。vLLM 的请求处理完成与 DWPDSim 的一次同步 process 完成不是天然等价，具体事件映射需在 connector 接入时明确，不能直接以每层 Dump 结束代替。

### 在线传输保护

真实系统增加传输引用约束。被引用、正在填充或写回的 block 不能被淘汰；段尾遇到受保护 block 时不能越过它删除前缀。候选合法性必须考虑此约束。

无合法 victim 时不退回 CLOCK、不越过引用强制回收。无可回收容量时返回 NoSpace。受保护候选的过滤会使并发执行不同于串行模拟器；策略等价测试采用无并发引用的受控序列。

## 6. 请求上下文与 connector

完整前缀链由 connector 提供，Store 不从 hash 或某次 Load/Dump 的批量顺序猜测 parent。

批量上下文入口 `ObserveRequest` 承载：

- 请求标识与本次观察序号，用于重复调度去重；
- 完整有序 block IDs；
- 策略时间戳；
当前接口不接收 session/affinity；自适应增长需补充该契约。

StoreV1、Python/native binding、pipeline 和 direct/layerwise connector 已贯通该入口；常规 Load/Dump 的地址格式继续沿用现有 Shard。空 block 列表表示释放请求上下文，在途任务单独持有拓扑引用。

请求事件创建/更新拓扑；数据就绪事件更新驻留状态。逻辑 recency 由请求观察和首次驻留事件明确驱动，不由多 stream 完成先后决定，不把逐层调用当成独立请求。

时间源：线上使用单调时钟；确定性回放传入 trace 时间。禁止把真实时间和 trace 时间混在同一索引中。

### worker 所有权与 scheduler Watcher

UCM 作为库在调用方进程内运行。vLLM 的 scheduler 和 worker 分别创建 UCM Store 实例，两种实例承担不同职责：

| 状态或操作 | worker 侧 Store | scheduler 侧 Store |
| --- | --- | --- |
| 前缀树、segment、context_lru | 持有并更新 | 不持有 |
| Memory / SSD 数据池 | 分配、读写和释放 | 不分配、不映射 payload |
| Load / Dump / WriteBack | 执行并维护传输引用 | 不执行 |
| 共享可用状态 | 发布和撤销 | 查询 |
| Lookup / LookupOnPrefix | 可查询当前状态 | 通过 Watcher 查询共享元数据 |

```text
scheduler connector
    ├─ Lookup → ContextStore Watcher → 共享可用状态
    └─ 完整请求上下文 → vLLM connector metadata → worker connector
                                                     ↓
                                               ObserveRequest
                                                     ↓
worker ContextStore：前缀树 + context_lru + 两个数据池 + 传输队列
    └─ 数据状态变更 → 发布共享可用状态
```

共享查询参考 CacheStore 的 `SharedBufferWatcherStrategy`：worker 创建并初始化共享元数据，scheduler 按命名空间打开，仅映射查询区域。context_lru 索引留在 owner worker；MLA payload 由各 worker 共享，scheduler 不参与策略执行。

上下文随现有 connector metadata 送达 worker，在相应 Load/Dump 之前调用 ObserveRequest。scheduler 不通过 Lookup 修改 worker 的策略索引，也不从 Load/Dump 后缀推导完整前缀。

共享表使用进程间同步保证查询与状态发布的一致性；表中使用值和相对位置，不存放供另一进程解引用的进程私有指针。生命周期由 worker 所有者管理，Watcher 只解除自身映射。

发布规则：

- D2H 完成且整个 block READY 后，发布 Memory 可用。
- WriteBack 完成后，在同一次共享状态更新中发布 SSD 可用并撤销 Memory 可用，再回收 Memory slot。
- Drop 时先撤销 Memory 可用，再回收 slot；SSD 可用标志保持原值。
- 失败或未完成的副本不发布可用。

Lookup 返回查询时的可用状态，不持有传输引用。worker 的 Load 必须重新查找并取得引用；若查询后数据已被淘汰，则返回加载失败，由 connector 走加载失败回退。该窗口和回退路径纳入接入验证。

GQA 每个 TP rank 独立拥有策略索引、Memory、模拟 SSD 和元数据表，native 命名追加 `_tp<rank>`。scheduler 对各 rank 可用性逐 block 取交集。

MLA 沿用 CacheStore 的单 rank 写、多 rank 读。connector 自动设置 `share_buffer_enable: true`，native 命名追加 `_mla`；只有 rank 0 分配 Memory／模拟 SSD 的 POSIX 共享 payload，维护策略索引并执行 D2H、Dump/Drop。其他 rank 延迟映射同一份 payload，分别注册到自己的设备并执行 H2D。scheduler 只查询这份 owner 元数据。

共享表除了可用性，还记录 Memory/SSD slot、布局签名、在途读引用和淘汰占用状态。reader 在取得位置时增加引用，实际传输同步后释放；owner 筛选 victim 后在共享锁下再次原子确认并占用，避免查询与淘汰间的竞态。写回期间不接纳新的读取，完成后发布 SSD 位置并释放 Memory slot。若所有候选均被读取保护，可返回 NoSpace，不强制回收。

所有 rank 的上下文和 I/O 使用 scheduler 逻辑 block ID。普通 KV、RoPE 和 indexer 按已有布局完整传输；MTP 注册层参与整个 block 的就绪，重复回调按批去重。MLA ObserveRequest 只有 rank 0 更新策略，其余 reader 不维护索引。MLA 保存成功反馈沿用单 rank 写语义。

范围为同机 GQA/MLA TP、PP=CP=1。GLM-5.1-W4A8 的目标实验为 TP+MTP、DP=1、关闭 C8/CP，尚未进行真实模型验证。GQA 容量按 rank，MLA 按一份 TP 组共享池；模拟 SSD 预设不会写满，不增加其回收策略。MLA 的 `/dev/shm` 需容纳两个 payload 池和元数据。跨机、硬崩溃读引用回收及热重启不在本实验范围。

## 7. 读写流程与完成边界

### Store::Dump

1. 校验任务对应的上下文与布局，按 block 获取或分配 Memory slot。
2. 空间不足时执行 context_lru 淘汰；仅 Dump victim 进入 WriteBack。
3. 等待前置 event，复用多 stream 方式提交各 shard 的 D2H。
4. 同步相关 stream，标记该任务的 shard 就绪，完整 block 才发布 READY。
5. 释放传输引用并完成 task。

同 key、同布局表示不可变 KV。已 READY 的重复保存可跳过；同一 block 的不同层允许陆续写入。重复的同一 shard 在途写入需合并/等待，不能并发覆盖。

部分层写入会占用整个 block slot。容量必须容纳当前未完成 block 集合；否则返回明确的 NoSpace。不要通过释放未完成 block 或无限阻塞“解决”逐层写入造成的容量压力。

### Store::Load

1. 优先查 READY Memory，否则查 READY SSD 副本；同时取得引用。
2. SSD 命中直接从 SSD 池 H2D，不回填 Memory。
3. 多 stream 提交后同步，释放引用，完成 task。

SSD slot 也需要满足所选 H2D 方法的 host 注册/映射要求。不能把“普通 malloc 内存也叫 DRAM”视为与 CacheStore 传输条件等价。具体 pinned/mapped 分配复用底层能力，并统计两个池的实际占用。

`Wait` 成功意味着该任务所有实际 D2H/H2D 已完成；分配过程中触发的 WriteBack 已完成相应空间回收。`Check` 表示任务是否终结，最终错误由 Wait 返回。入队成功不是数据完成。

### Lookup

单 block 命中条件为 Memory 或 SSD 至少一个完整 READY 副本；`LookupOnPrefix` 在第一个两层 miss 处停止。缺失 Load 返回失败，不分配空 slot 冒充命中。

Lookup 与请求观察的职责分开：查询不重复刷新 recency，策略访问通过上下文事件统一处理。

## 8. 模拟 SSD 的边界

模拟 SSD 使用有限、预分配容量，满时返回 NoSpace 并保留 Memory victim。没有 SSD 淘汰策略。实验应配置足够容量并报告峰值；满容量测试单独进行。

使用真实 Host 内存复制，记录真实 payload 字节。暂不模拟 SSD 的带宽、排队和延迟，也不延时 sleep。

若以后增加 SSD 淘汰，必须独立定义其 victim 选择；Memory context_lru 不能顺带决定 SSD 淘汰。SSD 删除后，仍有 Memory 副本的 block 保持有效。

## 9. 配置草案

以下为配置模板，容量占位符需替换为实际数值。可直接加载的示例见 `examples/ucm_context_config.yaml`。容量也支持 `_capacity_bytes`，同一池不能同时指定 bytes 和 gb。

```yaml
store_pipeline: ContextStore
context_memory_capacity_gb: <Memory 容量>
context_simulated_ssd_capacity_gb: <独立的 SSD 池容量>
context_alpha: 0.01
context_retention_ns: null  # null 表示淘汰时始终选择 Dump
context_max_eviction_blocks: 64
```

设备、block/shard/tensor 大小、stream 数、队列深度、超时、GDR/SDMA/IO aggregation 使用与复制后的传输实现一致的配置。只支持已实际接入的传输分支，不暴露无效开关。自适应参数在实现其请求事件契约时再加入配置。

容量以 payload 为准，metadata、队列及设备映射开销另报。配置校验至少覆盖 block 布局整除、池能容纳一个完整 block、alpha 范围及粒度正值。

## 10. 并发方案

worker 内用一个 metadata mutex 保护树、索引、slot 分配和引用状态。数据搬运与 stream 等待全部在锁外执行。完成时重新加锁提交状态。共享查询表另用进程间同步；发布顺序固定为先取得 worker metadata 锁，再取得共享表锁，Watcher 只取得共享表锁。

Load 和 Dump 各自使用工作队列及复制 stream，保持 CacheStore 的提交与传输重叠方式。WriteBack 初期同步在 Dump 工作线程执行，允许其他 Load 继续执行。

task 持有其触及 block/slot 的引用。传输已提交后，即使任务超时也不能立即释放地址；要等设备访问结束才能回收。worker 退出先停止接收新任务并撤销对外可用状态，再收敛已提交传输，最后释放池和共享元数据。scheduler Watcher 不释放 worker 的池，不删除共享区域。

## 11. 验证与性能比较

### 功能与策略

用公开接口的少量确定性场景验证：

- 新数据 Dump 后仅 Memory 驻留，SSD 写入计数为零。
- 强制淘汰触发 Dump，数据进入 SSD，后续 Load 内容一致。
- retention 触发 Drop，不产生 SSD 写入；已有 SSD 副本不被删除。
- Memory/SSD 双副本淘汰不产生重复 WRITE。
- 多层 block 部分完成不报告完整命中；传输中的 block 不被回收。
- 分叉、共享前缀、候选预算跨界、段尾截断与 DWPDSim 选择一致。
- 两层 miss、SSD 满、无可回收 block 均明确失败且不伪造数据。
- 分进程验证 worker 发布 READY、WriteBack 和 Drop 后，scheduler Watcher 的查询结果正确；Watcher 不分配 payload 或策略索引。
- 查询后淘汰再 Load 明确失败，connector 回退有效；Watcher 退出不破坏 worker 数据，worker 退出后不继续报告可用。

策略回放使用同容量、时间戳、ID 顺序和请求事件；逐步比较 victim、Drop/Dump、两层驻留及逻辑读写。线上并发和分层写入另行验证，不直接宣称与同步 Simulator 逐步等价。

### 指标

- 设备 D2H/H2D 字节、完成延迟及吞吐；
- Memory 命中、SSD 命中、连续前缀两层总命中与 global miss；
- 淘汰 block 数、Drop 数、Dump 数；
- 实际 SSD 写入 block/字节数、已有副本跳过数、SSD 读出字节数；
- 两池当前与峰值占用、NoSpace、队列等待和策略决策耗时。

选中 Dump 的 block 数与实际新增 SSD WRITE 数分开。模拟 SSD 性能只代表 DRAM 搬运，不能解释为真实 SSD 性能。

### 基线

1. `Cache|Empty` 对比驻留工作集的设备↔DRAM 传输性能，确保没有淘汰写回干扰，按提交至 Wait 完成计时。
2. 新 Store 内固定相同两池容量，对比 retention 关闭和开启，衡量写入减少与命中变化。
3. 两级方案与 Cache|Empty 的端到端结果同时报告额外 SSD 池容量，不能将额外容量收益全部归因于 context_lru。

## 12. 实现范围

- 独立 Store、两个数据池、完整 block 就绪、传输引用和 on-evict WriteBack 已实现。
- context_lru 使用增量 segment 索引，支持固定 retention、固定淘汰粒度和稳定排序。
- worker 发布共享可用状态，scheduler Watcher 查询；请求全前缀通过 connector metadata 传给 worker。
- 多 stream、event 和复制方法由 CacheStore 复制，原 CacheStore 未改动。
- SSD 命中不回填；SSD 满时返回 NoSpace；传输超时先收敛设备访问再返回。
- 自适应 retention/粒度及真实设备性能验证不在当前已完成范围内。多 TP 的 CPU simu 测试覆盖独立进程、部分就绪、独立淘汰和跨层回读。

## 源码依据

- UCM `ucm/store/cache/cc/copy_stream.h`：设备复制、stream 和 event。
- UCM `ucm/store/cache/cc/dump_queue.cc`：前置事件、D2H、同步和 backend 分界。
- UCM `ucm/store/cache/cc/load_queue.cc`：异步 H2D 与持有 buffer。
- UCM `ucm/store/cache/cc/trans_buffer.cc`：SharedBufferWatcherStrategy 的元数据映射；现有 CLOCK 不作为新策略的回收实现。
- UCM `ucm/store/detail/type/types.h`：BlockId、Shard、TaskDesc。
- UCM `ucm/integration/vllm/ucm_connector.py`：完整请求 hash 与外部查询后缀。
- DWPDSim `cpp/src/policies/memory/context_memory_lru_policy.cpp`：候选排序、自适应 retention/粒度。
- DWPDSim `cpp/src/policies/memory/indexed_memory_lru_policy.cpp`：segment 索引与拓扑更新。
- DWPDSim `cpp/src/simulator.cpp`：段尾选择、Drop/Dump 执行及已有存储副本的去重。
