# ContextStore|Fake 设计

配置见 [README](../../../ucm/store/context/README.md)。ContextStore 是独立 Store，传输代码从 CacheStore 复制到 Context 目录，不依赖 CacheStore 的管理类，也不保留另一套可切换的传输实现。

## 数据结构

- **radix 策略**：维护前缀拓扑、访问顺序、驻留集合及 context_lru victim 选择。MLA 由 rank0 的策略线程管理。
- **block 元数据**：共享 hash bucket 记录逻辑 block 及其已驻留 shard 链。它不预留整个 block 的 payload。首次入驻登记一次待处理事件，策略在 Observe、淘汰或统计时合并；不按 layer 重复 Insert。合并时只在 metadata 锁内摘下事件链，树维护在锁外完成。
- **shard handle**：复制 Cache 的所有权和生命周期形式，各 shard 独立维护引用及 LOADING/READY/FAILED。引用只计在 shard 上，命中不修改 block 引用或 radix；block 记录使用中 shard 的数量，只在引用 0→1、1→0 时更新；淘汰检查该计数，槽位不能在传输期间重用。每个 rank 可取得未就绪 shard 的填充权。
- **payload**：按 shard 分配共享槽位，使用 `(block, layer)` hash 查找，地址为 slot × shard_size；各 rank 独立注册，策略不保存设备地址。完整映射使用 MAP_POPULATE，在初始化期间建立映射。

命中使用与 Cache 相同的 `(block, layer)` hash、bucket 锁和 shard 锁，在 shard 引用从 0 变为 1 时取得填充权。尚未使用的槽通过共享游标取得；radix 释放的槽通过独立待用链表 O(1) 取得，不扫描仍在驻留的 shard。链表锁只保护槽位领取/归还，不覆盖 block/shard 查找或传输。新增 shard 的锁顺序是 block bucket → shard bucket → shard；只有创建 block 元数据时额外获取 metadata 锁。正常命中、引用复制和释放不获取 radix 锁。Handle 的复制不更新 block 状态；每个 shard 从无人持有变为有人持有以及最后一个引用释放时，才更新使用中 shard 计数。容量不足才进入策略等待；引用释放仅在容量压力期间通知等待者。

block 的 shard 链只用于入驻关联、写回和释放。选择 victim 时 O(1) 检查使用中 shard 数；预留时只获取一次 block bucket 锁，用一次 CAS 将零计数置为淘汰中，不扫描各层 shard 锁。Get 为此前无人持有的 shard 取得填充权时，同步登记使用中状态；若 block 已被预留则等待释放或撤销。任何已有传输引用都会阻止预留。

## 读取与保存

Load 调度线程按 Cache 的 rank 偏移取得 handle，填充者提交后端 Load，再将 handle 和后端任务交给 SPSC 传输队列。传输线程等待 shard READY，然后连续提交 H2D，在任务末尾同步并释放所有 handle。后续 shard 的提交与前面 shard 的后端等待/H2D 可以重叠。layerwise 只回填请求层，完整 Memory 命中必须全部层 READY。当前层全部入队后，按 Cache 的顺序预分配下一层的 shard 槽；预分配不取得填充权、不提交后端读取，空间不足则跳过，避免策略写回阻塞 dispatcher。shard 描述符使用 move 入队，任务末尾保留最后一个 handle 到同步完成。

Dump 队列等待设备前置事件、执行 D2H、同步并发布 shard READY，至此完成，不写后端。MLA 仅 rank0 保存设备 KV；任意 rank 都可执行后端 Load。普通 Load 不要求其他 rank 同步加入，晚到的读取自行命中或重新回填。

## On-evict write

无空闲 shard slot 时唤醒策略线程；各 rank 都能立即使用每个 victim 释放的 slot。一次压力周期回收到约 5% 槽空闲或暂无可淘汰候选，减少反复唤醒的往返；不是提前按水位启动的扫描。策略按 context_lru 选择完整逻辑 block，并通过 block 使用中 shard 计数预留 victim：

- Drop：释放该 block 的所有已驻留 shards，不删除已有后端记录。
- Dump：后端已有记录则跳过写入；否则写回所有 shard，等待成功后释放。
- 后端失败：保留 victim，撤销淘汰预留。

Lookup、Dump、Wait 在 radix 锁外执行，预留的 shard 在此期间不能复用；下一请求的 Observe 无需等待写回。释放时先归还 block 元数据，再发布空闲 shard，最后通知空间等待者。

部分设备写入尚无完整副本时不可正常淘汰；已从后端回源的部分层可以释放，后续按需重新读取。失败的空分配回收不计入策略淘汰。retention 只选择 Drop/Dump，没有到期删除或退出全量刷写。候选暂时被传输引用占用时，分配者等待释放事件后重试；未写完整且没有后端副本的数据不能强行淘汰。

纯 decode 没有 UCM Load/Dump，不重复生成完整前缀 observation。新请求/恢复以及有保存的 chunked prefill 保留 Observe；请求结束仍显式释放拓扑引用。

## 与 Cache 的对应关系

- CopyStream 的实现仅改命名空间；stream 数、轮转、事件等待、H2D/D2H 提交和同步一致。
- Load 保留 rank 错序、每 shard 单填充者、dispatcher/transfer 两阶段队列、逐 shard H2D 和任务末尾同步。Handle 的复制、移动、析构使用同样的 shard 引用生命周期。
- allocator 使用相同 hash、bucket/node 临界区和首次分配游标；Cache 的 clock victim 被 radix 的整 block victim 及其归还的 slot 替换。Context 不允许覆盖尚未完成 on-evict write 的槽，因此容量等待属于策略边界。下一层预分配在此边界选择跳过等待。
- Dump 保留事件 → D2H → 同步 → READY；后端写入从普通 Dump 移至策略淘汰。
- 错误和析构路径排空已提交 I/O 后才释放地址；没有保留旧 allocator、逐 Handle 的 block 聚合引用、rank0 专属 Load 或另一套可切换实现。

## 失败与验证

后续 shard 提交失败时，传输队列仍排空此前提交的后端任务和 DMA，之后才释放地址并完成 Wait。析构按队列顺序排空任务。首次填充失败发布 FAILED，后续无人持有时允许重试。

测试覆盖策略与参考树一致、完整 block 命中、仅淘汰写回、失败重试、重复保存、字节正确性、TP16 单填充者、非零 rank 独立回源，以及 MLA/GQA 跨进程共享。CPU-simu 用于正确性和主机管理耗时对照，不能替代 NPU/真实模型 TTFT 测量。
