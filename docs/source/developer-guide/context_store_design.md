# ContextStore|Fake 设计

运行参数和使用方法见 [ContextStore README](../../../ucm/store/context/README.md)。

## 组成

Pipeline 按 Fake → Context 的顺序 Stack，对外入口为 `ContextStore|Fake`。ContextStore 通过已有 StoreV1 后端接口访问 Fake，不再拥有第二级 payload 数据池。

ContextIndex 维护前缀拓扑、访问顺序、驻留集合和 context_lru 淘汰选择。BufferPool 只管理 Memory。SharedMetadata 只发布完整 Memory block 的位置、可用性和 MLA 读引用；后端可用性由 Fake.Lookup 决定，避免在 ContextStore 中保留过期的 Fake 命中副本。

## 数据和策略

设备 Dump 使用现有 CopyStream 等待设备 event，D2H 后同步。全部 layer/shard 完成才发布 block。普通 Dump 不调用后端。

Memory 不足时，先按 context_lru 选择可淘汰候选，再按 retention 选择 Drop 或 Dump：Drop 直接释放 Memory；Dump 查询后端，若不存在则提交该 block 的所有 shard 并等待完成，成功后才释放 Memory。失败保留未完成淘汰的 victim，撤销 MLA 淘汰预留。已有后端记录不重复写，也不因 Drop 删除。

Load 优先固定已驻留 Memory。后端命中时按需淘汰并分配一个完整 Memory block，调用后端 Load/Wait 填充尚未就绪的 shard，发布 READY 后传输本次请求的层。入驻的 block 加入 ContextIndex；重复 Load 命中 Memory。Fake.Load 本身为空操作，所以入驻的数据不是恢复出的正确 KV。

## 并行和生命周期

GQA 每 rank 独立策略和 Memory；canonical hash 用于前缀关系，后端 hash 的高 8 字节与 TP rank 做 XOR，区分同一逻辑 block 的 rank 副本。scheduler 逐 rank 合并 Memory 和后端命中后取交集。各进程使用一致的 TP 和 DP 命名配置。

MLA worker 在 Setup 阶段完成共享 payload 映射和设备注册。任意 rank 可先创建共享 payload，文件锁仅保护容量检查和空间预分配，各 rank 独立完成设备注册后才接受 Load，不等待 owner 元数据；初始化开销不计入首层 Load。

MLA 只由 rank 0 Dump、淘汰和回源入驻，其他 rank 通过进程共享条件变量等待 READY 或失败，持有读引用完成 H2D。在途批次按 block/shard 描述及重复提交顺序匹配，使用有界共享表记录参与/完成 rank 和首个失败码。rank 0 本地同步后立即完成 Load/Wait，独立后台线程等待 reader 完成后释放 block 引用，H2D 线程继续处理后续层；完成队列有界，容量复用 waiting_queue_depth，保留 task 引用到该批次成功或失败；读者的实际 DMA 始终由自身读引用保护。正常完成的批次可复用；超时且所有已加入参与者结束的批次可回收。支持最多 64 个 MLA TP rank，不维护长期请求状态。

节点、驻留项、引用计数使用哈希查询；策略排序保持原有有序结构。索引将入驻/淘汰的祖先增量合并，在 Select 和实际删除拓扑节点前从深到浅刷新，保证策略查询看到精确计数。ObserveRequest 不跳过新的访问时间，但同一 segment 的冷排序索引只刷新一次。

Dump 保留一个消费队列。Load 使用 prepare 和 H2D 两个线程；复用 CacheStore SPSC 队列及消费循环，逐 shard 移交 slot 地址；按 device_id 和 local_rank_size 错开提交顺序，每个 block 仅持有一次引用，传输与后续准备重叠。失败仍投递任务结束标记，传输线程同步所有已提交 DMA 后完成本地任务；MLA owner 的 block 引用延迟到 reader 结束才释放。Load 在改变容量前检查不同 block 数不超过 Memory slot 数；条目数已不超过 slot 数时无需重复构造去重集合。不实现滑动窗口。

首次 Memory miss 时批量查询剩余 block，全命中不做额外预扫描或后端查询，并按最多 32 blocks 合并 Load/Wait 与描述分配；部分准备失败仍处理之前已预留的 block，后端失败释放新槽位并保留原有 shard 状态。

提交时在同一次引用表查询中校验和增加引用：已有正引用保证拓扑存在，首次引用仍检查索引；提交失败回滚本次已增加的引用。完成时只对引用降为零的节点尝试清理。准备阶段的去重及后端查询表使用哈希容器并预留容量，不参与 victim 排序。

共享元数据仅在 entry 状态变化或新增可读 entry 时广播；尚未就绪且未插入的 block 不唤醒等待者。MLA 后台完成队列保持容量上限，非阻塞检查各 batch，优先释放已完成或失败的任务；全部仍在等待时最多每 100μs 检查一次，新任务可提前唤醒检查。保留 batch deadline 和退出时的失败处理，不改变本地 H2D 完成即返回的语义。所有 rank 都完成的连续尾部退出 batch 匹配扫描，保留在途 batch 及同签名任务的匹配顺序。

共享哈希表删除使用 backward-shift 修复探测链，避免 tombstone 累积。读路径分别记录 ObserveRequest、准备排队、准备耗时、MLA READY/完成等待、首次 H2D 和最终同步的 Histogram。

复用 CacheStore 的多 stream 传输方式。后端 Load/Dump 和 Wait 过程中不持有策略 mutex，busy 状态保护正在填充或淘汰的 entry。请求退休不影响已提交任务；任务完成后释放临时拓扑引用。

支持单机 TP、GQA/MLA、direct/layerwise、已注册 MTP shard；PP=CP=1。没有真实 SSD、跨机、后台刷写、额外淘汰框架或崩溃租约恢复。

## 验证

C++ 使用可注入失败的字节后端验证地址、readmission、全层可用性、Drop/Dump 和共享读引用。Python 使用真实 FakeStore 与 ContextStore pipeline 验证 Fake no-op 后的真实 CPU-simu 拷贝、TP 命中交集和跨进程 MLA 入驻读取，并验证 connector 布局及配置路径。

CPU-simu 测试不证明设备吞吐或真实 GLM 推理正确性；Fake payload 本身也不支持输出正确性验证。
