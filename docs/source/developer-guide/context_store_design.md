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

MLA 只由 rank 0 Dump、淘汰和回源入驻，其他 rank 等待 owner 发布共享 Memory，持有读引用完成 H2D。读引用和本地 task 引用均在 stream 排空后释放；淘汰预留原子检查全部 victim，防止读写重用同一 slot。

Load 和 Dump 各有一个消费队列，复用 CacheStore 的多 stream 传输方式。后端 Load/Dump 和 Wait 过程中不持有策略 mutex，busy 状态保护正在填充或淘汰的 entry。请求退休不影响已提交任务；任务完成后释放临时拓扑引用。

支持单机 TP、GQA/MLA、direct/layerwise、已注册 MTP shard；PP=CP=1。没有真实 SSD、跨机、后台刷写、额外淘汰框架或崩溃租约恢复。

## 验证

C++ 使用可注入失败的字节后端验证地址、readmission、全层可用性、Drop/Dump 和共享读引用。Python 使用真实 FakeStore 与 ContextStore pipeline 验证 Fake no-op 后的真实 CPU-simu 拷贝、TP 命中交集和跨进程 MLA 入驻读取，并验证 connector 布局及配置路径。

CPU-simu 测试不证明设备吞吐或真实 GLM 推理正确性；Fake payload 本身也不支持输出正确性验证。
