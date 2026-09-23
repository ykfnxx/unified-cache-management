# ContextStore|Fake 设计

配置见 [README](../../../ucm/store/context/README.md)。ContextStore 是独立 Store，传输代码从 CacheStore 复制到 Context 目录，不依赖 CacheStore 的管理类，也不保留另一套可切换的传输实现。

## 数据结构

- **radix 策略**：维护前缀拓扑、访问顺序、驻留集合及 context_lru victim 选择。MLA 由 rank0 的策略线程管理。
- **block slot**：共享 hash bucket 将逻辑 block 映射到稳定 slot。首次入驻登记一次待处理事件，策略在 Observe、淘汰或统计时合并；不按 layer 重复 Insert。合并时只在 allocation 锁内摘下事件链，树维护在锁外完成。
- **shard handle**：复制 Cache 的所有权和生命周期形式，各 shard 独立维护引用及 LOADING/READY/FAILED。持有 handle 同时 pin 其 block，槽位不能在传输期间重用。每个 rank 可取得未就绪 shard 的填充权。
- **payload**：按层组织共享内存，各 rank 独立注册；策略不保存设备地址。

命中和传输不获取策略锁。新分配通过 allocation → hash bucket 顺序加锁；已有 block 通过 bucket 取得引用后单独操作 shard。策略线程按 policy → allocation → bucket 更新驻留。完成回调只更新原子计数和释放 handle，不反向获取 policy。

## 读取与保存

Load 调度线程按 Cache 的 rank 偏移取得 handle，填充者提交后端 Load，再将 handle 和后端任务交给 SPSC 传输队列。传输线程等待 shard READY，然后连续提交 H2D，在任务末尾同步并释放所有 handle。后续 shard 的提交与前面 shard 的后端等待/H2D 可以重叠。layerwise 只回填请求层，完整 Memory 命中必须全部层 READY。

Dump 队列等待设备前置事件、执行 D2H、同步并发布 shard READY，至此完成，不写后端。MLA 仅 rank0 保存设备 KV；任意 rank 都可执行后端 Load。普通 Load 不要求其他 rank 同步加入，晚到的读取自行命中或重新回填。

## On-evict write

无空闲 slot 时唤醒策略线程；各 rank 都能使用已释放的 slot。策略按 context_lru 选择完整逻辑 block，并用原子引用检查预留 victim：

- Drop：直接释放，不删除已有后端记录。
- Dump：后端已有记录则跳过写入；否则写回所有 shard，等待成功后释放。
- 后端失败：保留 victim，撤销淘汰预留。

部分设备写入尚无完整副本时不可正常淘汰；已从后端回源的部分层可以释放，后续按需重新读取。失败的空分配回收不计入策略淘汰。retention 只选择 Drop/Dump，没有到期删除或退出全量刷写。

## 失败与验证

后续 shard 提交失败时，传输队列仍排空此前提交的后端任务和 DMA，之后才释放地址并完成 Wait。析构按队列顺序排空任务。首次填充失败发布 FAILED，后续无人持有时允许重试。

测试覆盖策略与参考树一致、完整 block 命中、仅淘汰写回、失败重试、重复保存、字节正确性、TP16 单填充者、非零 rank 独立回源，以及 MLA/GQA 跨进程共享。CPU-simu 用于正确性和主机管理耗时对照，不能替代 NPU/真实模型 TTFT 测量。
