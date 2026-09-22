# ContextStore|Fake

通过独立的 ContextStore 实现 context_lru，串接 FakeStore 做元数据后端。设备 Dump 只写 Memory；淘汰 Dump 才调用 Fake；Fake 命中 Load 会重新进入 Memory，参与后续策略。没有模拟 SSD payload。

使用 [通用配置](ucm_context_config.yaml) 或 [GLM-5.1 TP/MTP 配置](ucm_context_glm51_config.yaml)。参数、构建启动、TP/MLA 行为和统计见 [ContextStore README](../ucm/store/context/README.md)。

Fake.Load 不恢复真实 KV，但仍执行真实 H2D；本部署用于与 `Cache|Fake` 比较策略和设备↔DRAM 路径，不能验证模型输出正确性。比较时保持 Memory 容量、Fake 元数据容量、TP、layerwise、传输选项和工作负载一致；计时边界以 Wait 完成为准。
