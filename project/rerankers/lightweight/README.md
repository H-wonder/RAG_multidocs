# Lightweight Reranker

独立manifest+adapter:rank，依赖仅标准库及公共Reranker契约。配置目录后通过policy.reranker='lightweight'选择。

排序信号为0.8语义分数（缺失时使用倒数排名）加0.2候选内归一词面分；不是CrossEncoder，也不是正确概率。当前简化词面统计不是旧完整BM25的逐项等价实现，效果比较属于P1。

manifest的inprocess仅对宿主明确可信路径生效；同进程无强制超时保证，任意外部路径默认子进程。缺失、禁用、依赖/执行/非法输出降级原RRF顺序。禁用用registry.enabled=set()，不删除资料。

接口、分数范围、分发与测试见 ../../docs/RAG_PLUGIN_CONTRACTS.md。新增算法只加独立目录与配置，不改核心列表。
