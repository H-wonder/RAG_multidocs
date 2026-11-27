# Custom Reranker 独立样例

只依赖rag_multidocs.contracts，入口adapter:rank；按config.preferred_term（默认表格）是否出现在候选中调整顺序。ID/rank/score返回给核心，不能修改正文或来源。

复制到显式rerankers目录，policy.reranker='custom'即可调用，无需编辑核心业务源码。registry.enabled=set()禁用；失败返回原始RRF顺序和warning。复制其他名字组件并配置即可验证新增发现；不是在核心增加if/else。

此样例只证明架构接入，不声称提升检索准确率，不代表其他成员独立贡献或Review。完整契约与独立核心哈希验收见 ../../docs/RAG_PLUGIN_CONTRACTS.md。
