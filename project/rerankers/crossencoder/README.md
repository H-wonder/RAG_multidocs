# CrossEncoder Reranker

独立adapter:rank，Manifest要求sentence-transformers>=4.0。默认BAAI/bge-reranker-base、CPU、batch8、max512，logit经sigmoid只作排序信号，不是答案正确概率。

显式选择crossencoder才执行；local_files_only默认true，缺缓存/依赖报结构化warning并回退，不自动下载。现有CLI build_runtime批准本插件精确路径使用一个串行常驻Worker，模型在该进程缓存，超时/崩溃按既有规则回收。单独构造Registry/create_core不会自动得到相同批准，宿主需配置persistent_paths；没有授权时仍可走一次性隔离进程，不能把它称为常驻。

registry.close或RAGCore/Runtime.close释放资源。不是热加载、并发池或安全沙箱。契约、配置覆盖、禁用/故障、ID回联和测试见[公开协议](../../docs/RAG_PLUGIN_CONTRACTS.md)。用户Trace已确认常驻Worker真实运行，但不宣称每题更准或给出未经实测的加速倍数。
