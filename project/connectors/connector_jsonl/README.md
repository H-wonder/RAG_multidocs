# connector.jsonl

独立架构样例，将JSONL中每行的source/text/title/blocks转换为公共Document；每行失败返回RECORD_ERROR，其他行继续。用于验证新来源无需改核心，不冒充真实OCR。

manifest为connector.jsonl，入口adapter:load，依赖仅Python标准库和rag_multidocs.model/contracts。复制整个目录到显式connectors_dir，允许列表加入connector.jsonl后重启即可接入；禁用仅禁止新接入，不删除已入库知识。

记录可以含provenance/source_sha256/annotation/origin，保留为元数据。样例normalized.jsonl的reference_transcription必须保持该来源语义；文件哈希相符不证明转录正确。

完整公开契约、错误和独立测试见 ../../docs/RAG_PLUGIN_CONTRACTS.md。真实课程独立扩展及Review仍需由实际组员完成。
