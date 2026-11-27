# RAG Multidocs 产品目录

完整项目介绍、环境安装、架构、四种策略、贡献与AI核验见[仓库README](../README.md)。本目录包含Python包、独立插件、测试和课程设计文档，不改变现有目录和入口。

## 从本目录启动

```bash
conda activate rag
python -m pip install -e '.[dev,milvus,embed]'
# 仅首次新建配置；已有config.json勿覆盖
cp config.example.json config.json
PYTHONPATH=src python -m rag_multidocs --config config.json repl
```

配置Milvus集合、模型缓存、docparse解释器及凭据后再导入。若沿用仓库根的已有配置，则从根目录运行并显式指定该配置，不要重新创建空知识库。

## 接入入口

1. [USAGE](docs/USAGE.md)：实际命令、配置和排错。
2. [RAG_CORE_API](docs/RAG_CORE_API.md)：CLI同学优先阅读，公开签名、结果和生命周期。
3. [RAG_PLUGIN_CONTRACTS](docs/RAG_PLUGIN_CONTRACTS.md)：Connector与Reranker开发。
4. [DESIGN](docs/DESIGN.md) / [DOCX](docs/DESIGN.docx)：设计取舍、数据和流程。
5. [最终验收](docs/testing/FINAL_ACCEPTANCE.md) / [提交准备](docs/testing/DELIVERY_PREPARATION.md)。

源码使用MIT许可，外部解析器与模型的许可独立，见[第三方声明](THIRD_PARTY.md)。正式Commit、Tag、Review、CI及制品结果由团队完成，不能以本目录可运行宣称发布完成。
