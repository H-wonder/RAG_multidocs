# 第三方代码与来源声明

## 已有起点

本项目的部分模块改编自作者此前的 DocFlow 工程（`new1`）：

| 本项目文件 | 来源 | 处理 |
| --- | --- | --- |
| `src/rag_multidocs/model.py` | DocFlow 的 docparse IR / Chunk | 重写为课程可用的统一模型 |
| `src/rag_multidocs/chunking.py` | `任务二/docflow/chunker.py` | 去掉全局配置依赖，参数化；增加 TXT/Markdown 切块 |
| `src/rag_multidocs/index.py` | 新写 | BM25 关键词索引，纯标准库 |
| `src/rag_multidocs/registry.py` | `skills/registry.py` | 改为通用 Connector 注册表 |
| `src/rag_multidocs/connectors/pdf.py` | `docparse_connector.py` | 薄封装 docparse CLI |
| `src/rag_multidocs/index_milvus.py` | DocFlow 的 Milvus 检索设计 | 包成 `IndexBackend`，补 list_docs/persist |
| `src/rag_multidocs/milvus_store.py` | `任务二/docflow/services/rag_service.py` | 抽成可嵌入的 Milvus 混合检索类 |
| `src/rag_multidocs/embedding.py` | `rag_service.py` 的嵌入函数 | Embedder 协议 + bge/DashScope/hashing |
| `src/rag_multidocs/connectors/txt.py` | 新写 | 内置 TXT Connector |

本次新增的 CLI、TXT/Markdown Connector、错误模型、测试、CI、设计文档和
release 模板不计入既有代码。

## docparse（可选依赖）

PDF/扫描件/Office 解析由外部 docparse 提供，本项目不复制其源码，只在运行时通过
子进程调用 `DOCPARSE_PYTHON -m docparse`。

- 源码位置：`../已有资源/可选组件/docparse/`
- docparse 的 `docparse/vendor/` 来自 SeerRay-Lab/Xiaomi-OCR-0，**Apache-2.0**；
  完整许可见 `../已有资源/可选组件/docparse/docparse/vendor/LICENSE-Xiaomi-OCR-0.txt`。
- docparse 仓库根目录没有显式 LICENSE 文件；公开发布前请确认再分发许可。
- 未配置 docparse 时，PDF Connector 返回 `CONNECTOR_UNAVAILABLE`，核心功能不受影响。

## 第三方 Python 依赖

核心运行时只依赖 Python 标准库。开发/测试依赖：

| 依赖 | 许可 | 用途 |
| --- | --- | --- |
| pytest | MIT | 测试 |
| ruff | MIT | 静态检查 |

可选运行时（docparse、Milvus、Ollama）的依赖与许可见
`../已有资源/文档/06-来源许可与依赖.md`。
