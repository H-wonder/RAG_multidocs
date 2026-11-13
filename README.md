# RAG_multidocs

![CI](https://github.com/H-wonder/RAG_multidocs/actions/workflows/ci.yml/badge.svg)

课程大作业项目：**RAG Connector Platform——插件化多模态文档解析与检索**。

当前阶段：可运行的项目骨架已完成——CLI、Connector 注册、TXT/Markdown/PDF 接入、
Milvus 混合检索 + BM25 回退、检索增强问答、测试与 CI 已落地；最终发行物与成员协作记录待补。

## 目录

| 路径 | 内容 |
| --- | --- |
| [`project/`](project/) | 可运行产品：CLI、Core、内置 Connector、独立 Markdown 扩展、测试、CI |
| [`project/docs/DESIGN.docx`](project/docs/DESIGN.docx) | 系统设计文档（提交版） |
| [`已有资源/代码/`](已有资源/代码/) | 文档模型、切块、去重、BM25 索引、Milvus 混合检索、docparse Connector、插件注册 |
| [`已有资源/文档/`](已有资源/文档/) | 流水线、IR、切块/去重、Milvus、课程复用映射、来源许可、环境搭建 |
| [`已有资源/环境/`](已有资源/环境/) | `setup.sh`、`check_env.sh`、requirements、`.env.example`、Milvus Docker 配置 |
| [`已有资源/样例/`](已有资源/样例/) | 离线 RAG demo、Milvus/docparse 冒烟脚本、样例数据 |
| [`已有资源/可选组件/docparse/`](已有资源/可选组件/docparse/) | 多模态解析源码 |
| [`release/RELEASE.md`](release/RELEASE.md) | 发布记录模板（当前 HOLD，待生成最终发行物） |
| `大作业要求/` | 课程任务书、交付模板与选题参考 |

## 快速开始

```bash
# 平台运行在 rag conda 环境；docparse 由子进程调用
conda activate rag
cd project
pip install -e ".[dev,milvus]"
cp config.example.json config.json

# 启动 Milvus（首次部署见环境文档）
docker start milvus-standalone

# TXT 导入 + Milvus 混合检索（带来源）
rag-cli ingest --source txt --path samples/notes.txt
rag-cli search "检索器如何训练"

# 独立 Markdown 扩展
rag-cli ingest --source connector.markdown --path samples/markdown
rag-cli search "表格"
rag-cli connectors list

# 检索增强问答（API Key 只从环境变量读取）
export DEEPSEEK_API_KEY=...
rag-cli ask "检索器如何训练" --provider deepseek --model deepseek-chat

# 测试与检查
pytest -q -m "not integration"
ruff check src tests connectors
MILVUS_URI=http://localhost:19530 pytest -q -m integration
```

没有 conda 时可以用 venv 跑核心 + 关键词回退：

```bash
cd project
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
RAG_INDEX_BACKEND=keyword rag-cli ingest --source txt --path samples/notes.txt
```

PDF/扫描件解析需要 docparse 环境；完整环境步骤见
[`已有资源/文档/07-环境搭建与依赖.md`](已有资源/文档/07-环境搭建与依赖.md)，
产品说明与分工模板见 [`project/README.md`](project/README.md)。

## 环境与外部服务

- 平台环境：`rag`（Python 3.11，pymilvus + sentence-transformers）
- 解析环境：`docparse`（docparse + PyMuPDF + PaddleX）
- Milvus：Docker，`milvusdb/milvus:v2.5.6`
- OCR：宿主机 Ollama，`longwayxu/xiaomi-ocr-0:bf16`

配置模板见 [`已有资源/环境/.env.example`](已有资源/环境/.env.example)；
所有变量说明见环境文档第 3 节。

## 来源与许可

本项目基于作者此前的多模态文档入库工程（DocFlow）演化，复用/改编了其中的
文档模型、切块、去重、清单、Milvus 检索和 docparse 解析代码；本次新增了
可运行 CLI、Connector 契约与注册、TXT/Markdown Connector、Milvus 混合检索、
BM25 关键词回退、集成测试与 CI、设计文档和 release 模板。

来源、许可与第三方声明见
[`已有资源/文档/06-来源许可与依赖.md`](已有资源/文档/06-来源许可与依赖.md)。
