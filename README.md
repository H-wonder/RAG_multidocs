# RAG_multidocs

课程大作业项目：**RAG Connector Platform——插件化多模态文档解析与检索**。

当前阶段：已完成可复用模块、双 conda 环境配置、离线与外部服务验证 demo；
正式产品骨架（CLI、Connector 注册、端到端流程）待搭建。

## 目录

| 路径 | 内容 |
| --- | --- |
| [`已有资源/代码/`](已有资源/代码/) | 文档模型、切块、去重、BM25 索引、Milvus 混合检索、docparse Connector、插件注册 |
| [`已有资源/文档/`](已有资源/文档/) | 流水线、IR、切块/去重、Milvus、课程复用映射、来源许可、环境搭建 |
| [`已有资源/环境/`](已有资源/环境/) | `setup.sh`、`check_env.sh`、requirements、`.env.example`、Milvus Docker 配置 |
| [`已有资源/样例/`](已有资源/样例/) | 离线 RAG demo、Milvus/docparse 冒烟脚本、样例数据 |
| [`已有资源/可选组件/docparse/`](已有资源/可选组件/docparse/) | 多模态解析源码 |
| `docs/` | 系统设计文档（最终提交的 `DESIGN.docx` 放这里） |
| `大作业要求/` | 课程任务书、交付模板与选题参考 |

## 快速开始

完整的从零搭建步骤见 [`已有资源/文档/07-环境搭建与依赖.md`](已有资源/文档/07-环境搭建与依赖.md)。

```bash
# 1. 创建两个 conda 环境：rag（平台）+ docparse（解析）
bash 已有资源/环境/setup.sh

# 2. 检查工具、环境、依赖和服务状态
bash 已有资源/环境/check_env.sh

# 3. 离线验证：接入 → 切块 → BM25 索引 → 检索（带来源）
conda activate rag
cd 已有资源/代码 && python -m pytest -q
cd ../样例 && python demo_rag.py
```

可选的外部服务冒烟：

```bash
# 需要 Milvus 2.5+ 已启动
python 已有资源/样例/demo_milvus.py

# 需要 docparse 环境和 DOCPARSE_* 配置
DOCPARSE_REPO=已有资源/可选组件/docparse \
DOCPARSE_PYTHON=$(conda run -n docparse which python) \
python 已有资源/样例/demo_docparse.py /path/to/sample.pdf
```

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
BM25 关键词索引、双环境安装/自检、验证 demo 和课程向文档。

来源、许可与第三方声明见
[`已有资源/文档/06-来源许可与依赖.md`](已有资源/文档/06-来源许可与依赖.md)。
