# 已有资源：RAG Connector Platform 可复用代码与文档

本目录从作者此前的多模态文档入库工程（DocFlow / `new1`）中提取，目标是服务课程大作业的
**RAG Connector Platform** 选题。只保留能直接落到新项目里的代码和设计说明，不搬运
Airflow、邮件通知、人工复核等与本课程无关的运行时。

> 重要：这些代码是“已有起点”。课程任务书允许基于已有项目演化，但要求在 README
> 交代来源、既有能力和本次增量，不能把已有代码全部算作本组贡献。具体写法见
> [文档/05-课程项目复用映射.md](文档/05-课程项目复用映射.md) 与
> [文档/06-来源许可与依赖.md](文档/06-来源许可与依赖.md)。

## 目录

```text
已有资源/
├── 代码/
│   ├── document_model.py        # 统一文档/块模型（核心契约）
│   ├── chunking.py              # TXT / Markdown / docparse IR 三种切块
│   ├── dedup.py                 # SimHash 近重复检测（纯标准库）
│   ├── manifest_store.py        # SQLite 文档清单与状态机（增量入库）
│   ├── plugin_registry.py       # manifest + adapter 插件注册/发现
│   ├── keyword_index.py         # BM25 关键词索引（离线回退）
│   ├── embedders.py             # 嵌入后端协议与三种实现（RAG 主检索）
│   ├── milvus_hybrid_index.py   # Milvus dense + BM25 混合检索（RAG 主检索）
│   ├── docparse_connector.py    # 调用 docparse 的薄 Connector（多模态解析）
│   └── tests/                   # 37 个测试，全部通过
├── 可选组件/
│   └── docparse/                # 多模态解析完整源码副本（PDF/扫描件/Office）
├── 文档/
│   ├── 01-多模态入库流水线.md
│   ├── 02-文档模型与IR.md
│   ├── 03-切块与去重.md
│   ├── 04-Milvus混合检索.md
│   ├── 05-课程项目复用映射.md
│   ├── 06-来源许可与依赖.md
│   └── 07-环境搭建与依赖.md
├── 环境/
│   ├── setup.sh                 # 一键创建两个 conda 环境
│   ├── check_env.sh             # 环境自检（不安装、不启动服务）
│   ├── requirements-rag.txt     # 平台环境依赖
│   ├── requirements-core.txt    # 平台测试工具
│   ├── requirements-docparse.txt # 解析环境依赖
│   ├── milvus/                  # Docker 部署用的 embedEtcd.yaml / user.yaml
│   ├── README.md                # 安装入口
│   └── .env.example
└── 样例/
    ├── demo_rag.py              # 离线闭环：接入→切块→索引→检索
    ├── demo_milvus.py           # 可选：Milvus 混合检索冒烟
    ├── demo_docparse.py         # 可选：docparse 多模态解析冒烟
    ├── demo.py                  # 只演示三种输入的切块结果
    ├── notes.txt
    ├── markdown/
    └── docparse_ir_sample.json
```

## 代码在目标架构里的落点

| 模块 | 运行环境 | 作用 |
| --- | --- | --- |
| `document_model.py`、`chunking.py`、`dedup.py`、`manifest_store.py`、`plugin_registry.py` | **rag（平台）** | 平台核心：文档模型、切块、去重、增量清单、Connector 注册 |
| `embedders.py`、`milvus_hybrid_index.py` | **rag** + Docker Milvus | RAG 主检索：dense + 服务端 BM25，命中带来源 |
| `docparse_connector.py`、`可选组件/docparse/` | **docparse（解析）**，平台子进程调用 | 多模态解析：PDF 原生抽取、扫描件 OCR、Office |
| `keyword_index.py` 关键词倒排 | **rag** | 离线回退；Milvus 不可用时仍能完成接入—索引—检索 |

**原则**：两个 conda 环境职责分离——平台不 import docparse，只通过子进程调用。
paddle 与 torch 分属两个环境/进程，互不污染依赖。

Milvus 是主检索，但必须保留关键词回退，保证演示不依赖单一外部服务。

## 快速验证

```bash
bash 环境/check_env.sh        # 只检查环境，不安装

conda activate rag
cd 已有资源/代码
python -m pytest -q
# 37 passed

cd ../样例
python demo_rag.py            # 离线闭环：接入→切块→索引→检索（带来源）
python demo.py                # 只演示三种输入的切块结果
```

可选冒烟（需要对应外部服务）：

```bash
python demo_milvus.py         # 需要 Milvus 已启动；否则打印 skip
python demo_docparse.py /path/to/sample.pdf   # 需要 docparse 环境和 DOCPARSE_* 配置
```

平台核心只依赖 Python 3.10+ 标准库；RAG/多模态能力由
`环境/requirements-rag.txt` 与 `环境/requirements-docparse.txt` 提供。

## 环境搭建（双环境，推荐）

目标形态是“插件平台 + 多模态文档解析 + RAG”。使用两个职责分离的 conda 环境：

```bash
cd 已有资源
bash 环境/setup.sh
```

| 组件 | 形态 | 说明 |
| --- | --- | --- |
| 平台主体 + 嵌入 + 测试 | **rag conda env** | 依赖见 [环境/requirements-rag.txt](环境/requirements-rag.txt) + [环境/requirements-core.txt](环境/requirements-core.txt) |
| 多模态解析 | **docparse conda env** | 依赖见 [环境/requirements-docparse.txt](环境/requirements-docparse.txt)；平台通过子进程调用 |
| Milvus 向量/BM25 | **Docker** | `milvusdb/milvus:v2.5.6`，容器 `milvus-standalone`；可选 Attu |
| OCR 模型 | **宿主机 Ollama** | `longwayxu/xiaomi-ocr-0:bf16`，只处理扫描页/图片 |

完整步骤、配置总表、Docker/Ollama 命令见
[文档/07-环境搭建与依赖.md](文档/07-环境搭建与依赖.md)。
`.env.example` 是配置模板，不要提交真实 `.env`。

## 推荐使用顺序

1. 读 [文档/05-课程项目复用映射.md](文档/05-课程项目复用映射.md)，确认目标版架构与插件契约。
2. 用 `document_model.py` 定义 Connector 的输出契约。
3. 内置 Connector：TXT + docparse（PDF/扫描件/Office）；用 `chunking.py` 切块。
4. 独立扩展：Markdown 目录 Connector，按 `plugin_registry.py` 接入，核心不改。
5. 检索：Milvus 混合检索为主（`milvus_hybrid_index.py`），关键词倒排作为离线回退。
6. 用 `dedup.py` 做重复检测，用 `manifest_store.py` 做增量入库（按需）。

## 来源与许可

- 主要来源：原工程 `new1` 的 `任务二/docflow/`（chunker、dedup、manifest、config）、
  `任务二/docflow/services/rag_service.py`、`任务二/vendor/docparse/`、
  `skills/registry.py`。
- `可选组件/docparse/` 中的 `docparse/vendor/` 来自 Xiaomi-OCR-0，**Apache-2.0**，
  完整许可见该目录下 `LICENSE-Xiaomi-OCR-0.txt`。
- docparse 仓库根目录没有显式 LICENSE 文件；在把整个目录用于公开发布前，请先确认
  作者/课程对再分发的许可要求。课程内部复用请保留来源说明。
- 原项目中的 `.env` 含真实密钥，**没有**复制到本目录；任何 API Key 都应通过环境变量注入。

详细映射见 [文档/06-来源许可与依赖.md](文档/06-来源许可与依赖.md)。

## 没有提取的内容

- Airflow DAG 编排、XCom、ApprovalOperator（课程项目不需要工作流引擎）；
- 邮件/ntfy 通知、人工复核回路、Docker Compose；
- `ocr_service.py` 的 HTTP 包装层（直接用 `docparse_connector.py` 子进程调用即可）；
- 真实 `.env`、日志、缓存、模型权重。
