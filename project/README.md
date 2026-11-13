# RAG Multidocs——插件化多模态文档接入与检索

![CI](https://github.com/H-wonder/RAG_multidocs/actions/workflows/ci.yml/badge.svg)

面向课程资料使用者：用户指定数据源、导入文档、输入关键词，获得**带来源**的命中文本。
本项目把“接入 → 统一文档 → 切块 → 索引 → 检索”做成可扩展的插件平台：
核心只依赖 Connector 公开契约，新增数据源不需要修改核心源码。索引层默认为
**Milvus dense + BM25 混合检索**，Milvus 不可用时回退到本地 BM25 关键词索引。

- 仓库：<https://github.com/H-wonder/RAG_multidocs>
- 设计文档：[docs/DESIGN.md](docs/DESIGN.md)（提交版为 `docs/DESIGN.docx`）
- 已有资源与来源说明：[../已有资源/README.md](../已有资源/README.md)

## 用户、场景与范围

**谁**：需要检索课程资料的教师/学生。
**什么情况**：手头有 TXT、Markdown、PDF/扫描件等多来源资料，希望统一导入后按关键词检索，
并能看到命中来自哪个文件、哪一页、哪一节。
**输入 → 处理 → 输出**：

```text
TXT / Markdown / PDF ──► Connector ──► Document ──► Chunk ──► Milvus（dense+BM25）
                                                                  │          或 keyword 回退
用户查询 ─────────────────────────────────────────────────────────┘
        └──► 命中文本 + source + page + section_path
              └──►（可选）LLM 依据命中生成答案 + [编号] 引用
```

**基础能力（内置）**：

- 统一 `Document` / `Chunk` 模型；
- 内置 TXT Connector（单文件或目录）；
- 内置 PDF/扫描件/Office Connector（通过 docparse 子进程，可选）；
- Milvus 混合检索：dense 向量 + 服务端 BM25，命中带 `source/page/section_path`；
- 本地 BM25 关键词索引：Milvus 不可用时的离线回退；
- 检索增强问答：`rag-cli ask` 把命中证据交给 OpenAI 兼容 LLM，输出答案 + 引用；
- 模型选择：embedding/chat 均支持 provider + model 配置/命令行覆盖；
- 接入失败的结构化错误（`FILE_NOT_FOUND`、`EMPTY_SOURCE`、`PARSE_ERROR` 等）。

**独立扩展**：

- `connectors/connector_markdown/`：Markdown 目录 Connector；
- 通过 `manifest.json + adapter.py` 接入，核心源码不改；
- 禁用后 TXT/PDF 与检索仍可用。

**本次不做什么**：

- 不做用户登录、权限、多租户；
- 不做爬虫集群、插件市场、热加载；
- 不做多轮对话记忆、Agent 工具调用或复杂评测；生成答案必须附带检索引用；
- Milvus/嵌入模型不是硬依赖：不可用时回退关键词索引，保证离线闭环。

## 快速开始

### 1. 安装

核心只依赖 Python 3.10+ 标准库：

项目主体运行在 `rag` conda 环境；docparse 解析环境由子进程调用。

```bash
conda activate rag
cd project
pip install -e ".[dev,milvus]"
```

如果机器上没有 conda，可以用 venv 跑纯核心/关键词模式：

```bash
cd project
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Milvus 先启动：

```bash
docker start milvus-standalone
curl -s http://localhost:9091/healthz
```

PDF/扫描件解析需要另外准备 docparse 环境，见
[../已有资源/文档/07-环境搭建与依赖.md](../已有资源/文档/07-环境搭建与依赖.md)。

### 2. 配置

```bash
cp config.example.json config.json
# 默认 index_backend=milvus、embed_backend=local
# 没有 Milvus 时会自动回退 keyword，并打印 WARN
# 只跑 CI/冒烟可用 embed_backend=hashing，避免下载模型
```

关键字段：

| 字段 | 默认 | 说明 |
| --- | --- | --- |
| `index_backend` | `milvus` | `milvus`（主）或 `keyword`（离线） |
| `fallback_to_keyword` | `true` | Milvus 不可用时自动回退 |
| `milvus_uri` | `http://localhost:19530` | Milvus 地址 |
| `milvus_db` / `milvus_collection` | `rag` / `rag_chunks` | 数据库/集合 |
| `embed_backend` | `local` | `local`/`hashing`/`dashscope`/`openai`/`ollama` |
| `embed_model` / `embed_dim` | 空 / 0 | 空/0 表示用 provider 默认；必须与集合维度一致 |
| `chat_provider` | 空 | `deepseek`/`dashscope`/`openai`/`ollama`；空表示不启用问答 |
| `chat_model` | 空 | 空表示用 provider 默认模型 |
| `rag_top_k` | 5 | 问答检索证据条数 |
| `docparse_repo` / `docparse_python` | 空 | PDF 多模态解析 |

### 3. 导入与检索

```bash
# TXT
rag-cli ingest --source txt --path samples/notes.txt

# 独立 Markdown Connector（默认启用）
rag-cli ingest --source connector.markdown --path samples/markdown

# 检索（带来源）
rag-cli search "表格"
rag-cli search "检索器如何训练" --top-k 3

# 查看文档/连接器/索引
rag-cli docs list
rag-cli connectors list
rag-cli stats
```

### 4. 检索增强问答

API Key 只从环境变量读取，不写入配置：

```bash
# 1. 查看可用 provider 和当前选择
rag-cli models list

# 2. 只返回检索证据，不调用 LLM
rag-cli ask "检索器如何训练" --no-generate

# 3. 用 DeepSeek 生成答案 + 引用
export DEEPSEEK_API_KEY=...
rag-cli ask "检索器如何训练" --provider deepseek --model deepseek-chat

# 4. 持久选择 chat provider/model（不保存 Key）
rag-cli models use --kind chat --provider deepseek --model deepseek-chat

# 5. 换 embedding provider（会提示新建 Milvus 集合）
rag-cli models use --kind embedding --provider dashscope --model text-embedding-v4 --dim 1024
```

输出包含答案和来源：

```text
答案：
检索器通过联合优化或固定生成器训练 [1]。
来源：
[1] notes.txt p0
```

API Key 环境变量：`DEEPSEEK_API_KEY`、`DASHSCOPE_API_KEY`、`OPENAI_API_KEY`；
Ollama 不需要 Key。Key 不写入 `config.json`。

示例检索输出：

```text
[1] 6.2538  chunking.md p0  §切块策略 > 按结构切 > 表格
    | 策略 | 优点 | 缺点 | ...
[2] 6.0203  chunking.md p0  §切块策略 > 按结构切 > 表格
    表格必须整块保留。按字符数切会把一行数据切成两半，检索结果无法阅读。
```

### 4. 测试

```bash
pytest -q -m "not integration"       # 离线单元测试
ruff check src tests connectors
MILVUS_URI=http://localhost:19530 pytest -q -m integration
```

## 核心与扩展

架构与接口设计见 [docs/DESIGN.md](docs/DESIGN.md)。

```text
CLI ──► Core（IngestService / SearchService / ConnectorRegistry）
             │
             ├── 内置 Connector：TXT、PDF/扫描件/Office（docparse 子进程）
             └── 独立扩展：Markdown 目录（connectors/connector_markdown）
             │
             ├── MilvusIndex（dense + BM25，主）
             │     └── Embedder（bge / DashScope / OpenAI / Ollama / hashing）
             └── KeywordIndex（BM25，本地 index.json，回退）

CLI ask ──► SearchService ──► RagQA（证据 + 引用）──► ChatModel
                                                  （DeepSeek/DashScope/OpenAI/Ollama）
```

### 独立扩展如何接入

扩展目录：

```text
connectors/connector_markdown/
├── manifest.json
├── adapter.py
└── README.md
```

核心通过 `ConnectorRegistry` 扫描 `connectors_dir`，读取 `manifest.json`，
按 `entry` 加载 `adapter.py`。扩展只依赖 `rag_multidocs.model` 与
`rag_multidocs.contracts`，不 import 核心内部实现。

启用/禁用：

```bash
rag-cli connectors list
rag-cli connectors disable connector.markdown
rag-cli connectors enable connector.markdown
```

禁用后核心仍可完成 TXT 导入与检索；重新启用后同一检索入口命中 Markdown 资料。
接入/禁用不需要修改 `src/rag_multidocs/`，也不需要重新构建核心。

### 接入时核心源码不变的核验

```bash
git diff --stat -- src/rag_multidocs
# 修改扩展前后，核心目录 diff 为空
```

## 测试与检查

| 类型 | 验证内容 |
| --- | --- |
| 正常 | TXT 导入 → 检索命中文档/片段，断言 `source` |
| 边界/失败 | 路径不存在、空文件、空目录、未知 Connector、禁用 Connector |
| 扩展接入 | Markdown 目录导入 → 同一检索入口命中；禁用后核心仍可用 |
| PDF 映射 | docparse IR → Document/Chunk，断言页码、章节、表格块 |
| 回退 | Milvus 不可用时自动回退 keyword，并打印 WARN |
| Milvus 集成 | `@pytest.mark.integration`：写入 → dense+BM25 检索 → 来源断言 |
| 问答 | FakeChat：证据拼入 prompt、答案带引用、无证据不调用模型 |
| 模型选择 | provider 列表、环境变量 Key 检查、`models use` 持久化 |
| CLI | `ingest/search/ask/models/docs/connectors` 的输出与退出码 |

本地命令：

```bash
pytest -q -m "not integration"       # 离线单元测试
ruff check src tests connectors

# 需要 Milvus 2.5+
MILVUS_URI=http://localhost:19530 pytest -q -m integration
```

CI 配置：仓库根 `.github/workflows/ci.yml`，包含 unit job 和 Milvus integration job。

**故意引入错误演示**：把 `TxtConnector` 中 `if not text.strip()` 改成
`if False`，`test_empty_file_and_empty_directory` 会失败；恢复后测试通过。

## 协作与贡献

- 问题：在 GitHub Issue 描述输入、期望结果、实际结果；
- 分支：`feat/<topic>`、`fix/<topic>`；
- 提交 PR/MR，至少一名成员审查；审查意见需针对行为、边界、失败处理或测试；
- 合并：核心/契约变更由核心负责人确认，扩展变更由扩展负责人确认；
- 每个成员至少一次实质代码/测试改动和一次他人改动审查。

## 成员贡献与 AI 核验

| 姓名/学号 | 本人实质改动及 Commit/PR | 审查他人的改动及结果 | 关键决策与 AI 核验 |
| --- | --- | --- | --- |
| 成员A | 待填写 | 待填写 | 待填写 |
| 成员B | 待填写 | 待填写 | 待填写 |
| 成员C | 待填写 | 待填写 | 待填写 |

> 提交前把“待填写”替换为实际 Commit/PR、审查意见与处理结果。
> 使用 AI 的建议需写明是否采用、如何验证；未使用写“未使用”。

## 许可与已知限制

项目使用 **MIT License**，见 [LICENSE](LICENSE)。选择理由：核心为自研 Python 代码，
MIT 简洁、便于课程展示与后续复用；第三方/来源代码的许可见
[THIRD_PARTY.md](THIRD_PARTY.md)。

已知限制：

- Milvus 主检索已接入 dense + BM25；关键词回退仍是词面检索，同义改写召回有限；
- 本地嵌入首次运行需下载 bge 模型；换模型必须新建同维度集合；
- PDF/扫描件依赖外部 docparse 与 Ollama，未配置时返回
  `CONNECTOR_UNAVAILABLE`，不影响 TXT/Markdown；
- 中文分词为单字 + 二元组，未做领域词典；
- 关键词回退索引为本地 JSON，适合课程规模；Milvus 主路径由 Docker 卷持久化；
- 生成式问答需要外部 LLM API 或本地 Ollama；未实现流式输出、多轮记忆；
- 未实现用户系统、权限、在线编辑、插件市场或热加载。

## 已有起点与本次增量

本项目基于作者此前的多模态文档入库工程（DocFlow）演化：

- 复用/改编：统一文档模型、切块、关键词索引、插件注册、Milvus 检索思路、docparse 封装；
- 本次新增：课程项目 CLI、TXT/Markdown Connector 契约与实现、错误模型、测试、
  CI、设计文档、release 模板；
- 原工程的 Airflow 编排、邮件通知、人工复核未纳入本项目。

来源与许可明细见 [../已有资源/文档/06-来源许可与依赖.md](../已有资源/文档/06-来源许可与依赖.md)
与 [THIRD_PARTY.md](THIRD_PARTY.md)。
