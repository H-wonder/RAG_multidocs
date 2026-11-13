# RAG Multidocs——插件化多模态文档接入与检索

面向课程资料使用者：用户指定数据源、导入文档、输入关键词，获得**带来源**的命中文本。
本项目把“接入 → 统一文档 → 切块 → 索引 → 检索”做成可扩展的插件平台：
核心只依赖 Connector 公开契约，新增数据源不需要修改核心源码。

- 仓库：<https://github.com/H-wonder/RAG_multidocs>
- 设计文档：[docs/DESIGN.md](docs/DESIGN.md)（提交版为 `docs/DESIGN.docx`）
- 已有资源与来源说明：[../已有资源/README.md](../已有资源/README.md)

## 用户、场景与范围

**谁**：需要检索课程资料的教师/学生。
**什么情况**：手头有 TXT、Markdown、PDF/扫描件等多来源资料，希望统一导入后按关键词检索，
并能看到命中来自哪个文件、哪一页、哪一节。
**输入 → 处理 → 输出**：

```text
TXT / Markdown / PDF ──► Connector ──► Document ──► Chunk ──► BM25 索引
                                                                  │
用户查询 ─────────────────────────────────────────────────────────┘
        └──► 命中文本 + source + page + section_path（可选 bbox）
```

**基础能力（内置）**：

- 统一 `Document` / `Chunk` 模型；
- 内置 TXT Connector（单文件或目录）；
- 内置 PDF/扫描件/Office Connector（通过 docparse 子进程，可选）；
- BM25 关键词索引与检索，带来源输出；
- 接入失败的结构化错误（`FILE_NOT_FOUND`、`EMPTY_SOURCE`、`PARSE_ERROR` 等）。

**独立扩展**：

- `connectors/connector_markdown/`：Markdown 目录 Connector；
- 通过 `manifest.json + adapter.py` 接入，核心源码不改；
- 禁用后 TXT/PDF 与检索仍可用。

**本次不做什么**：

- 不做用户登录、权限、多租户；
- 不做爬虫集群、插件市场、热加载；
- 不做生成式回答（只做可核对的检索证据）；
- 不把 Milvus/嵌入模型作为运行必需项（关键词索引即可完成闭环）。

## 快速开始

### 1. 安装

核心只依赖 Python 3.10+ 标准库：

```bash
cd project
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

PDF/扫描件解析需要另外准备 docparse 环境，见
[../已有资源/文档/07-环境搭建与依赖.md](../已有资源/文档/07-环境搭建与依赖.md)。

### 2. 配置

```bash
cp config.example.json config.json
# 按需修改 docparse_repo / docparse_python；不导入 PDF 时可忽略
```

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

示例输出：

```text
[1] 6.2538  chunking.md p0  §切块策略 > 按结构切 > 表格
    | 策略 | 优点 | 缺点 | ...
[2] 6.0203  chunking.md p0  §切块策略 > 按结构切 > 表格
    表格必须整块保留。按字符数切会把一行数据切成两半，检索结果无法阅读。
```

### 4. 测试

```bash
pytest -q
ruff check src tests connectors
```

## 核心与扩展

架构与接口设计见 [docs/DESIGN.md](docs/DESIGN.md)。

```text
CLI ──► Core（IngestService / SearchService / ConnectorRegistry）
             │
             ├── 内置 Connector：TXT、PDF/扫描件/Office（docparse 子进程）
             └── 独立扩展：Markdown 目录（connectors/connector_markdown）
             │
             └── KeywordIndex（BM25，本地 index.json）
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
| CLI | `ingest/search/docs/connectors` 的输出与退出码 |

CI 配置：仓库根 `.github/workflows/ci.yml`，执行 `pytest` 与 `ruff check`。

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

- BM25 是词面检索，同义改写召回有限；向量检索未纳入核心；
- PDF/扫描件依赖外部 docparse 与 Ollama，未配置时返回
  `CONNECTOR_UNAVAILABLE`，不影响 TXT/Markdown；
- 中文分词为单字 + 二元组，未做领域词典；
- 索引为本地 JSON，适合课程规模，不适合大规模并发/增量更新；
- 未实现用户系统、权限、在线编辑、插件市场或热加载。

## 已有起点与本次增量

本项目基于作者此前的多模态文档入库工程（DocFlow）演化：

- 复用/改编：统一文档模型、切块、关键词索引、插件注册、Milvus 检索思路、docparse 封装；
- 本次新增：课程项目 CLI、TXT/Markdown Connector 契约与实现、错误模型、测试、
  CI、设计文档、release 模板；
- 原工程的 Airflow 编排、邮件通知、人工复核未纳入本项目。

来源与许可明细见 [../已有资源/文档/06-来源许可与依赖.md](../已有资源/文档/06-来源许可与依赖.md)
与 [THIRD_PARTY.md](THIRD_PARTY.md)。
