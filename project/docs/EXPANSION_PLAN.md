# RAG Multidocs 拓展方案

> 面向项目组后续开发。目标不是增加代码行数，而是把“多模态接入 → 可核对检索 → 可评估问答”做深，
> 同时把 CLI 从“能跑”提升到“好用、可演示、可排错”。
>
> 当前基线：约 6,800 行 Python（含测试）、65 个单元测试、1 个 Milvus 集成测试、
> 20 页设计文档。已有 TXT/Markdown/HTML/PDF 插件、Milvus 混合检索、关键词回退、
> provider 选择、检索增强问答、样例集和 CI。
>
> 相关文档：[README](../README.md)、[USAGE](USAGE.md)、[DESIGN.md](DESIGN.md)、
> [ROADMAP.md](ROADMAP.md)、[环境搭建](../../已有资源/文档/07-环境搭建与依赖.md)。

## 0. 判断与原则

### 0.1 不要以行数为目标

课程任务书明确不以代码量、页面数或技术栈数量作为主要评价依据。继续堆 CRUD、Web 页面或
重复封装，不会提高完成质量。更值得投入的是：

1. 多模态链路的真实深度：扫描 PDF、版面、表格、bbox、OCR 质量；
2. 检索/问答的可评估性：命中率、引用准确率、拒答；
3. 交互的可解释性：进度、错误恢复、引用定位、配置诊断；
4. 交付证据：测试、CI、Release、成员 Review。

### 0.2 两条主线

| 主线 | 目标 | 主要产出 |
| --- | --- | --- |
| A. RAG 链路 | 让文档接入、解析、切块、索引、检索、问答形成可核对闭环 | OCR/版面、bbox 引用、增量去重、重排、评估集、真实模型验收 |
| B. 交互体验 | 让用户知道“现在在做什么、为什么失败、答案来自哪里” | doctor、进度、模型选择、文档浏览、引用截图、REPL/CLI 拆分、可选 Web |

两条线可以并行，但 A 的 bbox/元数据是 B 的引用定位的前置。

### 0.3 目标代码结构

不要一次性大重构。按模块逐步迁移，保持测试通过。

**第一轮机械拆分已完成**（2026-11-24）：

```text
src/rag_multidocs/
├── model.py            # 公共文档模型（插件契约）
├── contracts.py        # 公共 Connector 契约（插件契约）
├── config.py           # Settings
├── credentials.py      # Key 文件
├── providers/          # presets.py / embedding.py / discovery.py
├── ingestion/          # chunking.py / service.py
├── connectors/         # registry.py / txt.py
├── index/              # keyword.py / milvus.py / milvus_store.py
├── retrieval/          # search.py
├── qa/                 # llm.py
└── cli/                # app.py / repl.py / wizard.py
```

后续可继续拆：

```text
├── core/               # 若需要，把 model/contracts/errors 收进去
├── ocr/                # docparse_client.py / ir_mapping.py / quality.py
├── config/             # settings.py / credentials.py
├── cli/commands/       # 子命令拆分
└── web/                # app.py（可选）
```

外部插件仍放在 `project/connectors/`，只依赖 `rag_multidocs.model` 与
`rag_multidocs.contracts`，不进入核心包。

---

# A. RAG 链路

## A1. 多模态接入与 OCR 深度

### A1.1 拆分 docparse 客户端与 IR 映射

**现状**：`connectors/connector_pdf/adapter.py` 同时负责子进程、错误转换、IR 映射、
Connector 适配，约 350 行。

**改造**：

```text
connectors/connector_pdf/
├── adapter.py          # load(payload, context)，只做编排
├── docparse_client.py  # subprocess、超时、重试、错误
├── ir_mapping.py       # IR → Document / Chunk 元数据
└── README.md
```

**验收**：

- `adapter.py` 不直接构造 `subprocess`；
- 子进程超时、退出码非 0、无 IR、Ollama 不可达都有独立错误；
- 现有 PDF 假 docparse 测试不改行为。

**估算**：1–2 天。

### A1.2 扫描 PDF 的版面分析验收

**现状**：已用 `samples/campus-corpus/pdf/scanned-library-notice.png` 实测 OCR；
但没有多页扫描 PDF、双栏、表格/票据的验收。

**改造**：

- 新增 `samples/scanned-pdf/`：
  - `native-2pages.pdf`：原生文本；
  - `scanned-2pages.pdf`：扫描页；
  - `scanned-table.pdf`：含表格；
  - `two-column.pdf`：双栏；
  - `bad-text-layer.pdf`：坏文本层。
- 用 `--mode auto` 和 `--mode region` 分别跑；
- 记录页数、`native/ocr/failed` 页数、耗时、表格块数、`needs_review`；
- 失败路径：损坏 PDF、docparse 未配置、Ollama 未启动。

**验收**：

- `rag-cli ingest --source pdf --path ... --mode region`；
- `rag-cli search "表格内容"` 命中 `content_type=table`；
- `docs list` 能看到页数和 chunk 数；
- 测试至少覆盖：页数、表格块、OCR 失败告警。

**估算**：3–5 天（含样本制作）。

### A1.3 bbox 引用链路

**现状**：`ingestion/chunking.py` 已把 `bbox` 写入 `Chunk`，Milvus schema 也存了 `bbox`/
`page_end`，但 `MilvusIndex.search()` 和 `KeywordIndex.search()` 返回的 hit
没有 `bbox` / `page_end`。CLI 和问答引用只能显示 `source/page/section_path`。

**改造**：

1. `search()` 返回 `bbox`、`page_end`；
2. `rag-cli search --json` 带 bbox；
3. `rag-cli ask` 的 citations 带 bbox；
4. 新增 `rag-cli cite`：根据 `source/page/bbox` 用 PyMuPDF 裁剪页面并输出 PNG；
5. `docs show <doc_id>` 显示文档元数据、页数、OCR 警告。

**验收**：

- 扫描 PDF 的检索结果带非空 bbox；
- `rag-cli cite` 输出图片，bbox 与命中块一致；
- 问答引用能显示页码和 bbox；
- 测试断言 bbox 从 Chunk → index → search → citation 不丢失。

**估算**：3–5 天。

### A1.4 OCR 质量与失败恢复

**改造**：

- 把 docparse IR 的 `needs_review`、`warnings`、每页 `mode` 暴露到 `IngestReport`；
- `rag-cli ingest` 输出：

```text
pages=12 native=9 ocr=2 blank=1 needs_review=2
warnings: low_confidence(2), page_failed(1)
```

- `--strict`：有 high/medium warning 时返回非 0；
- 失败页不拖垮整篇文档，但必须显式列出。

**验收**：损坏页/空白页/低置信页有明确输出；测试覆盖 `needs_review`。

**估算**：2–3 天。

### A1.5 图片目录与多文件入库

**改造**：

- `connector_pdf` 支持图片目录（已有 `SUPPORTED`，补测试）；
- 多张图片逐张 OCR，失败继续；
- 用 `sha256` 去重，避免同一图片重复入库；
- 输出每张图的状态。

**验收**：3 张 PNG/JPG，一张坏图；成功 2 张、失败 1 张有独立 issue。

**估算**：1–2 天。

## A2. 切块与文档元数据

### A2.1 结构化切块策略

**改造**：

- 按 `content_type` 区分：`text` / `table` / `formula` / `figure` / `page_text`；
- 表格整块保留；公式带上下文；figure 关联 caption；
- 页眉/页脚/重复标题去重；
- `section_path` 保留层级；
- 增加 `min_tokens` / `max_tokens` / `overlap` 配置。

**验收**：

- 表格查询命中 table chunk；
- 双栏 PDF 的阅读顺序不乱；
- 同一文档重复导入 chunk 数不翻倍。

**估算**：2–3 天。

### A2.2 元数据与过滤

**改造**：

- Chunk 元数据补：`page_end`、`bbox`、`block_ids`、`ocr_mode`、`needs_review`；
- 检索过滤：`--doc-id`、`--page`、`--content-type`；
- `docs list/show` 展示元数据。

**验收**：`rag-cli search "退货" --content-type table --page 3` 只返回目标块。

**估算**：2 天。

## A3. 索引与检索质量

### A3.1 维度与集合安全

**现状**：`ensure_collection()` 发现集合已存在就直接复用，不检查维度；换 embedding
模型后可能写入失败。

**改造**：

- 本地模型自动读取真实维度；
- 集合名带 `provider/model/dim` 或写入 collection metadata；
- 维度不一致时拒绝写入，提示新集合名；
- `models use` 切换时提示目标集合。

**验收**：换 bge 512 → DashScope 1024 自动使用新集合；混用时明确报错。

**估算**：2–3 天。

### A3.2 重排与查询扩写

**改造**：

- 可选 rerank：DashScope `gte-rerank-v2` 或本地 cross-encoder；
- 查询扩写可配置，失败静默降级；
- 记录 rerank 前后分数/耗时；
- 评测集对比 hit@k / MRR。

**验收**：开关可配置；关闭时行为不变；失败不影响基础检索。

**估算**：3–4 天。

### A3.3 增量入库与去重

**改造**：

- 接入 `已有资源/代码/manifest_store.py`：记录 doc_id、sha256、version、chunks；
- 内容未变跳过；变化则先删后插；
- 接入 `dedup.py`：SimHash 近重复文档不入索引；
- `rag-cli ingest --incremental` / `--force`。

**验收**：重复导入不增加 chunk；修改文件只更新该文档；删除文件可清理索引。

**估算**：3–4 天。

## A4. 问答与评估

### A4.1 引用校验

**改造**：

- 生成答案后校验每个 `[n]` 是否对应证据；
- 引用编号越界/无来源时标红或拒绝；
- 无证据时不调用 LLM；
- prompt 设置 token 上限。

**验收**：伪造引用被检测；无证据拒答；引用编号与来源一一对应。

**估算**：2 天。

### A4.2 评估集

**改造**：

- `samples/eval/questions.json`：20–30 个问题，标注期望 source/page；
- 指标：hit@1/hit@3、citation precision、refusal accuracy、平均耗时；
- `rag-cli eval` 输出 JSON/Markdown；
- CI 可跑离线 keyword 评测，Milvus 评测标 integration。

**验收**：一条命令输出指标；改动检索/切块后能对比。

**估算**：2–3 天。

### A4.3 真实 provider 验收

- DeepSeek / DashScope / OpenAI / Ollama 各至少一次真实调用；
- 流式输出、超时、401/429、错误模型名；
- 记录命令和脱敏输出。

**估算**：2 天。

## A5. 运维与可观测性

- `rag-cli doctor`：检查 conda、Milvus、Ollama、docparse、Key、集合维度；
- 结构化日志：request id、耗时、backend、模型名，不打印 Key/全文；
- Milvus 集合备份/恢复/删除确认；
- CI：unit + Milvus integration + OCR mock；失败保留容器日志。

**估算**：2–3 天。

---

# B. 交互体验

## B1. 首次使用与配置

### B1.1 `rag-cli doctor`

一条命令输出：

```text
[PASS] conda env rag
[PASS] Milvus :19530
[WARN] Ollama 未运行（扫描件 OCR 不可用）
[PASS] docparse python
[PASS] embedding local dim=512
[WARN] Milvus collection rag_chunks dim=1024 != embedder dim=512
建议：设置 MILVUS_COLLECTION=rag_chunks_bge512
```

**验收**：缺依赖/服务/Key/维度不匹配都有可执行建议。

**估算**：2–3 天。

### B1.2 模型选择与测试

已做：`/model` 无参数菜单、`models discover`、动态模型列表。

继续：

- `rag-cli models test --kind embedding/chat`：真实调用一次，输出模型名/维度/耗时；
- 无 Key、模型不存在、超时分别有明确错误；
- `/model` 菜单标注“当前 Key 可用/不可用”；
- 保存前提示将使用的 Milvus 集合。

**验收**：`models test` 对本地/远程各跑通一个；错误不泄露 Key。

**估算**：2–3 天。

### B1.3 配置诊断与迁移

- `rag-cli config show` / `config validate`；
- 检测旧变量（`PLATFORM_INDEX_BACKEND`、`CONNECTORS_DIR` 等）并提示迁移；
- 维度/集合不匹配时给出新集合名；
- 提供 `config profiles`：`offline`（keyword）、`local`（bge + Ollama）、`cloud`（DashScope/DeepSeek）。

**估算**：2–3 天。

## B2. CLI / REPL

### B2.1 拆分模块

`cli/app.py` 859 行、`cli/repl.py` 712 行，建议拆成：

```text
cli/
├── main.py          # argparse + dispatch
├── commands/        # ingest/search/ask/models/docs/connectors/keys/stats
├── repl/            # session.py commands.py model_picker.py render.py
└── render.py        # 命中、引用、错误、进度
```

**验收**：行为不变，测试不改；`rag-cli` 入口不变。

**估算**：3–5 天（机械重构）。

### B2.2 导入进度与批处理

- 目录导入显示 `[3/12] file.pdf ... OCR 12s`；
- 每页 OCR 进度（如果 docparse 支持 stdout 事件）；
- `Ctrl-C` 取消当前文件，不退出整个 REPL；
- `--continue-on-error` / `--strict`；
- 失败文件汇总。

**估算**：2–3 天。

### B2.3 交互命令增强

- `/last` 查看上次命中；
- `/open <n>` 打开第 n 条来源（打印路径/页码/bbox）；
- `/cite <n>` 裁剪对应页面；
- `/history` / `/clear`；
- `/docs <doc_id>` 查看文档详情；
- 分页显示长结果；
- shell completion（bash/zsh）。

**估算**：3–4 天。

### B2.4 错误与恢复

- 所有错误统一 `CODE: message` + 建议命令；
- LLM 401/429/超时：保留检索证据，提示重试/换模型；
- Milvus/embedding 失败：明确显示 `backend=keyword`；
- 不打印完整正文/Key。

**估算**：2 天。

## B3. 引用与文档浏览

### B3.1 `docs` / `chunks` 浏览

```bash
rag-cli docs list
rag-cli docs show <doc_id>
rag-cli chunks list --doc-id <doc_id> --page 2
rag-cli chunks show <chunk_id>
```

显示：source、page、section、content_type、bbox、ocr_mode、warnings。

**估算**：2–3 天。

### B3.2 `cite` 截图

```bash
rag-cli cite --source samples/scanned.pdf --page 3 --bbox 100,200,900,500
# -> /tmp/citation-xxxx.png
```

用 PyMuPDF 渲染页面并裁剪 bbox；可选 `--open`。

**验收**：bbox 与命中块一致；扫描件/原生 PDF 都能裁。

**估算**：2 天。

### B3.3 终端高亮

- `search` 命中词在文本中高亮；
- `ask` 的 `[n]` 与来源列表颜色对应；
- `NO_COLOR` / 非 TTY 自动关闭。

**估算**：1–2 天。

## B4. Web（可选，P2）

FastAPI + 单页：

- 连接器列表、导入、检索/问答；
- 结果卡片显示 source/page/section/bbox；
- 点击引用显示页面截图；
- 复用 Core/Service，不重新实现业务逻辑。

**估算**：5–8 天。只在 P0/P1 完成后做。

## B5. 演示与文档

- `rag-cli demo --offline`：TXT/Markdown/HTML + keyword；
- `rag-cli demo --full`：PDF/OCR/Milvus/QA；
- 自动选择样例、打印预期来源；
- README/USAGE 增加截图/命令；
- `docs/DEMO.md` 维护演示脚本。

**估算**：2 天。

---

# 里程碑建议

| 里程碑 | 内容 | 依赖 | 估算 |
| --- | --- | --- | --- |
| M1 | 扫描 PDF + region 版面 + bbox 引用 + cite | docparse/Ollama | 5–7 天 |
| M2 | 维度/集合安全、增量去重、重排、评估集 | M1、Milvus | 7–10 天 |
| M3 | doctor、models test、导入进度、docs/chunks 浏览 | A 的元数据 | 6–8 天 |
| M4 | CLI/REPL 拆分、Web（可选）、Release | M1–M3 | 5–8 天 |

建议顺序：**M1 → M2 → M3 → M4**。M1 是最能提升演示效果的纵向切片。

## 分工建议

- **B（多模态/插件）**：A1、A2、B3；负责扫描 PDF、bbox、cite；
- **C（索引/问答/CI）**：A3、A4、A5、B1、B2；负责维度、重排、评估、doctor；
- **A（负责人）**：接口评审、合并、Release 确认；
- 交叉 Review：B 审 C 的索引/CI，C 审 B 的 OCR/引用。

## 不做什么

- 用户体系、权限、多租户；
- 插件市场、热加载；
- 复杂 Agent、多轮记忆作为核心卖点；
- 为凑行数复制 CRUD；
- 合并 `rag` / `docparse` 两个 conda 环境。

## 完成定义

每个任务必须同时满足：

1. 代码/测试提交，工作区干净；
2. 正常、边界/失败、扩展接入至少各有一条断言；
3. CLI 有可复现命令和输出；
4. 文档同步；
5. 至少一名其他成员 Review，意见已处理；
6. 涉及外部服务时，提供 Mock/离线路径。
