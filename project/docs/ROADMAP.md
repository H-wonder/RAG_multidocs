# 后续优化方向与组内分工

> 面向项目组其他成员。用途：明确当前已经做到哪一步、还缺什么、谁可以接手、每项工作的验收标准。
> 当前分支：`main`；基线提交见仓库最新 commit。
> 相关文档：[README](../README.md)、[DESIGN.md](DESIGN.md)、[环境搭建](../../已有资源/文档/07-环境搭建与依赖.md)。

## 1. 当前基线（2026-11）

### 1.1 已实现

| 能力 | 状态 | 位置/说明 |
| --- | --- | --- |
| 统一 `Document` / `Chunk` 模型 | 完成 | `src/rag_multidocs/model.py` |
| 内置 TXT Connector | 完成 | `src/rag_multidocs/connectors/txt.py` |
| 内置 PDF/扫描件/Office Connector | 代码完成，真实 OCR 待验证 | `connectors/pdf.py`，子进程调用 docparse |
| 独立 Markdown 目录 Connector | 完成 | `connectors/connector_markdown/`，manifest + adapter |
| Connector 契约/注册/错误模型 | 完成 | `contracts.py`、`registry.py` |
| Milvus 混合检索（dense + BM25） | 完成并通过本地集成测试 | `index_milvus.py`、`milvus_store.py` |
| 关键词 BM25 回退 | 完成 | `index.py`，Milvus 不可用时自动回退 |
| Embedding provider 选择 | 完成 | `providers.py`、`embedding.py`；local/hashing/dashscope/openai/ollama |
| 检索增强问答 `rag-cli ask` | 完成，真实 API 待验证 | `llm.py`；证据 prompt + `[n]` 引用 |
| Chat provider 选择 | 完成 | deepseek/dashscope/openai/ollama；API Key 仅环境变量 |
| CLI | 完成 | `ingest/search/ask/models/docs/connectors/stats` |
| 单元测试 | 25 passed | `pytest -q -m "not integration"` |
| Milvus 集成测试 | 1 passed（本地隔离容器） | `pytest -q -m integration` |
| CI | unit job 已真实成功；integration job 待确认 | 仓库根 `.github/workflows/ci.yml` |
| README/设计文档/环境文档 | 已完成，需随版本更新 | `README.md`、`docs/DESIGN.docx` |

### 1.2 明确未完成

- 真实 PDF / 扫描件 / OCR 全链路未在项目中跑过；
- 真实 bge 嵌入 + 真实 DeepSeek/DashScope/OpenAI/Ollama 问答未做人工验收；
- CI integration job 尚未看到实际绿色 run；
- 成员贡献表、PR/Review 记录、最终 Tag/发行物未完成；
- Web 界面未实现（可选）；
- 重排、查询扩写、增量清单、SimHash 去重未接入主流程。

## 2. 优先级总览

> 分工前提：A（项目负责人）已完成核心、Connector 契约、Milvus、provider、问答和文档骨架，
> 不再承担新增功能；剩余 P0/P1 由 **B、C 两位成员**分担，A 只做最终合并、Release 确认和必要答疑。

### P0：提交前必须完成

| ID | 任务 | 负责人 | 依赖 | 验收标准 |
| --- | --- | --- | --- | --- |
| P0-1 | 真实 PDF / 扫描件 / OCR 验收 | **B** | docparse 环境、Ollama | 至少 1 个原生 PDF + 1 个扫描件跑通；记录页数、chunks、是否需 OCR；失败行为符合 `CONNECTOR_UNAVAILABLE/PARSE_ERROR` |
| P0-2 | 真实 embedding 验收（bge/DashScope） | **C** | Milvus 2.5+、模型或 Key | `rag-cli ingest` 入库；`search` 命中带 `source/page/section_path`；维度与集合一致 |
| P0-3 | 真实 LLM 问答验收 | **C**（B 复核引用） | 任一 chat Key 或 Ollama | `rag-cli ask` 返回答案 + `[n]` 来源；无证据不调用 LLM；错误 Key 有明确提示 |
| P0-4 | CI integration 实际运行 | **C** | GitHub Actions | Actions 出现 integration job 绿色 run；run 链接/编号写入 README/RELEASE |
| P0-5 | 成员贡献与 Review 记录 | **B、C**（A 已有部分只确认） | Git 历史 | 每人填自己的 Commit/PR 和一次具体 Review；不能只有 LGTM |
| P0-6 | 设计文档最终同步 | **B**（C 复核） | 代码/接口稳定 | `docs/DESIGN.docx` 与实现一致；含架构图、流程图、扩展接入示例 |
| P0-7 | 发布 Tag/发行物 | **C** | P0-1~P0-5 | 工作区干净、测试通过、发行 ZIP 可解压运行；`release/RELEASE.md` 记录 Commit/Tag/SHA-256 |
| P0-8 | Milvus 数据卷问题处理 | **C** | 现有 `milvus-standalone` | 说明 embedded etcd 重启后 `node not match` 的处理方式；新环境可重复执行 |

### P1：P0 完成后再做，直接影响完成度

| ID | 任务 | 负责人 | 验收标准 |
| --- | --- | --- | --- |
| P1-1 | `models test` 命令 | **B** | `rag-cli models test --kind chat/embedding` 实际调用并返回耗时/模型名；无 Key 时提示环境变量 |
| P1-2 | 嵌入维度自动探测 | **C** | 本地模型不填 dim 时自动读取维度；维度与集合不一致时拒绝写入并提示 |
| P1-3 | 集合命名带 model/dim | **C** | 换 embedding 模型自动使用新集合；文档说明 |
| P1-4 | 问答质量与引用测试 | **B** | 引用编号与来源一一对应；无证据/矛盾证据拒答；prompt 有长度上限 |
| P1-5 | PDF 样例与契约测试 | **B** | 新增 `samples/pdfs/`；原生 PDF 的 IR 映射/切块断言；扫描件可标 integration |
| P1-6 | Mock LLM HTTP 测试 | **B** | 用本地 mock 测试 `OpenAICompatibleChat` 请求体/错误/重试；CI 不依赖真实 Key |
| P1-7 | CLI 错误与日志 | **C** | 所有失败有 `CODE: message`；不打印 Key/全文；超时可配置 |
| P1-8 | CI 缓存/超时/失败日志 | **C** | pip cache；Milvus 启动失败输出容器日志；integration job 超时合理 |

### P2：可选增强，不强制、不得挤占 P0

| ID | 任务 | 建议 | 说明 |
| --- | --- | --- | --- |
| P2-1 | 简易 Web 检索页 | C（有余力） | FastAPI + 单页：连接器列表、导入、检索/问答结果带来源；复用 Core |
| P2-2 | HTML Connector | B（有余力） | 与 Markdown 二选一；不要同时做两套扩展机制 |
| P2-3 | SimHash 去重 | B/C 协商 | 参考 `已有资源/代码/dedup.py`；重复文档不入索引 |
| P2-4 | 增量清单 | B/C 协商 | SQLite manifest；内容变化升 version，删除文件清理索引 |
| P2-5 | 查询扩写/重排 | C（有余力） | 可选接入 DashScope/DeepSeek；失败静默降级 |
| P2-6 | 文档截图高亮 | B（有余力） | 用 docparse bbox 裁原文图；至少保留 page + bbox |

**不要做**：用户体系、权限、多租户、插件市场、热加载、复杂 Agent、跨组部署。
这些不增加课程评分，且会稀释 P0。

## 3. 剩余两位成员的工作包

### B：多模态 / 独立扩展 / 文档验收

**你接手的目标**：证明多模态和扩展真实可用，并让设计文档与实现一致。

1. **P0-1 PDF/扫描件验收**
   - 准备 `samples/pdfs/`：一个原生文本 PDF、一个扫描件/图片；
   - 原生 PDF：`rag-cli ingest --source pdf ...`，核对页数、`section_path`、表格块；
   - 扫描件：启动 Ollama + docparse，记录 OCR 耗时、失败模式、是否需要 `region`；
   - 失败路径：不存在的 PDF、损坏 PDF、docparse 未配置。
2. **P0-6 设计文档同步**
   - 代码/接口有变化时更新 `DESIGN.md` 并重新导出 `DESIGN.docx`；
   - 至少保留两张图：架构图、关键流程图；
   - 扩展接入示例与 `manifest.json` 保持一致。
3. **P1-1 / P1-4 / P1-5 / P1-6**
   - `models test`：至少测一个 embedding + 一个 chat provider；无 Key 时给出明确提示；
   - 问答引用测试：引用编号/来源对应；无证据拒答；prompt 长度上限；
   - PDF 契约测试：IR → Document/Chunk 的页码、章节、表格断言；
   - mock LLM HTTP：请求体、401/429/超时/畸形响应。
4. **可选 P2-2 / P2-6**：HTML Connector、bbox 截图高亮；只在 P0/P1 完成后做。

**B 的验收**：PDF 成功/失败可复现；扩展接入前后 `git diff -- src/rag_multidocs` 为空；
`pytest -m "not integration"` 全绿；文档与实现一致。

### C：索引 / 问答 / CI / 发布

**你接手的目标**：让 Milvus、真实模型、CI 和最终交付可演示、可追溯。

1. **P0-2 真实 embedding**
   - 启动 Milvus 2.5+（Docker）；
   - 用 `local` bge 或 `dashscope` 入库；换模型新建集合；
   - 记录维度与 collection；`search` 命中带来源。
2. **P0-3 真实问答**
   - 设置 `DEEPSEEK_API_KEY`/`DASHSCOPE_API_KEY`/`OPENAI_API_KEY` 或本地 Ollama；
   - `rag-cli ask` 检查答案、引用、无证据拒答；
   - 记录一次成功运行的命令/输出（脱敏）。
3. **P0-4 / P0-8 CI 与 Milvus**
   - 确认 unit + integration 两个 job 实际运行；
   - integration 失败时输出 Milvus 日志；
   - 现有 `milvus-standalone` embedded etcd `node not match`：备份后重建卷，或用隔离容器；
   - 把 run 链接/编号写进 README/RELEASE。
4. **P0-7 发布**
   - 生成发行 ZIP/可运行目录；
   - 新目录解压按 README 跑通；
   - Tag + `release/RELEASE.md`（Commit/Tag/SHA-256/结果）。
5. **P1-2 / P1-3 / P1-7 / P1-8**
   - 维度自动探测、集合命名、CLI 错误/日志、CI 缓存/超时/失败日志。
6. **可选 P2-1 / P2-5**：Web、查询扩写/重排；只在 P0/P1 完成后做。

**C 的验收**：Milvus/问答真实链路可复现；Actions 两个 job 有真实记录；
Release 记录完整；新目录可运行。

### 交叉 Review 与合并

- B 的 PDF/文档改动由 C 审查；C 的 Milvus/CI/发布改动由 B 审查；
- A 只做最终合并确认和必要的接口答疑，不再承担新增功能；
- 每条 Review 至少指出一个行为、边界、失败处理或测试问题，不能只有 LGTM；
- 合并前确认工作区干净、相关测试通过、文档已同步。

## 4. 建议时间安排（按 7 天缓冲）

| 天 | 工作 | 负责人 |
| --- | --- | --- |
| D1 | P0-1 PDF 样例 + 真实解析；P0-2 Milvus bge 入库 | B / C |
| D2 | P0-3 真实问答；P1-1 models test | C / B |
| D3 | P0-4 CI integration；P0-8 Milvus 数据卷 | C |
| D4 | P0-6 设计文档；P1-4/P1-5 PDF/引用测试 | B |
| D5 | P0-5 成员表/Review；P1-2/P1-3 维度/集合 | B / C |
| D6 | P0-7 发行物、Tag、RELEASE；新目录自检 | C |
| D7 | 缓冲：修 CI/文档/演示；不新增 P2 | B / C |

## 5. 风险与处理

| 风险 | 影响 | 处理 |
| --- | --- | --- |
| 现有 Milvus embedded etcd 重启后 `node not match` | 无法建新集合 | 用新数据卷/新容器；不要删旧数据，先备份；集成测试用隔离容器 |
| 真实 API Key 不可用/限流 | 问答演示失败 | 用 Ollama 本地模型；`--no-generate` 退回检索；错误提示明确 |
| docparse/Ollama 依赖重 | PDF 验收慢 | 先跑原生 PDF（无需模型）；扫描件单独安排 |
| bge 模型下载失败 | 本地 embedding 不可用 | 设 `HF_ENDPOINT=https://hf-mirror.com`；或先用 `hashing` 验证链路 |
| 维度不一致 | Milvus 写入失败/检索错误 | `models use` 重置默认；换模型新建集合；写测试 |
| CI Docker 拉取慢/启动失败 | integration job 红 | 增加等待循环/日志；失败时保留容器日志 |
| 时间不足 | 交付风险 | 严格 P0；P1/P2 可砍；先保证 unit + 关键词回退可演示 |

## 6. 课程要求对照

| 任务书要求 | 当前状态 | 还需做什么 |
| --- | --- | --- |
| 一个核心场景 | 已实现 TXT/Markdown/PDF 接入 + 检索/问答 | P0-1/P0-3 真实验收 |
| 一种扩展机制 + 一个独立扩展 | Markdown manifest/adapter | B 补契约测试与禁用验证记录 |
| 扩展接入不改核心 | 已满足 | 保留 `git diff -- src/rag_multidocs` 记录 |
| 正常/边界/失败测试 | 25 unit + 1 integration | P0-1/P0-4/P1-5/P1-6 |
| 构建/测试命令 + CI 真实成功 | unit 已成功 | integration 实跑并记录 |
| 基础检查 | ruff 已通过 | 保持 |
| 三人协作/Review | 未完成 | P0-5 |
| Tag/发行物/RELEASE | HOLD | P0-7 |
| 设计文档 | 17 页，已含图 | P0-6 同步最终版本 |
| 许可/README/来源 | 已有 MIT/THIRD_PARTY/来源说明 | 提交前核对 |

## 7. 完成定义（Definition of Done）

一个任务只有同时满足以下条件才算完成：

1. 代码/测试已提交，工作区干净；
2. 相关测试通过（unit 或 integration），并有命令/结果记录；
3. 错误路径有明确反馈；
4. 文档（README/DESIGN/ROADMAP）与实现一致；
5. 至少一名其他成员 Review，意见已处理；
6. 涉及用户可见行为时，有可复现的 CLI 命令和输出。

提交前最后检查：

```bash
cd project
pytest -q -m "not integration"
ruff check src tests connectors
MILVUS_URI=http://localhost:19530 pytest -q -m integration
git status
```

以上全部通过、`release/RELEASE.md` 从 HOLD 改为 RELEASE 后，再提交最终 ZIP。
