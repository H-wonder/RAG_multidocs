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

### P0：提交前必须完成

| ID | 任务 | 建议负责人 | 依赖 | 验收标准 |
| --- | --- | --- | --- | --- |
| P0-1 | 真实 PDF 解析验收 | B（扩展/多模态） | docparse 环境、Ollama | 至少 1 个原生 PDF + 1 个扫描件跑通；记录页数、chunks、是否需 OCR；失败行为符合 `CONNECTOR_UNAVAILABLE/PARSE_ERROR` |
| P0-2 | 真实 embedding 验收 | C（索引/发布） | Milvus 2.5+、bge 模型 | 用 `local` 或 `dashscope` 入库；`search` 命中带 `source/page/section_path`；维度与集合一致 |
| P0-3 | 真实问答验收 | C + A | 任一 chat Key 或 Ollama | `rag-cli ask` 返回答案 + `[n]` 来源；无证据时不调用 LLM；错误 Key 有明确提示 |
| P0-4 | CI integration 实际运行 | C | GitHub Actions | Actions 页面出现 integration job 的绿色 run；把 run 链接/编号写进 README/RELEASE |
| P0-5 | 成员贡献与 Review 记录 | 全员 | Git 历史 | README 成员表填真实 Commit/PR；至少每人一次实质改动 + 一次具体 Review |
| P0-6 | 设计文档最终同步 | A | 代码/接口稳定 | `docs/DESIGN.docx` 与当前实现一致；含架构图、流程图、扩展接入示例 |
| P0-7 | 发布 Tag/发行物 | C | P0-1~P0-5 | 工作区干净、测试通过、发行 ZIP 可解压运行；`release/RELEASE.md` 记录 Commit/Tag/SHA-256 |
| P0-8 | Milvus 数据卷问题处理 | C | 现有 `milvus-standalone` | 说明 embedded etcd 重启后 `node not match` 的处理方式；新环境可重复执行 |

### P1：时间允许时做，直接影响完成度

| ID | 任务 | 建议负责人 | 验收标准 |
| --- | --- | --- | --- |
| P1-1 | `models test` 命令 | A | `rag-cli models test --kind chat/embedding` 能实际调用并返回耗时/模型名；无 Key 时提示环境变量 |
| P1-2 | 嵌入维度自动探测 | A | 指定本地模型但不填 dim 时自动读取模型维度；维度与集合不一致时拒绝写入并给出提示 |
| P1-3 | 集合命名带 model/dim | C | 换 embedding 模型自动使用新集合，避免旧向量混用；文档说明 |
| P1-4 | 问答质量与引用格式 | C | 检查引用编号与来源一一对应；无证据/矛盾证据有拒答；prompt 有长度上限 |
| P1-5 | PDF 样例与契约测试 | B | 新增 `samples/pdfs/` 与测试；至少一个原生 PDF 的 IR 映射/切块断言；扫描件可标记 integration |
| P1-6 | Mock LLM HTTP 测试 | A | 用本地 mock 测试 `OpenAICompatibleChat` 请求体/错误/重试；CI 不依赖真实 Key |
| P1-7 | CLI 错误与日志 | C | 所有失败有 `CODE: message`；不打印 Key/全文；超时可配置 |
| P1-8 | CI 缓存/超时/失败日志 | C | pip cache；Milvus 启动失败时输出容器日志；integration job 超时合理 |

### P2：可选增强，不要挤占 P0

| ID | 任务 | 负责人 | 说明 |
| --- | --- | --- | --- |
| P2-1 | 简易 Web 检索页 | C | FastAPI + 单页：连接器列表、导入表单、检索/问答结果带来源；复用 Core，不写第二套业务逻辑 |
| P2-2 | HTML Connector | B | 与 Markdown 二选一即可；不要同时做两套扩展机制 |
| P2-3 | SimHash 去重 | A | 参考 `已有资源/代码/dedup.py`；重复文档不入索引，保留状态/原因 |
| P2-4 | 增量清单 | A | SQLite manifest；内容变化只升 version，删除文件清理索引 |
| P2-5 | 查询扩写/重排 | C | 可选接入 DashScope/DeepSeek；失败静默降级，不影响主查询 |
| P2-6 | 文档截图高亮 | B | 用 docparse bbox 裁原文图；至少保留 page + bbox 字段 |

**不要做**：用户体系、权限、多租户、插件市场、热加载、复杂 Agent、跨组部署。这些不增加课程评分，且会稀释 P0。

## 3. 按角色的工作包

### A：核心/契约/模型层

**目标**：让接口稳定、模型选择可验证、错误可解释。

1. **Provider 与模型校验**
   - `models test`：embedding/chat 实际调用；输出 provider/model/耗时；失败返回非 0；
   - 维度自动探测：本地模型未填 dim 时读取 `get_sentence_embedding_dimension()`；
   - 维度不一致时拒绝写入 Milvus，提示新建集合；
   - 配置校验：未知 provider、空 Key、非法 URL 统一报错。
2. **问答契约**
   - prompt 证据长度上限、top-k 上限；
   - 引用编号与 `citations` 一致性测试；
   - 无证据/矛盾证据拒答测试；
   - 真实 LLM 错误（401/429/超时）转成 `LLM_ERROR`，不泄露 Key。
3. **测试**
   - mock HTTP 测 `OpenAICompatibleChat`；
   - provider 选择/环境变量 Key 测试；
   - 审查 B 的扩展是否只依赖公开契约。

**验收**：`pytest -m "not integration"` 全绿；`ruff` 通过；`models test` 在至少一个 provider 上成功（无 Key 时明确失败）。

### B：独立扩展/多模态

**目标**：证明扩展可插拔，并补齐多模态真实链路。

1. **Markdown 扩展加固**
   - 标题/空文件/编码错误/空目录测试；
   - 禁用扩展开关与“核心源码不变”的 diff 记录；
   - 扩展 README 保持与 manifest/接口一致。
2. **PDF/扫描件验收**
   - 准备 2 个样例：原生文本 PDF、扫描件/图片；
   - 原生 PDF：验证页数、`section_path`、表格块；
   - 扫描件：启动 Ollama + docparse，记录 OCR 耗时/失败模式；
   - 结果写入 README/设计文档“已知限制”。
3. **可选 HTML Connector**（P2）
   - 只在 Markdown 扩展稳定后做；复用同一 manifest/adapter 契约。

**验收**：`rag-cli ingest --source pdf ...` 成功/失败行为可复现；扩展接入前后 `git diff -- src/rag_multidocs` 为空；至少一条契约/集成测试。

### C：索引/CLI/CI/发布

**目标**：让 Milvus、问答、CI 和最终交付可演示、可追溯。

1. **Milvus + 真实 embedding**
   - 启动 Milvus 2.5+（Docker）；
   - 用 `local` bge 或 `dashscope` 入库、检索；
   - 换模型时新建集合；记录维度；
   - 处理现有 `milvus-standalone` embedded etcd `node not match`：清理旧卷重建或换新容器。
2. **问答验收**
   - 设置 `DEEPSEEK_API_KEY`/`DASHSCOPE_API_KEY`/`OPENAI_API_KEY` 或本地 Ollama；
   - `rag-cli ask` 检查答案、引用、无证据拒答；
   - 记录一次成功运行的命令和输出（脱敏）。
3. **CI**
   - 确认 unit + integration 两个 job 实际运行；
   - integration 失败时输出 Milvus 日志；
   - 把 run 链接/编号写进 README/RELEASE。
4. **发布**
   - 生成发行 ZIP/可运行目录；
   - 新目录解压按 README 跑通；
   - Tag + `release/RELEASE.md`（Commit/Tag/SHA-256/结果）。

**验收**：Actions 两个 job 有真实记录；Release 记录完整；新目录可运行。

## 4. 建议时间安排（按 7 天缓冲）

| 天 | 工作 | 负责人 |
| --- | --- | --- |
| D1 | P0-1 PDF 样例 + 真实解析；P0-2 Milvus bge 入库 | B / C |
| D2 | P0-3 真实问答；修错误提示/引用 | C / A |
| D3 | P0-4 CI integration 实跑；P0-8 Milvus 数据卷处理 | C |
| D4 | P1-1/P1-2 models test/维度探测；P1-5 PDF 测试 | A / B |
| D5 | P0-5 Review/成员表；P0-6 设计文档同步 | 全员 / A |
| D6 | P0-7 发行物、Tag、RELEASE；新目录自检 | C |
| D7 | 缓冲：修 CI/文档/演示；不要新增功能 | 全员 |

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
