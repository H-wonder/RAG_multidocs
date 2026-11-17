# 本次版本交付

- 版本及选择理由：0.1.0。可运行骨架，已接入 TXT/Markdown/PDF Connector、Milvus 混合检索与关键词回退；尚未做最终发行验证。
- 最终 Commit（完整 SHA）：待发布时填写
- Tag 及其目标 Commit（完整 SHA）：待发布
- 工作区状态、构建/测试结果及 CI 真实状态：
  - 工作区：待发布时执行 `git status`
  - 本地单元测试：`cd project && pytest -q -m "not integration"`（50 passed）
  - 本地 Milvus 集成测试：`MILVUS_URI=... pytest -q -m integration`（1 passed）
  - CI：PR #1 已触发并通过 GitHub Actions（unit + Milvus integration；badge 为 passing），run 链接待补
- 交付物名称、环境要求与运行命令：待生成发行 ZIP/可运行目录后填写
- 交付物 SHA-256；原件/副本一致性：待发布时填写
- 核心场景及扩展操作的实际结果：TXT 导入、Markdown 扩展导入、检索带来源、禁用扩展、
  错误路径、PDF 经 docparse 子进程导入、交互式问答与流式输出（mock LLM）已通过 CLI 实测
- 本次变化、已知限制/未验证范围：
  - 新增项目骨架、CLI、TXT/Markdown/PDF Connector、Milvus 混合检索、BM25 回退、
    检索增强问答、模型 provider 选择、测试、CI、设计文档
  - 新增交互式 REPL、`/setup` 向导、本地 API Key 管理、流式答案和 `-p` 一次性问答
  - PDF/扫描件依赖外部 docparse，CI 只验证 IR 映射与不可用错误路径，未跑真实 OCR
  - 本地 Milvus 集成测试已通过；PR #1 的 GitHub Actions integration job 已通过
  - 真实 DeepSeek/DashScope/OpenAI 问答未做人工验收（无 Key）
  - 未实现 Web、重排/查询扩写、多轮对话记忆
- 发布决定：**HOLD**。原因：尚未生成并验证最终发行物、未打 Tag；完成 release 自检后再改为 RELEASE。

## 发行前检查

- [ ] 工作区无未提交改动
- [ ] `cd project && pytest -q -m "not integration"` 通过
- [ ] `MILVUS_URI=... pytest -q -m integration` 通过
- [ ] `ruff check src tests connectors` 通过
- [ ] 生成发行 ZIP/可运行目录，并在新目录解压运行
- [ ] 记录交付物 SHA-256
- [ ] 打附注 Tag，更新本文件的 Commit/Tag/SHA
- [ ] 把发布决定改为 RELEASE 并写明理由（或保留 HOLD 并说明缺口）
