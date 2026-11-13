# 本次版本交付

- 版本及选择理由：0.1.0。首个可运行骨架，核心场景、扩展机制与测试已具备，但尚未做最终发行验证。
- 最终 Commit（完整 SHA）：待发布时填写
- Tag 及其目标 Commit（完整 SHA）：待发布
- 工作区状态、构建/测试结果及 CI 真实状态：
  - 工作区：待发布时执行 `git status`
  - 本地测试：`cd project && pytest -q`（当前 12 passed）
  - CI：推送后以 GitHub Actions 实际运行结果为准
- 交付物名称、环境要求与运行命令：待生成发行 ZIP/可运行目录后填写
- 交付物 SHA-256；原件/副本一致性：待发布时填写
- 核心场景及扩展操作的实际结果：TXT 导入、Markdown 扩展导入、检索带来源、禁用扩展、错误路径已通过 CLI 实测
- 本次变化、已知限制/未验证范围：
  - 新增项目骨架、CLI、TXT/Markdown Connector、BM25 索引、测试、CI、设计文档
  - PDF/扫描件依赖外部 docparse，未在 CI 中验证
  - 未实现 Web、向量检索、生成回答
- 发布决定：**HOLD**。原因：尚未生成并验证最终发行物、未打 Tag；完成 release 自检后再改为 RELEASE。

## 发行前检查

- [ ] 工作区无未提交改动
- [ ] `cd project && pytest -q` 通过
- [ ] `ruff check src tests connectors` 通过
- [ ] 生成发行 ZIP/可运行目录，并在新目录解压运行
- [ ] 记录交付物 SHA-256
- [ ] 打附注 Tag，更新本文件的 Commit/Tag/SHA
- [ ] 把发布决定改为 RELEASE 并写明理由（或保留 HOLD 并说明缺口）
