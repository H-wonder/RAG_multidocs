# B版任务书符合性追踪

依据仓库原始《大作业任务书》与《项目交付模板与检查清单》，不引入任务书之外的发布格式。用户已选定B候选，功能开发结束；本轮只做交付整理，不把待团队事项伪造为完成。

| 任务书要求 | 当前实现或材料 | 核查结论与责任 |
|---|---|---|
| 一个真实用户核心闭环 | CLI/REPL→Connector→Document/Chunk→Milvus→问答/来源 | 用户多轮真实验收支持，长文/漏引局限保留；功能负责人 |
| 公开扩展协议与独立扩展 | Document/ConnectorResult、RerankRequest/RankedCandidate；JSONL/custom | 接口和Manifest已核对；JSONL专项及真实业务有用户反馈，实际非核心成员归属仍需团队确认 |
| 注册 发现 禁用 失败处理 | Registry、独立目录及结构化警告 | 现有契约测试和用户专项；不宣称热加载/沙箱 |
| 正常 边界 失败 扩展测试 | project/tests、最终人工报告 | 代码与有限证据存在；本轮未重跑全量，不以数量证明可用 |
| 故意小错误 失败 恢复通过 | 提供临时副本演示步骤，历史原记录不冒充B新结果 | 待团队现场执行/确认真实记录，不修改B源码 |
| 清晰构建 测试及基础检查 | pyproject、pytest、Ruff、现有ci.yml | 命令已核对；当前Ruff检查有14处已有测试风格/未用导入问题，不能宣称CI可绿 |
| CI默认构建 测试 真实成功 | .github/workflows/ci.yml已有离线与Milvus测试 | 缺显式发行制品构建/上传；B未运行远程CI，本轮不改工作流 |
| README及使用/配置说明 | 根README、project/README、USAGE | 已更新，命令/签名/源码一致性做最小检查 |
| 设计文档与两张图 | project/docs/DESIGN.md与DESIGN.docx | 按B架构重写并渲染检查；保留架构及关键流程图 |
| 真实成员贡献与具体Review | README责任表、现有Git历史 | 姓名/学号/实际审查人/意见/处理SHA未获完整确认，待团队填写 |
| 一次可追溯发布 Commit/Tag/摘要 | release/RELEASE.md为HOLD准备记录 | 无新Commit/Tag/PR/发行；最终身份与制品SHA待发布后填写 |
| 一个ZIP含完整.git与运行物 | 当前仅准备方案 | worktree的.git是外部指针，不能直接当完整历史打包；最终需完整仓库副本及外置release |
| 原件解压新目录复现 | 现有入口与依赖说明 | 尚无最终ZIP，待团队按README执行，不能以快照启动代替最终制品验证 |
| 许可与敏感信息保护 | MIT及THIRD_PARTY、忽略规则、显式提交清单 | 模板无密钥；私人原文/日志/历史全文排除；docparse整体再分发许可待确认 |

最终提交结构沿用课程格式：组号_项目名/project/放完整工程（含真正.git），并列release/放制品及RELEASE.md。工程可保留现有内部project/src与已有资源结构，不为目录美观移动包和插件。课程要求不另交Prompt、日志、截图或个人原始资料。

[最终人工验收](testing/FINAL_ACCEPTANCE.md)区分模型/策略/语料；[提交准备](testing/DELIVERY_PREPARATION.md)列出实际检查、建议提交/排除范围和未完成事项。
