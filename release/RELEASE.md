# 本次版本交付准备

- 发布决定：**HOLD**。用户选定B功能候选，尚未完成正式提交、Review、最新CI与制品核验。
- 软件包版本：沿用pyproject.toml的0.1.0；B是源码快照标识，不冒充Git Tag或正式发行。
- 原始参考基线：f38a46de9ee250fe9b9cf7499b8cb1622a893a66。
- 当前恢复点：c5439493f311b08aeaf79d36b5d9748eccadf752，checkpoint-20261126保留；最终Commit及发行Tag待用户操作后填写。
- 工作区：B增量和交付文档尚未提交；本轮未add/commit/push、未建PR/合并、未改工作流。
- 功能验收：用户多轮真实REPL结果见project/docs/testing/FINAL_ACCEPTANCE.md，模型/策略/语料分别统计，局限保留。
- 当前CI：已有离线Lint/Test和Milvus集成作业；B无最新远程成功记录。现有测试代码14项Ruff问题及缺显式制品构建需团队处理，不能沿用旧run宣称通过。
- 敏感范围：两份已跟踪个人验收脚本及历史含真实身份信息，不能直接推送当前历史；凭据、个人资料、模型/数据库、原始日志及A/B快照不公开。
- 许可：沿用项目MIT；docparse整体及模型/依赖再分发许可需团队确认。
- 制品名称、SHA-256、原件/副本一致性：未生成，待发布时填写。
- 最终环境和命令：按仓库README准备Python3.11、Milvus、现有模型与解析环境；从完整工程根执行 `PYTHONPATH=project/src python -m rag_multidocs --config project/config.json repl`，已有配置用户继续用自己的路径。
- 最终课程包：project/内保留完整工程及真正.git（不能仅复制worktree指针），release/与其并列。生成后在新目录解压，按README运行并填写结果。

本文件仍处于仓库内的准备模板；正式课程ZIP中按任务书放在源码仓库外的release/，避免补最终SHA又改动已发行Commit。当前没有制品运行结果，不以源码快照启动冒充最终发布验证。
