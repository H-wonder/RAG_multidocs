# B版工程交付准备

## 1 本轮范围与结论

用户选定B后只整理文档、历史与提交清单。没有修改Python源码、插件Manifest、依赖、运行配置或数据库；没有调用问答模型、重新入库、下载模型或执行Git暂存/提交/推送。最终源码与B及本轮开始清单逐项核对，A/B快照原样保留。

**工程文档准备已完成，但目前未达到可无阻塞提交PR的条件。** 已确认的CI静态问题、已跟踪脚本的私人身份/历史内容，以及团队Review/许可与制品事项需要人工处理。不能为追求准备完成而修改已冻结B业务逻辑或宣称CI已通过。

## 2 最终目录与阅读顺序

```text
README.md                         仓库入口
project/
  README.md                       产品目录入口
  src/ connectors/ rerankers/      与B一致
  tests/                          与B一致
  docs/
    README.md                     文档索引
    USAGE.md                      安装 命令 配置 排错
    RAG_CORE_API.md                CLI同学优先阅读
    RAG_PLUGIN_CONTRACTS.md        独立插件开发
    DESIGN.md DESIGN.docx         当前设计
    RAG_REQUIREMENTS_TRACEABILITY.md
    assets/                       设计图
    testing/
      FINAL_ACCEPTANCE.md         用户最终真实验收
      DELIVERY_PREPARATION.md     本文件
      SUBMISSION_INCLUDE.txt      建议公开提交路径
      SUBMISSION_EXCLUDE.txt      必须排除项
    history/
      README.md                   历史索引
      *.md                        历史脱敏摘要
      local/                      仅本地原始档案 被忽略
release/RELEASE.md                 HOLD准备状态
```

原始课程文件、资源包说明、各插件README、许可证和GitHub文件保留实际作用位置；历史全文归档而非销毁。原全文含临时授权、私人原文和旧结论，不作为公开测试证据。文档移动对应的旧路径删除与新摘要应由人工在提交清单中成对核对。

## 3 源码与最小检查

- 开始时210个源码/协议/Manifest/依赖条目与B冻结清单一致，结束时按同一清单复核；新增的虚构JSONL样例不是业务代码。
- Python文件、插件目录与公开协议未移动；Core及CLI签名由AST逐项核对，示例只做语法/调用签名检查，不借此隐式导入用户数据或调用模型。
- 原入口 `PYTHONPATH=project/src python -m rag_multidocs --config config.json repl` 仅执行/stats和/exit作启动核查；未运行问答或导入。
- Markdown相对链接、Manifest字段、配置模板及敏感提交范围作最小检查；DOCX用随附LibreOffice渲染并逐页检查，内部PDF/PNG不作为交付附件。
- 本轮没有重跑全量业务/单元测试。用户专项与人工结果见FINAL_ACCEPTANCE，不虚构逐题日志。

本机完整校验结果与整理前文档备份保存在仓库外的本地交付检查目录，不纳入公开Git。源码B清单与用户私人运行数据仍分别保护；源码相同不证明数据库状态是历史快照。

## 4 Git状态与变更边界

实际分支 `work/pure-milvus-20261126T235753`，HEAD为 `c5439493f311b08aeaf79d36b5d9748eccadf752`，已有 `checkpoint-20261126` 保留。origin为 https://github.com/H-wonder/RAG_multidocs.git 。本地原始main参考f38a46de9ee250fe9b9cf7499b8cb1622a893a66；本轮未fetch、未访问远端状态，不能称其为实时远端最新。

相对f38a46d，HEAD已包含3个后续提交；此外还有B的未提交增量和本轮文档整理。对照以本地基线为准，不能只提交本轮几个Markdown而遗漏B的既有未提交成果。CLI相对基线有历史变更，但本轮开始后未改；未提交文件的作者与其他成员授权不能仅按文件路径或git blame推定，团队Review需确认。

不执行git add、commit、push、PR或merge；不存在新B提交SHA。SUBMISSION_INCLUDE列的是**建议最终文件路径**，不是已经暂存的清单。旧文档删除项也须人工核对；不要使用git add .。历史归档路径和私密配置不可整体暂存。

## 5 敏感范围与历史风险

不得公开个人测试文档、联系人/学生资料、原始Prompt/上下文/模型回复、真实凭据或未脱敏日志。授权远程验收不等于允许GitHub发布。新JSONL样例明确虚构，现有安全样例保留；没有为了交付读取用户独立验收集。

当前候选文本/代码的针对性检查发现两个本地脚本包含真实身份标识：

- `project/tests/p0_live.py`
- `project/tools/rag_personal_acceptance.py`

两文件在HEAD中也已存在，且不是f38a46d的文件。本轮保持源码不变，将它们排除在建议公开文件清单；**排除清单或新增ignore规则不能从已跟踪文件和Git历史中去除内容**。不能直接推送当前历史并宣称已脱敏。后续需用户/团队选择合法的脱敏提交方案，并处理最终含.git的课程包；本轮不重写历史、不删除文件。

检查未发现建议范围内的实际API token模式命中或超过5MiB的大文件，但针对性扫描不是完整PII/秘密检测，也不证明所有历史提交安全。密钥、配置副本、运行目录、模型缓存、Milvus/SQLite及A/B快照均不纳入建议提交。真实配置保留原处不修改。

## 6 现有CI及缺口

现有 `.github/workflows/ci.yml` 包含：main push、PR和手动触发；Python3.11离线作业安装dev、Ruff及非integration测试；另一个作业启动Milvus2.5.6并执行integration测试，使用专用CI数据库/集合，最后移除CI容器。

本轮只读取工作流，不改动其他成员配置。B最新代码尚未运行远程CI；历史run记录不能算B当前成功。工作流没有显式build发行包或upload-artifact步骤，pip install不是完整的制品核验；课程“默认构建与测试、至少一次真实成功”和最终制品仍有缺口。

实际运行既有Ruff命令的静态检查结果：14项，分布在3份已有测试文件，未修改修复：

| 文件 | 已有问题 |
|---|---|
| tests/test_csv_connector.py | 6处E702 |
| tests/test_pure_milvus_runtime.py | 1处F401、1处E702、3处E701 |
| tests/test_temporal_cells.py | 2处E702、1处E701 |

这些不等于业务算法失败，但会阻塞当前CI的Lint步骤。后续由团队确认是否独立提交纯测试格式修正；本轮不改B源码、不屏蔽规则、不把警告说成CI通过。建议工作流维护者补充实际构建/安装或制品检查及保留成功run，在批准后修改，不另建架构。

## 7 课程质量与制品待办

README责任表待真实成员填写姓名/学号、本人实质改动、实际审查人、具体意见、处理/回应及SHA；不能用用户人工验收代替每成员Review。docparse现有目录含第三方许可，但未确认其整体再分发授权；MIT不自动覆盖所有资源/模型。

故意注错示例应只在临时副本执行：选择已有JSONL专项中关于CONNECTOR_ERROR或禁用的断言，在副本故意将对应Registry错误码改成错误字符串，确认该专项失败；恢复副本源码后确认通过。实际文件为src/rag_multidocs/connectors/registry.py，测试入口为test_a_plus_plugins.py::test_independent_connector_full_flow_disabled_and_failure。本轮没有运行这项注错演示，不编造红绿结果，也不在B工作区改错。

最终课程ZIP保持指定结构：组号_项目名/project/放**整个工程根目录**，内含现有project/src与已有资源；并列release/放实际制品和RELEASE.md。不为套模板移动现有包。本worktree的.git是指向其他目录的文件，不能直接打包冒充完整Git历史；用户须在隐私问题解决后准备完整Git仓库副本，保留必要隐藏历史。

尚无最终Commit、Tag、制品SHA、构建后的真实运行或新目录解压验证；release/RELEASE.md明确HOLD。正式提交前由团队人工处理这些缺口。本轮整理完成即停止，不借此开启RAG功能修复。


## 8 本轮最小检查实际结果

B源码/协议/Manifest/依赖210条hash一致，两个冻结快照源码hash保持一致；未暂存任何文件。公开接口18个签名逐项对应，示例调用按真实签名绑定，10条CLI示例通过现有parser，Python代码块语法检查完成；163条实际相对链接有效（代码内占位示例不当作真实链接）。设计文档最终8页，架构及流程图、每页文字和表格已渲染查看。原始CLI REPL读现有Milvus集合788行并退出，未提问、未调用模型、未导入文档。静态CI检查14项问题如上，故PR准备仍有阻塞。
