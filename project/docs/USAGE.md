# B版使用说明

安装与整体架构见[仓库README](../../README.md)。以下命令均从仓库根目录执行；激活已有rag环境，显式PYTHONPATH避免editable安装指向其他worktree。

## 配置与环境

`Settings.load(path=None)`读取JSON，再由已实现的环境变量覆盖；配置目录决定Connector相对路径及来源基准。模板位于 `project/config.example.json`，首次复制为本地 `project/config.json`。保留既有配置及集合时不要复制覆盖。

| 字段或环境变量 | 作用 |
|---|---|
| index_backend / RAG_INDEX_BACKEND | milvus或keyword；正常Milvus路径无SQLite Catalog |
| fallback_to_keyword | 仅CLI构建Runtime时允许Milvus失败告警回退；create_core直接调用不回退 |
| milvus_uri/db/collection / MILVUS_URI/DB/COLLECTION | 实际知识库地址和集合；命令/Trace不能替代确认此值 |
| embed_backend/model/dim/base_url / EMBED_* | 现有Embedding Provider，维度必须与集合一致 |
| chat_provider/model/base_url / CHAT_* | 对话Provider/模型/地址；空Provider为只检索 |
| chat_context_tokens / CHAT_CONTEXT_TOKENS | 0采用现有Provider自动窗口，显式值可覆盖 |
| chat_max_tokens / CHAT_MAX_TOKENS | 0采用现有自动输出预算，具体预算看Trace |
| rag_policy / RAG_POLICY | stable、lightweight、quality、experimental |
| connectors_dir / RAG_CONNECTORS_DIR | 插件目录；enabled_connectors是允许列表 |
| docparse_repo/python/mode/timeout_s | 独立解析进程；对应DOCPARSE_*环境变量 |
| credentials_file / RAG_CREDENTIALS | 本地凭据，0600，不进入Git |

环境变量名以 `config.py` 为准，不任意增加变量。API密钥从相应Provider环境变量或凭据文件加载；环境优先。启动现有服务：`docker start milvus-standalone`；Ollama按现有服务启动方式运行。新部署和模型准备见原[环境资源](../../已有资源/文档/07-环境搭建与依赖.md)。

## CLI与REPL

```bash
PYTHONPATH=project/src python -m rag_multidocs --config project/config.json --help
PYTHONPATH=project/src python -m rag_multidocs --config project/config.json repl
```

```text
/ingest auto project/samples/notes.txt project/samples/markdown
/search 检索器
/ask 文档描述了哪些检索步骤？
/trace
/policy
/policy quality
/config
/stats
/docs
/connectors
/models
/model chat ollama gemma3:4b
/exit
```

`/model`会写本地配置；只改对话模型无需重建索引。更改Embedding模型/维度会影响向量兼容，先选择匹配集合，不能拿既有库尝试不兼容参数。

```bash
PYTHONPATH=project/src python -m rag_multidocs --config project/config.json ingest --source auto --path project/samples/notes.txt project/samples/markdown --json
PYTHONPATH=project/src python -m rag_multidocs --config project/config.json search '切块' --top-k 3 --json
PYTHONPATH=project/src python -m rag_multidocs --config project/config.json ask '文档如何切块？' --policy quality --json --trace
PYTHONPATH=project/src python -m rag_multidocs --config project/config.json docs list
PYTHONPATH=project/src python -m rag_multidocs --config project/config.json connectors list
```

`--doc-id`可在search/ask重复传入，用于明确文档范围；同名文件存在歧义时使用准确路径或doc_id。普通自然语言不应自动硬限制到主题相似文件。doc_ids不是完整权限系统，服务层仍须在检索前设置数据隔离；敏感资料不得因切换Provider而默认外发。

## 模型与插件设置

```bash
PYTHONPATH=project/src python -m rag_multidocs --config project/config.json keys set --provider dashscope
PYTHONPATH=project/src python -m rag_multidocs --config project/config.json models use --kind chat --provider dashscope --model deepseek-v4.1-flash-1017
PYTHONPATH=project/src python -m rag_multidocs --config project/config.json connectors disable connector.jsonl
PYTHONPATH=project/src python -m rag_multidocs --config project/config.json connectors enable connector.jsonl
```

`keys set`隐藏输入；不在命令行提供真实密钥。插件启停只影响新导入，不删除已有索引。复制新插件到目录后，显式允许其名称并重启；Reranker通过策略选择，未提供通用Reranker热管理CLI。对话模型是否可用由实际服务决定，本轮不下载模型。

## 更新与结果

相同来源再次导入沿用稳定身份，内容/配置签名相同可报告unchanged；变更时写新Chunk并清除该文档旧余块，普通写异常尝试恢复旧数据，不是完整数据库事务。删除文档只有Core公开 `delete(doc_id)`；当前CLI没有对应通用删除子命令，不编造命令。更新时保留来源基准/命名空间，不以删除知识库代替更新。

`answered`表示路径返回答案，不是完整事实认证。`partial`仍可能包含有价值内容；`insufficient_evidence`与`generation_failed/retrieval_failed`分开。来源为本次真正上下文片段，编号可不连续；`page=0`或空bbox表示未有可靠位置，不能补造。`/trace`不应公开包含私人材料的完整Prompt/上下文。

## 常见故障

| 现象 | 核对方式 |
|---|---|
| 意外keyword/空结果 | 看backend及warning，核对Milvus服务和配置，两个后端不共享数据 |
| PDF或Office解析不可用 | 检查docparse_repo、docparse_python与解析环境依赖；不能把解析失败说成资料不足 |
| 扫描件OCR失败 | 检查现有Ollama OCR模型、服务地址和解析模式；不要切远程处理私人资料 |
| CrossEncoder回退 | 看RERANKER警告及执行mode，核对已有缓存/依赖；不代表CE已实际运行 |
| 连接错误/输出为空 | 查看Provider、模型、警告和token预算；程序不靠无限重试保证成功 |
| 长文漏召回/跨文档漏引用 | 属于已知局限；可用准确来源范围辅助，不能把answered当完整正确 |
| 切换目录后出现新ID | 核对source_namespace及配置根；不要全库重入或清库 |

详细公开签名及错误边界见[Core接口](RAG_CORE_API.md)。
