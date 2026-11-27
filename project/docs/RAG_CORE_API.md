# RAG Core 接口与CLI接入

适用用户选定的冻结B版。公开Core位于 `rag_multidocs.core`，组装工厂位于 `rag_multidocs.services`，配置位于 `rag_multidocs.config`。既有CLI适配器是 `rag_multidocs.cli.app.Runtime/build_runtime`。本文件不新建接口，不要求CLI读取qa、ingestion、索引或Worker内部实现。

## 1 接入选择与生命周期

CLI同学优先复用已有Runtime：它处理Provider/模型签名变化、策略选择、无生成模式、CLI后端回退及常驻Worker授权。仅需服务的其他适配层使用create_core，显式传入策略和需要的RerankerRegistry；工厂默认stable，不自动读取settings.rag_policy作为策略，也不自动进行Milvus到keyword回退。创建对象可能连接/创建所配置的Milvus集合；生产使用前确认目标，测试勿指向主库。

调用方拥有Core/Runtime生命周期，放在try/finally中close；不要共享一个会被其他调用方关闭的Index/Registry。不要把构造RAGCore本身当作线程安全/事务保证。插件进程spawn要求脚本入口采用 `if __name__ == '__main__'`。正常Milvus没有SQLite活动版本，旧catalog_path不是新接入方案。

### 最小完整接入示例

以下从仓库根运行（PYTHONPATH=project/src）；sample仅为公开样例。实际执行会导入该样例并只在其doc_ids范围内问答，可能调用已配置模型，本轮只核对签名和语法，不自动对用户库执行。

```python
from pathlib import Path
from rag_multidocs.config import Settings
from rag_multidocs.credentials import apply_keys
from rag_multidocs.cli.app import build_runtime

def main():
    repo = Path.cwd()
    settings = Settings.load(repo / 'project/config.json')
    # 库调用不会隐式读取密钥文件；已有Shell环境变量优先。
    apply_keys(settings.credentials_file)
    runtime = build_runtime(settings)
    try:
        print('backend:', runtime.backend, 'warning:', runtime.warning)
        report = runtime.core.ingest('auto', str(repo / 'project/samples/notes.txt'))
        payload = report.as_dict()
        print(payload)
        sample_ids = sorted({doc['doc_id'] for file in payload.get('files', [])
                             for doc in file.get('documents', [])})
        if not sample_ids:
            raise RuntimeError('样例未成功导入，停止问答')
        search = runtime.core.search('检索器', doc_ids=sample_ids)
        for hit in search['hits']:
            print(hit['source'], hit.get('page', 0), hit['text'])
        result = runtime.ask(settings, '文档如何描述检索器？', doc_ids=sample_ids)
        print(result['status'], result['answer'])
        for citation in result['citations']:
            print(citation['index'], citation['source'], citation.get('page', 0))
        print(result.get('warnings', []))
        print(runtime.core.stats())
    finally:
        runtime.close()

if __name__ == '__main__':
    main()
```

独立服务组装（不依赖CLI）：

```python
from pathlib import Path
from rag_multidocs.config import Settings
from rag_multidocs.credentials import apply_keys
from rag_multidocs.services import create_core
from rag_multidocs.retrieval.profiles import recommended_policies
from rag_multidocs.rerankers import RerankerRegistry

def main():
    repo = Path.cwd()
    settings = Settings.load(repo / 'project/config.json')
    apply_keys(settings.credentials_file)
    ranks = RerankerRegistry(timeout_s=60, persistent_paths=[repo/'project/rerankers/crossencoder'])
    try:
        core = create_core(settings, base_dir=repo, rerankers=ranks,
                           policy=recommended_policies()[settings.rag_policy])
    except BaseException:
        ranks.close()
        raise
    try:
        result = core.ask('文档的关键内容是什么？', style='concise')
        print(result['answer'], result['citations'])
    finally:
        core.close()

if __name__ == '__main__':
    main()
```

此例的RerankerRegistry构造为现有公开扩展服务；不要求CLI直接实例化内部Worker。若create_core构造抛错，调用方还应关闭已自行构造的ranks；已有build_runtime已处理这条异常清理分支。

## 2 构造 配置与服务入口

### Settings.load

```python
def load(cls, path: str | Path | None=None) -> 'Settings':
```

`from rag_multidocs.config import Settings`。path为str/Path/None；默认读取RAG_CONFIG或config.json。返回Settings，缺文件采用默认值；非法JSON/值可抛JSONDecodeError/ValueError/OSError。读取配置和环境，不修改文件。Connector相对路径/来源基准与配置目录有关；data_dir及credentials相对路径在不同入口仍受base_dir/CWD影响，部署建议用明确绝对路径。

### Settings.save

```python
def save(self, path: str | Path | None=None) -> Path:
```

`settings.save(path)`返回Path并写本地配置；None使用RAG_CONFIG或当前目录config.json，**不会自动记住load(path)的路径**；保存已有配置应显式传回同一路径。磁盘/权限错误可抛OSError。不要把凭据嵌入Settings；本轮不修改用户配置。

### create_core

```python
def create_core(settings, *, base_dir, data_dir=None, rerankers=None, policy=None, offline=False, index=None, registry=None, chat=None, rewriter=None, chunk_options=None):
```

`from rag_multidocs.services import create_core`。settings为Settings；base_dir必填str/Path。data_dir覆盖数据目录；registry/index可传已有实例，否则按配置创建；chat为ChatModel，None且配置Provider时建立延迟工厂；rerankers/policy/rewriter/chunk_options可显式注入；offline仅控制工厂本地Embedding加载，不构成全局断网/隐私保证。返回RAGCore，初始化错误可能是ValueError、连接错误或依赖异常。使用传入资源后close也会关闭它们；不可在不清楚所有权时共享。

### RAGCore.__init__

```python
def __init__(self, registry, index, *, rerankers=None, policy=None, chat=None, chat_factory=None, chat_budget_factory=None, rewriter=None, chunk_options=None, support_level3=True, catalog_path=None):
```

`from rag_multidocs.core import RAGCore`。registry与index必填，须满足既有Registry/Index协议。chat/chat_factory/chat_budget_factory供宿主注入；policy默认stable；support_level3只影响保留的结构读取校验，普通自然语言不走旧审核链。catalog_path为历史兼容参数，正常纯Milvus工厂传None，不推荐新接入设置。返回实例，有资源持有/可选历史Catalog初始化副作用。

### build_runtime

```python
def build_runtime(settings: Settings) -> Runtime:
```

`from rag_multidocs.cli.app import build_runtime`。settings为Settings；返回Runtime(core, search, backend, warning)。沿用CLI后端回退，CrossEncoder精确路径授权常驻；不自行读取密钥文件，CLI main在此前apply_keys，库调用需显式加载或设置环境。可能连接Milvus，失败且配置允许则返回keyword和warning；配置/初始化失败仍可能抛异常。

### Runtime.ask

```python
def ask(self, settings, query, *, top_k=None, doc_ids=None, no_generate=False):
```

runtime.ask的settings必填，query为str；top_k=None或正整数（上下文最大6）、doc_ids=None/list[str]、no_generate=False。根据settings.rag_policy更新实际Core策略；Provider/模型/预算签名改变时重建Chat；返回问答dict或只检索dict。可能检索/调用模型/产生费用；不会自动授权外传。此适配方法没有filters/style参数，需此能力可直接用Core。

### Runtime.invalidate_chat

```python
def invalidate_chat(self):
```

无参数，返回None；清除Chat签名和core.chat以便下次runtime.ask重新创建，不清空索引。没有Core.set_model，不编造此方法。

### Runtime.close

```python
def close(self):
```

无参数，返回None；委托Core关闭资源。Runtime不提供稳定异步或流式上下文管理协议，使用try/finally。

## 3 RAGCore公开方法

下列方法均从 `rag_multidocs.core.RAGCore` 调用，签名逐项对应B源码。标记“读”为不修改文档数据，仍可能使用网络、缓存或模型；副作用需结合配置判断。

### RAGCore.ingest

```python
def ingest(self, connector, path, options=None):
```

connector为str（auto或公开插件名），path为str/Path或受支持路径列表，options=None/dict。返回IngestReport或BatchIngestReport；auto总走批量报告，不能假设其有documents属性。导入有解析、Embedding和持久化副作用。常见文件/解析/发布错误进入issues，其他配置或后端异常也可能抛出。例：`core.ingest("auto", "project/samples/notes.txt").as_dict()`。相同来源复用ID，内容/处理签名未变可unchanged；没有单独update接口，正常重新导入即更新。

### RAGCore.ingest_many

```python
def ingest_many(self, connector, paths, options=None):
```

connector为str，paths为单一路径或可迭代路径，options=None/dict共享各文件。返回BatchIngestReport；自动递归目录并去除重复实际路径，逐文件继续；结构性输入问题见input_issues。多文件不是跨文件事务。例：`core.ingest_many("auto", ["notes.txt", "docs/"])`。

### RAGCore.list_documents

```python
def list_documents(self):
```

无参数，返回list[dict]。Milvus记录含doc_id/source/title/chunks/pages；其他后端字段可不同，不强制推断有版本号。读操作，会访问存储；连接错误可能抛出。例：`for row in core.list_documents(): print(row["doc_id"], row["source"])`。

### RAGCore.document

```python
def document(self, doc_id):
```

doc_id为str；返回文档dict或None。包含可恢复的原始Document/IR元数据，Milvus version=None，旧数据缺完整payload时只返回已有元数据；完整性损坏可抛ValueError(DOCUMENT_METADATA_INCOMPLETE)，不补造内容。读操作。例：`doc = core.document(doc_id)`；可能包含私密原文，不能直接公开。

### RAGCore.stats

```python
def stats(self):
```

无参数；返回后端统计dict，Milvus常见collection/exists/rows/backend，以及capabilities.version_publication与recovery_warnings。读操作。例：`print(core.stats())`；rows是Chunk数，不是文档数；连接错误可抛出。

### RAGCore.search

```python
def search(self, query, *, doc_ids=None, filters=None, policy=None):
```

query为str；doc_ids=None或list[str]，filters=None/dict由现有索引支持，policy=None或RetrievalPolicy仅覆盖本次。返回dict，主要hits/retrieval_trace/context_tokens/index_calls/warnings/usage/retrieval_seconds；不是命中list。读操作及可选检索改写/Embedding/重排。捕获TimeoutError/OSError/RuntimeError并给retrieval_error=True及RETRIEVAL_FAILED；配置ValueError等不一定捕获。例：`r = core.search("流程", doc_ids=[doc_id]); print(r["hits"])`。

### RAGCore.ask

```python
def ask(self, query, *, style='professional', doc_ids=None, filters=None):
```

query为str，style=professional/teaching/concise；doc_ids、filters同search。返回问答dict见第5节。检索后自然语言最多一次正常生成，也可能确定性提取/计算0次；Experimental改写另计。Provider错误转generation_failed；参数/预算等ValueError可抛出。会调用配置模型及可能产生费用；权限/远程授权由宿主在检索前限制范围，不应只在输出后过滤。例：`core.ask("谁负责？", style="concise", doc_ids=[doc_id])`。

### RAGCore.answer_evidence

```python
def answer_evidence(self, question, hits, *, style='professional'):
```

question为str，hits为已有真实命中list[dict]，style同ask。不执行检索，要求core.chat已初始化，否则ValueError(CHAT_PROVIDER_REQUIRED)；不会通过chat_factory隐式初始化。返回共享回答结果，但没有完整检索Trace和ask的总usage封装。可能调用模型，有外发副作用；宿主负责hits来源/隐私范围。例：在一次core.ask初始化Chat后，`core.answer_evidence("总结证据", r["evidence"])`。无doc_ids/on_token参数。

### RAGCore.delete

```python
def delete(self, doc_id):
```

doc_id为str；正常Milvus返回{"doc_id":...,"status":"deleted"}，删除该文档Chunk并调用persist。**破坏性文档操作**，不是清库；不存在ID也可能返回deleted，不证明曾有记录。后端错误可抛出；无稳定批量删除/撤销承诺。例：经用户确认后`core.delete(doc_id)`。

### RAGCore.plugins

```python
def plugins(self):
```

无参数；返回connectors/rerankers列表及warnings，过滤manifest.config，包含发现状态而非保证解析或模型执行成功。读操作，运行Manifest发现，不自动安装/热加载。例：`print(core.plugins()["warnings"])`。修改启用状态没有Core通用公开setter；通过配置重启，或已有CLI Connector命令。

### RAGCore.recover

```python
def recover(self):
```

无参数。正常纯Milvus抛ValueError(INDEX_CAPABILITY_UNAVAILABLE: recovery)，不能当作事务恢复；只在历史Catalog路径返回reconcile结果。不要在新CLI上宣称纯Milvus有版本回滚。

### RAGCore.cleanup

```python
def cleanup(self, doc_id=None):
```

doc_id=None/str。正常纯Milvus抛ValueError(INDEX_CAPABILITY_UNAVAILABLE: version cleanup)；仅旧Catalog路径可清理历史版本。不能用此方法删除库。

### RAGCore.close

```python
def close(self):
```

无参数，返回None；释放Reranker常驻进程、可选历史Catalog和Index连接。必须在退出/异常分支调用；不删除持久化数据。不承诺可无限重复close或关闭后继续使用。

## 4 数据类型与参数契约

### Document与Chunk

导入路径 `rag_multidocs.model`，以下为实际dataclass字段，省略字段只能使用其真实默认值。source为逻辑来源，doc_id由来源身份稳定派生，不是可随意随机生成的版本。metadata中sha256/content_sha256/processing信息用于去重/更新；同一文件换命名空间可能成为新来源。

#### Document（rag_multidocs.model）

```python
    doc_id: str
    source: str
    content: str
    title: str = ''
    media_type: str = 'text/plain'
    metadata: dict[str, Any] = field(default_factory=dict)
    segments: list[dict[str, Any]] = field(default_factory=list)
```

#### Chunk（rag_multidocs.model）

```python
    id: str
    doc_id: str
    text: str
    source: str
    index: int
    content_type: str = 'text'
    section_path: str = ''
    page: int = 0
    page_end: int = 0
    bbox: str = ''
    block_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
```

#### ConnectorConfig（rag_multidocs.contracts）

```python
    name: str
    path: str
    options: dict[str, Any] = field(default_factory=dict)
```

#### ConnectorIssue（rag_multidocs.contracts）

```python
    code: str
    message: str
    source: str = ''
```

#### ConnectorResult（rag_multidocs.contracts）

```python
    documents: list[Document] = field(default_factory=list)
    issues: list[ConnectorIssue] = field(default_factory=list)
```

#### SearchHit（rag_multidocs.contracts）

```python
    candidate_id: str
    text: str
    source: str = ''
    doc_id: str = ''
    score: float = 0.0
    semantic_score: float | None = None
```

#### RerankRequest（rag_multidocs.contracts）

```python
    query: str
    candidates: tuple[SearchHit, ...]
    config: dict[str, Any] = field(default_factory=dict)
    contract_version: str = '1'
```

#### RankedCandidate（rag_multidocs.contracts）

```python
    candidate_id: str
    rank: int
    score: float
```

`Document.from_text(content, source, *, title="", media_type="text/plain", metadata=None)`建立来源ID和正文hash；`to_dict()/from_dict(data)`为序列化转换。Chunk.as_dict()使用id/index并将block_ids序列化为逗号串；Core命中规范化后提供chunk_id/chunk_index及列表block_ids，不应混淆。Document.segments是公开Block字典，meta可有location/table_data/table_headers/semantic_cells/source_table_id；row quote才是原始证据，retrieval_text是派生检索视图，不能作原文引用。

ConnectorIssue只有code/message/source，不等同于Core warning中的severity。ConnectorResult.documents允许部分成功同时有issues。ConnectorError(code,message,source="")是RuntimeError子类；Registry将受控错误转换成Issue。

### 配置与策略

Settings是dataclass，完整公开字段和默认值见下表（模板与环境覆盖后的实际配置可不同）。

| 字段 | 注解 | 默认 |
|---|---|---|
| `data_dir` | `str` | `'data'` |
| `connectors_dir` | `str` | `'connectors'` |
| `enabled_connectors` | `list[str]` | `field(default_factory=lambda: list(DEFAULT_CONNECTORS))` |
| `index_backend` | `str` | `'milvus'` |
| `fallback_to_keyword` | `bool` | `True` |
| `milvus_uri` | `str` | `'http://localhost:19530'` |
| `milvus_db` | `str` | `'rag'` |
| `milvus_collection` | `str` | `'rag_chunks'` |
| `milvus_dense_weight` | `float` | `0.7` |
| `milvus_sparse_weight` | `float` | `0.3` |
| `embed_backend` | `str` | `'local'` |
| `embed_model` | `str` | `''` |
| `embed_dim` | `int` | `0` |
| `embed_base_url` | `str` | `''` |
| `chat_provider` | `str` | `''` |
| `chat_model` | `str` | `''` |
| `chat_base_url` | `str` | `''` |
| `chat_temperature` | `float` | `0.2` |
| `chat_max_tokens` | `int` | `0` |
| `chat_context_tokens` | `int` | `0` |
| `rag_policy` | `str` | `'stable'` |
| `rag_top_k` | `int` | `5` |
| `docparse_repo` | `str` | `''` |
| `docparse_python` | `str` | `''` |
| `docparse_mode` | `str` | `'auto'` |
| `docparse_timeout_s` | `int` | `1800` |
| `credentials_file` | `str` | `'.rag/credentials.json'` |

`Settings.to_dict()`返回字段dict；`index_file`是data_dir/index.json的Path属性。只有Connector路径通过resolved_connector_settings解析；凭据相对路径由credentials_path按CWD解释。`Settings.load`不会自动读取凭据，需要apply_keys或已有CLI main。配置模板不是个人配置备份，切模型不要覆盖知识库参数。

`RetrievalPolicy`来自 `rag_multidocs.retrieval.controller`；`recommended_policies()`来自 `rag_multidocs.retrieval.profiles`，返回四种预设。参数包括adaptive/candidate_k/context_k/context_tokens/reranker、alpha/beta/gamma、use_mmr/use_neighbors/use_recall/use_rewrite/use_coverage、max_index_calls/max_generation_calls、allow_calculation/structured_table_answers及模型预算。resources(query)返回kind/candidate_k/context_k/context_tokens/reranker，非法数量/窗口/权重可ValueError。现有限制候选<=48、上下文<=6、证据token<=3000、检索调用<=4；不是任意任务调度接口。ChatBudget位于qa.llm，字段context_window/initial_tokens/retry_tokens/reasoning_effort，经as_trace提供预算，用户不必复制其内部模型表。

### 导入报告

IngestReport位于 `rag_multidocs.ingestion.service`，字段connector/path/documents/chunks/issues；ok只表示有返回文档，不代表所有文件成功。documents常含doc_id/source/title/chunks/version/status/needs_review/warnings/quality_status/near_duplicate_candidates及摘要指纹。

BatchIngestReport位于 `rag_multidocs.ingestion.batch`，字段files/input_issues；as_dict返回discovered/succeeded/failed/skipped/partial/document_count/chunks/input_issues/files。**failed计数包含partial**，不能和partial相加当独立文件数。每个files条目有path/connector/status/document_count/chunks/documents/issues；status为succeeded/partial/failed/skipped。近似重复默认报告，不等于自动删重；纯Milvus不承诺旧Catalog的近重复事务能力。

## 5 问答 命中 引用和Trace

所有dict采用加性兼容：使用get读取可选诊断字段，不假设不同路径完全同构。

| 结果字段 | 实际意义 |
|---|---|
| status / answer | 状态字符串与展示文本；详见下一节 |
| evidence | 检索控制器选出的上下文候选，不一定全部送模型 |
| generation_evidence | 真正送入生成Prompt的片段；提取/计算路径可以为空 |
| extraction_evidence / calculation_citations | 结构读取/运算实际依据；仅有关路径出现 |
| answer_evidence | 回答阶段使用的证据；不是语义正确证明 |
| citations | 本次真实编号、来源元数据和excerpts；编号可不连续 |
| warnings | list[dict]，通常code/message，部分有severity/plugin/fallback |
| claims | 兼容表示；自然语言可能是整段答案与所有标注来源，不是逐句已审核claim |
| used_llm / model_calls | 该回答生成路径是否调用、逻辑生成次数，不含改写 |
| attempts | 实际Prompt/raw及格式诊断；可能有私人数据，不应公开或默认落盘 |
| usage | request级embedding_calls/items、rewrite_calls、generation_calls、transport_attempts/retries及Provider报告token；可选字段取决于Provider |
| retrieval_trace | 每轮资源、原始/融合候选、排序/覆盖、reranker_execution、索引调用等；Trace含原文，发布前脱敏 |
| token_budget | 实际模型窗口、初始/预留输出；与retrieval/context估计配合查看 |
| source_constraints | explicit/matches/doc_ids/blocked/warnings，主题匹配不得变强制范围；它不是完整访问控制系统 |
| fact_verified | false；不得显示“已完成通用事实认证” |

命中一般有chunk_id/doc_id/source/text/content_type、page/page_end/section_path/bbox/block_ids、metadata/location和阶段分数score/rrf_score/rerank_score/selection_relevance。原始不同分数含义不同，不能跨阶段直接比较或解释为正确概率。

Citation实际提供index/evidence_id/chunk_id/doc_id/source/page/page_end/bbox/block_ids/location/ocr_mode/needs_review/version/origin/warnings/fact_verified/excerpts；page=0、bbox空表示无可靠位置，不能补造。表格可用location.worksheet、row/column及真实单元格地址；不是所有解析后端都提供所有字段。

`trace_summary(result, profile)`位于既有CLI的 `rag_multidocs.cli.asking`，用于展示policy/status/answer_mode、候选/上下文来源、重排实际执行、usage及预算。没有新的Core.trace()接口。真正0模型结构读取应generation_calls=transport_attempts=0；自然生成至少记录逻辑生成，transport可能因改写/重试更多，二者不可混为一谈。profile是调用方标签，诊断仍须同时看resources与实际mode。

## 6 状态 错误与展示

| 状态 | 处理建议 |
|---|---|
| answered | 返回了回答或确定性结果；**不认证完整性、实体关系或每句引用** |
| partial | 保留有用部分，展示答案与警告；不当作系统运行失败 |
| insufficient_evidence | 无有效依据或明确无法确定，正常展示；不等同Provider错误 |
| generation_failed | 初始化/模型调用/空输出失败；Core检索失败时也可能映射到此状态，结合warnings判断 |
| retrieval_failed | CLI只检索封装在retrieval_error时使用；Core.search本身无status而有retrieval_error |
| retrieval_only | Runtime未配置Provider或no_generate=True；返回证据，不伪装模型回答 |

CLI应展示答案、真实来源和关键警告，不能自行按数字/词汇重做复杂事实审核。正常自然语言已取消UNSUPPORTED_NUMBER强制删句和字段值自动补引用；仍检查引用编号范围，保留真实原文约束Prompt。无引用或无效引用可partial，程序无法发现的漏答/漏引仍可能answered；最新人工结果也保留这一局限。

常见Connector反馈包括FILE_NOT_FOUND/EMPTY_SOURCE/CONNECTOR_DISABLED/CONNECTOR_NOT_FOUND/CONNECTOR_DUPLICATE/CONNECTOR_ERROR及解析依赖、超时、CSV解码/列数等错误。Reranker失败按原候选排序回退并告警，不删除文档；后续上下文选择仍可能改序。配置非法、存储连接和元数据损坏可能抛异常，不承诺所有异常都转dict；上层应保留错误边界并finally close，不把故障转换成“资料不足”。

## 7 模型 策略和流式边界

Runtime接入推荐：

```python
# 接续已创建的settings/runtime；保持同一Milvus集合和Embedding。
settings.rag_policy = 'quality'
settings.chat_provider = 'ollama'
settings.chat_model = 'gemma3:4b'
settings.chat_base_url = ''
local_result = runtime.ask(settings, '文档中的流程是什么？')
# 如需远程，先取得资料外传授权并设置doc_ids隔离，再切已支持Provider。
# settings.chat_provider = 'dashscope'
# settings.chat_model = 'deepseek-v4.1-flash-1017'
# settings.chat_base_url = ''
# remote_result = runtime.ask(settings, '公开资料的问题', doc_ids=approved_doc_ids)
```

Runtime.ask按配置签名更新模型；修改settings本身不持久化，调用settings.save才写盘。REPL `/policy`是当前会话切换，`/model`保存配置。直接Core没有set_model/set_policy公开方法：策略在create_core(policy=...)构造时注入，search可单次policy覆盖；需要更换Core配置可关闭旧实例后按同一数据配置重建，不清库。高级宿主可以显式注入chat/chat_factory，但不要求CLI使用内部成员更换模型。

`ChatModel.complete(messages)`为低层协议；OpenAICompatibleChat有stream，但RAGCore.ask/Runtime.ask没有on_token参数。当前CLI/REPL整段展示，ask --stream是兼容参数；不得在文档承诺公开异步token流或旧“先打印再删句”流程。

## 8 明确不新增的接口

没有公开的通用表格执行引擎、任意算术Agent、版本历史恢复、热插拔、完整权限系统或自动远程授权。doc_ids/filters可作为宿主隔离工具，不能自动承担租户认证。常驻Worker超时可回收，可信插件不是恶意代码沙箱。详见[插件协议](RAG_PLUGIN_CONTRACTS.md)、[设计说明](DESIGN.md)和[最终验收](testing/FINAL_ACCEPTANCE.md)。
