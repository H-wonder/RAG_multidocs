# RAG 策略、入库与评测

实现入口：`已有资源/样例/rag.py`。策略模块：`rag_strategy.py`；增量目录：`rag_catalog.py`。
旧 `demo_pdf_milvus.py` 继续使用独立测试集合。新流程默认使用 `rag_strategy_bge_v1`，不自动迁移或删除旧集合。

## 本机运行

所有命令从包含 `已有资源/` 的项目根目录执行：

```bash
conda activate rag
set -a
source .env
set +a
# 已创建的 Milvus 容器未运行时执行
# docker start milvus-standalone

# --root 是知识库逻辑根目录；文件必须位于其中，身份由相对路径决定。
python 已有资源/样例/rag.py ingest /path/to/knowledge/sub/report.pdf --root /path/to/knowledge
# 可一次传入多个文件；TXT/MD无需解析器，PDF/Office走docparse子进程。
python 已有资源/样例/rag.py list
python 已有资源/样例/rag.py ask "类别加权为什么会降低准确率？"
python 已有资源/样例/rag.py ask "这个结果是什么意思？" --tone teaching
python 已有资源/样例/rag.py ask "类别加权的效果？" --reranker crossencoder
# 限定来源：--source 可重复，使用list显示的相对路径。
python 已有资源/样例/rag.py ask "结论是什么？" --source sub/report.pdf --tone concise
```

全局选项 `--state 路径` 和 `--collection 集合名` 放在 ingest/ask/list 前。
默认状态路径为项目 `data/rag_strategy/`；模型、集合、Milvus URI/DB、切块版本与该状态目录绑定。
改变模型或索引空间时使用新状态目录和新集合。首次使用 BGE 或 CrossEncoder 需要下载权重。
所有查询写入 `query_*.json`，保留重排分数、最终上下文ID、token估计、原文、生成结果及重试过程。

## 文档切分与增量发布

1. 按 UTF-8 读取 TXT/Markdown；其他格式交给 docparse 输出 IR。保留页码、标题路径和表格类型。
2. 普通正文按段落/句界切分，目标 320 个估计 token、48 token 重叠。选择小于 BGE 512 token 的窗口以容纳标题与实际分词差异。
3. 标题路径作为嵌入前缀，Milvus保存原文。表格不在新流程中静默截断行数；对实际 tokenizer 超窗的结构块明确报错，要求拆分，不发布新版本。
4. `doc_id = hash(相对路径)`，跨目录同名文件不会覆盖；`sha256` 决定内容变化。相同内容且 Milvus 活动版本行数完整就跳过。
5. SQLite登记 STAGING 版本；用带版本的哈希主键写 Milvus、flush并用Strong一致性验证行数。
6. 验证文件在入库期间未变化，再在SQLite事务中激活新版本。查询只加载已激活版本；失败更新保留旧版本可用。历史/失败版本暂留 Milvus，后续可单独实现垃圾回收。
7. 写入使用本机文件锁串行化；这是本机单知识库方案，不是分布式事务。SQLite目录及Milvus数据必须一起备份。外部直接操作集合不受该发布约定保护。

该方式避免原先“先删整篇文档、再插入”在失败时导致旧知识不可检索。首次入库失败的文档不会出现在活动目录。
不自动把未传入的文件标记删除，不会因为单次增量导入误删其他文档。

## 表结构

Milvus `rag_strategy_bge_v1` 复用已有 schema：

| 字段 | 类型/用途 |
| --- | --- |
| id | VARCHAR(64) 主键，包含文档身份、版本、块序号的哈希 |
| doc_id / version | VARCHAR(64) / INT64，版本过滤 |
| text | VARCHAR(8000)，按UTF-8字节检查；中文 analyzer；BM25输入 |
| dense | FLOAT_VECTOR(512)，BGE归一化向量 |
| sparse | SPARSE_FLOAT_VECTOR，由服务端BM25 Function生成 |
| source / section_path | VARCHAR(1024)，相对路径与章节 |
| page / page_end | INT64，PDF从1开始；TXT/Markdown用0表示无物理页码 |
| content_type / bbox | VARCHAR(32) / VARCHAR(64)，类型与定位 |
| sha256 | VARCHAR(64)，源文件哈希 |

Dense 使用 HNSW、IP，M=16，efConstruction=64，搜索 ef 至少为候选窗口。
Sparse 使用 SPARSE_INVERTED_INDEX、BM25，k1=1.2，b=0.75。

SQLite `catalog.sqlite`：

| 表 | 关键字段/职责 |
| --- | --- |
| settings | 模型、维度、集合、服务地址、切块策略签名 |
| documents | doc_id主键、source唯一、sha256、active_version、chunks、pages |
| versions | (doc_id, version)联合主键，状态、哈希、错误、创建时间 |
| chunks | id主键、doc_id、version、position、payload；供邻块回填与追溯 |

活动版本通过SQLite生成Milvus过滤表达式；此方法适合课程/本机规模，文档量很大时应改用服务端可扩展的发布/分区方案。

## 召回和重排序

默认取24个融合候选；Milvus并行召回dense和BM25，以RRF融合，避免把两个不同量纲的原始分数直接相加。

提供两种可运行的重排策略：

- `lightweight`（默认）：重算候选与问题的BGE余弦，结合候选集内BM25：`0.8*cosine + 0.2*(BM25/max_BM25)`。它是可解释的二阶段轻量排序，不是CrossEncoder精排。
- `crossencoder`：`BAAI/bge-reranker-base`对(question, passage)联合评分，CPU、batch=8、max_length=512，logit经sigmoid供MMR排序。可用 `--rerank-model` 替换；模型错误明确报错，不偷偷回退为另一算法。截断窗口和分数阈值需按所选模型评测。

两种方案均保留dense_score、lexical_score、rerank_score；这些分数是排序信号，不是正确概率。
`--min-dense` 默认0.35是粗筛起点，未经大样本校准，不能单独判断是否可回答。
CrossEncoder仍先用BGE分数做该粗筛，以免精排在全不相关候选中选出“相对最好”的内容。

参考：[Milvus混合检索](https://milvus.io/docs/v2.5.x/multi-vector-search.md)、[Sentence Transformers Retrieve & Re-Rank](https://sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html)。

## 上下文加载

- 先取排名靠前的三个命中块的相邻块（同文档、同激活版本、前后各一块），合并后重新打分。邻块不直接无条件塞入提示词。
- MMR选择：`0.8*relevance - 0.2*max_similarity_to_selected`；重复相似度用分词集合Jaccard估计，超过0.9的块跳过。
- 默认最多3块、1800估计token；连同文件名及元数据开销计入预算。整块放入，超预算跳过，避免把表格中途截断。
- 模型窗口为8192，生成保留1024 token；上下文上限限定3000估计token，为系统指令/JSON/问题留空间。这里是保守启发式预算，不是Gemma精确token计数。
- `--context-k`、`--context-tokens`、`--candidate-k`可调整；对长文应提高候选而非无限增加模型上下文。
- `--history history.json` 接收消息列表，只取最近两个用户问题用于“它/上述”等指代消解。历史助手结论不作为知识证据，不直接写入系统角色。历史记录由调用方提供，当前不是持久会话服务。

## 回答口吻与重新生成

`--model`默认已有的Ollama `gemma3:4b`，temperature=0。三种口吻：

- professional：客观克制，先结论再依据，说明条件。
- teaching：耐心解释术语和因果，例子不能超出证据。
- concise：直接给必需结论和依据。

生成使用结构化claims，每段必须携带证据编号与连续原文摘录。程序校验编号和摘录，页码从原始元数据补回。
失败段落明确省略，原始输出和省略原因保留；这只验证可追溯性，不能证明所有事实或因果推断正确。
用户应核对原始PDF，尤其是数字、公式及类别/基线区别。

两层有界修复：

1. 引用校验全部失败：生成器最多纠正一次；不会把模型错误引文替换成“看似正确”的引文。
2. 没有合格上下文、证据不足、生成校验失败或有省略段落：策略最多重新召回一次，候选扩大至48并增加原问题关键词视角；最终证据压缩至最多两块，减少噪声。已有有效回答不会被第二次拒答覆盖。

第二轮仍没有证据则明确拒答；网络/服务不可用直接报错，避免把服务故障描述成知识不足。
`--no-recall`关闭第二轮，方便测量额外延迟。每轮记录query、候选、上下文、错误和生成状态。

## 小样本评测与调优

`已有资源/样例/eval_rag.py` 接收人工标注JSON：

```json
[
  {"question":"类别加权为什么会降低准确率？", "source":"report.pdf", "expected_pages":[7]},
  {"question":"文档外问题", "unanswerable":true}
]
```

```bash
python 已有资源/样例/eval_rag.py cases.json --output data/eval_light.json
python 已有资源/样例/eval_rag.py cases.json --reranker crossencoder --output data/eval_cross.json
```

报告包含页级MRR、最终上下文页命中率、相关问题回答率和文档外拒答率，同时保留每题完整证据。
多文档评测务必填写source，避免相同页码的其他文件被误计为正确；无source时仅适用于单文档库。
页级命中比块级宽松，回答率不代表事实正确率。需要增加真实问题和困难负例，人工标注支持段落，才能比较算法是否稳定提升。
优先调候选召回覆盖，再调重排，再调MMR和上下文，最后调整口吻；每次保持同一题集比较质量与时延。

## 本机验证记录（2026-11-22）

- 平台测试共71项通过；新增覆盖重排顺序、MMR去重/预算、低相关不生成、有界重召回、失败生成状态、历史消息隔离、同名文件、未变化跳过、失败发布保留旧版本、邻块版本过滤和模型绑定。
- 真实9页实验报告按新策略生成26块；重复导入返回UNCHANGED。
- 独立版本测试集合中，将“保修期12个月”改为“24个月”，激活版本2后查询只返回新版本。
- 四题题集包含3个报告内问题和1个文档外问题。轻量方案的页级MRR为1.0；同一新集合纯RRF召回基线为0.833。三个报告内问题均生成回答，文档外问题返回证据不足。这不是大样本准确率结论。
- CrossEncoder已在本机下载并真实执行，能将类别加权证据排在第7页并生成回答；另一个Bootstrap问题虽召回正确页，生成仍可能拒答。暂不替换默认轻量方案。
- 本机完整结果在`data/strategy_acceptance/`，其中`evaluation_final.json`、`version_result.json`与`summary.json`分别保存逐题结果、版本发布验证和摘要；data默认不进入Git。
