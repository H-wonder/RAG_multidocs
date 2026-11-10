# Milvus 混合检索

> 来源：`任务二/docflow/services/rag_service.py`、
> `任务二/docflow/config.py`、`任务二/README.md` 第 7 节。
> 可复用实现：`代码/milvus_hybrid_index.py`、`代码/embedders.py`。
>
> 本项目目标版用 Milvus 做**主检索**（dense + BM25），但关键词倒排必须保留为
> **离线回退**：Milvus 不可用时，核心接入—索引—检索仍要能跑通。任务书把向量数据库
> 列为可选，所以不要让 Milvus 成为演示的唯一路径。
>
> 项目用 Docker 部署 Milvus：镜像 `milvusdb/milvus:v2.5.6`，容器 `milvus-standalone`，
> 可选 Attu `zilliz/attu:v2.5`。完整启动命令、挂载和健康检查见
> [07-环境搭建与依赖.md](07-环境搭建与依赖.md) 第 4.2 节。

## 1. 为什么用混合检索

纯向量检索擅长语义相似，但对专有名词、编号、表格里的精确词不敏感；
纯 BM25 擅长词面匹配，但中文问英文文档、或换个说法就召回不到。
原项目的方案是：

```text
dense（向量，IP）+ sparse（Milvus 服务端 BM25）
        ↓
WeightedRanker(0.7, 0.3) 或 RRF
        ↓
（可选）跨查询 RRF 融合
        ↓
（可选）gte-rerank 重排
        ↓
带 page / bbox / section_path 的命中
```

实测（8 条问题的 A/B）：

| 方案 | hit@5 | hit@1 |
| --- | --- | --- |
| 本地 bge-small-zh | 0.75 | — |
| DashScope text-embedding-v4 | 0.875 | — |
| DashScope + gte-rerank | 0.875 | 0.875 |

样本很小，只说明“重排可能把正确答案提到第一”，不能当成普遍结论。

## 2. 集合设计

原项目每个嵌入后端一个集合，互不干扰：

| 集合 | 嵌入 | 维度 |
| --- | --- | --- |
| `docflow_chunks_dashscope` | text-embedding-v4 | 1024 |
| `docflow_chunks_local` | bge-small-zh-v1.5 | 512 |

`代码/milvus_hybrid_index.py` 简化为一个索引实例一个集合，集合名通过
`collection` 参数配置；换嵌入模型时新建集合即可，不要混用不同维度。

### Schema

| 字段 | 类型 | 作用 |
| --- | --- | --- |
| `id` | VARCHAR(64), primary | 确定性 chunk id |
| `text` | VARCHAR(8000), analyzer=chinese | 原文；同时作为 BM25 输入 |
| `doc_id` | VARCHAR(64) | 文档过滤、按文档删除 |
| `version` | INT64 | 文档版本 |
| `content_type` | VARCHAR(32) | text/table/formula/... |
| `section_path` | VARCHAR(1024) | 章节路径 |
| `page` / `page_end` | INT64 | 页码 |
| `bbox` | VARCHAR(64) | 归一化坐标 JSON |
| `source` | VARCHAR(1024) | 相对路径/URI |
| `sha256` | VARCHAR(64) | 内容哈希 |
| `sparse` | SPARSE_FLOAT_VECTOR | 服务端 BM25 自动生成 |
| `dense` | FLOAT_VECTOR(dim) | 嵌入向量 |

### 索引

```python
# dense：HNSW + 内积（向量已归一化，内积等价余弦相似度）
{"index_type": "HNSW", "metric_type": "IP", "params": {"M": 16, "efConstruction": 64}}

# sparse：倒排 + BM25
{"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "BM25",
 "params": {"inverted_index_algo": "DAAT_MAXSCORE", "bm25_k1": 1.2, "bm25_b": 0.75}}
```

BM25 通过 Milvus 的 `Function(FunctionType.BM25)` 从 `text` 字段自动生成
`sparse` 向量，入库时不需要自己算稀疏向量。

## 3. 入库

关键点：

1. **按 doc_id 先删后插**：重跑同一文档不会产生重复；
2. **向量输入带章节前缀**：`section_path + "\n" + text` 只用于生成向量，
   存进 Milvus 的仍是原文。表格块只有数据、没有“这是哪一节”时，语义检索很难排上来；
3. **批量插入**：每批 256 行，最后 `flush`；
4. **确定性 id**：chunk id 由 (doc_id, 类型, 序号) 决定。

```python
from embedders import SentenceTransformerEmbedder
from milvus_hybrid_index import MilvusHybridIndex

index = MilvusHybridIndex(
    SentenceTransformerEmbedder("BAAI/bge-small-zh-v1.5", dim=512),
    uri="http://localhost:19530",
    db_name="course_rag",
    collection="course_chunks",
)
index.ingest(
    doc_id="a1b2c3d4e5f6",
    chunks=chunks,            # list[document_model.Chunk]
    version=1,
    source="notes/rag.md",
    sha256="…",
)
```

## 4. 查询

### 4.1 混合检索参数

```python
result = index.query(
    "RAG 的检索器如何训练？",
    top_k=5,
    doc_ids=None,             # 可选：只在指定文档内检索
    fusion="weighted",        # weighted | rrf
    expand=None,              # 可选：中文问题 → 英文检索词
    rerank=None,              # 可选：重排回调
)
```

- `limit = top_k * 4`；打开重排时至少召回 20 条，否则重排没有发挥空间；
- dense 和 sparse 各召回 `limit` 条，用 WeightedRanker 融合；
- 多个查询（原问题 + 扩写）的结果再按 RRF 融合：
  `contribution = 1 / (60 + rank + 1)`；
- 最终按 `top_k` 截断。

### 4.2 查询扩写（可选）

原项目在中文问题命中时调用 DeepSeek 把它改写成英文技术查询，原因是：
中文 query 打英文文档时，BM25 的词面匹配容易把“顺带提到关键词”的文档顶上来，
多一路英文查询可以改善语义召回。失败时静默降级，不影响主查询。

课程项目如果不接 LLM，`expand=None` 即可，不要为此引入付费依赖。

### 4.3 重排（可选）

`rerank` 是一个回调：

```python
def rerank(query: str, texts: list[str], top_n: int) -> list[int]:
    # 调用 gte-rerank / 本地 cross-encoder，返回按相关性排序的下标
    ...
```

原项目用 DashScope `gte-rerank-v2`，实测能把表格类问题从第 20 名提到第 1 名。
没有重排服务时不要假装有；直接返回原顺序即可。

### 4.4 命中结构

```json
{
  "question": "RAG 的检索器如何训练？",
  "expanded_query": "",
  "collection": "course_chunks",
  "fusion": "weighted(dense=0.7,sparse=0.3)",
  "hits": [
    {
      "rank": 1,
      "id": "…",
      "score": 0.032,
      "doc_id": "a1b2c3d4e5f6",
      "version": 1,
      "page": 3,
      "page_end": 3,
      "bbox": "[100,200,900,400]",
      "content_type": "text",
      "section_path": "3 Method > 3.2 Retrieval",
      "source": "papers/rag.pdf",
      "text": "…"
    }
  ]
}
```

`score` 在不同融合方式下不可直接比较；展示时用 `rank` 和原始相似度，不要把它
解释成“相关概率”。

## 5. 相似度查询

`similar(text, top_k)` 只做稠密检索，不融合、不重排，用于：

- 跨文档去重：看新文本和库里哪块最像；
- “找相似段落”功能；
- 调试嵌入模型是否把语义相近的文本放在一起。

返回的分数是归一化向量的内积，等价于余弦相似度（0~1）。

## 6. 运维与排错

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `集合不存在` | 还没入库或集合名写错 | 先 `ingest`，或检查 `collection` |
| 行数只增不减 | `get_collection_stats` 把未 compact 的删除行也算进去 | 用 `count(*)` 精确计数 |
| 写入后立刻查不到 | 未 flush / 索引未建完 | `ingest` 后 `flush`；查询前确认索引状态 |
| 维度不匹配 | 换嵌入模型但复用了旧集合 | 新建集合，不要混用不同维度 |
| 中文 BM25 效果差 | analyzer 未配置 | `text` 字段设置 `analyzer_params={"type": "chinese"}` |
| 检索结果重复 | 没按 doc_id 先删后插 | 使用确定性 id + 删除旧版本 |

## 7. 对课程项目的建议

| 场景 | 建议 |
| --- | --- |
| 项目目标版主路径 | Milvus 混合检索（dense + BM25）；Docker 起 `milvus-standalone` |
| Milvus 不可用 / CI 环境 | 切关键词倒排，结果里标注 `index=keyword`，不能崩溃 |
| 离线演示 | 用 `HashingEmbedder` 做链路冒烟，但 README 要说明它没有语义能力 |
| 真实语义检索 | 本地 bge-small-zh + Milvus；模型约 100MB，集合维度 512 |
| 展示混合检索 | dense + BM25 + WeightedRanker；这是本文件的主要价值 |
| 展示重排 | 需要额外 rerank 服务；没有就明确写“未实现” |
| CI 测试 | 单元测试用 fake/离线路径；Milvus 集成测试标记 `integration`，无服务时跳过 |

**范围提醒**：课程任务书说“不做复杂向量检索”。本项目把 Milvus 作为目标版主检索，
但核心工程要求（核心场景、扩展机制、测试、协作、发布）仍然要完整；同时保留
关键词倒排作为回退，不能用“我们用了 Milvus”替代核心闭环。
