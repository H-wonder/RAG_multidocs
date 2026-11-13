"""Milvus 混合检索索引：稠密向量 + 服务端 BM25，带来源字段。

来源：``任务二/docflow/services/rag_service.py``。保留核心检索设计，做了以下适配：
- 从 FastAPI 服务抽成可嵌入的 ``MilvusHybridIndex`` 类；
- 嵌入后端通过 ``Embedder`` 协议注入，不再写死 local/dashscope；
- 入库接受 ``document_model.Chunk``，与 Connector/切块层解耦；
- 查询扩写与重排改成可选回调，不配置就不调用外部 LLM。

依赖（仅本文件需要）：
    pip install "pymilvus>=2.5"        # text BM25 Function 需要 Milvus 2.5+
    pip install sentence-transformers  # 仅本地嵌入后端需要

项目用 Docker 部署：镜像 milvusdb/milvus:v2.5.6，容器 milvus-standalone。
完整启动命令（embedded etcd + 本地存储 + Attu）见
``文档/07-环境搭建与依赖.md`` 第 4.2 节。
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

try:
    from pymilvus import (
        AnnSearchRequest,
        DataType,
        Function,
        FunctionType,
        MilvusClient,
        RRFRanker,
        WeightedRanker,
    )

    _IMPORT_ERROR: Exception | None = None
except ImportError as exc:  # 允许在没装 pymilvus 的环境里 import 本模块
    AnnSearchRequest = DataType = Function = FunctionType = None  # type: ignore
    MilvusClient = RRFRanker = WeightedRanker = None  # type: ignore
    _IMPORT_ERROR = exc

from .model import Chunk
from .embedding import Embedder

OUTPUT_FIELDS = [
    "doc_id",
    "version",
    "text",
    "content_type",
    "section_path",
    "page",
    "page_end",
    "bbox",
    "source",
    "sha256",
]

# 重排回调：给定查询和候选文本，返回按相关性排序的下标
RerankFn = Callable[[str, list[str], int], list[int]]
# 查询扩写回调：给定原问题，返回补充查询（可为空字符串）
ExpandFn = Callable[[str], str]


class MilvusHybridIndex:
    def __init__(
        self,
        embedder: Embedder,
        *,
        uri: str = "http://localhost:19530",
        db_name: str = "rag",
        collection: str = "rag_chunks",
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        metric_type: str = "IP",
    ):
        if _IMPORT_ERROR is not None:
            raise RuntimeError(
                "未安装 pymilvus；请先 pip install 'pymilvus>=2.5'"
            ) from _IMPORT_ERROR
        self.embedder = embedder
        self.uri = uri
        self.db_name = db_name
        self.collection = collection
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.metric_type = metric_type
        self._client: MilvusClient | None = None

    # ---------------------------------------------------------------- 连接

    def _admin_client(self) -> "MilvusClient":
        return MilvusClient(uri=self.uri)

    def client(self) -> "MilvusClient":
        if self._client is None:
            self._client = MilvusClient(uri=self.uri, db_name=self.db_name)
        return self._client

    def ensure_database(self) -> None:
        admin = self._admin_client()
        try:
            if self.db_name not in admin.list_databases():
                admin.create_database(self.db_name)
        finally:
            admin.close()

    def ensure_collection(self) -> str:
        """幂等建集合。每次都向 Milvus 确认，不信任进程内缓存。

        集合可能在服务运行期间被外部删除（清理数据、重建环境），
        只缓存“已就绪”会让后续写入落到幽灵集合上。
        """
        client = self.client()
        name = self.collection
        if client.has_collection(name):
            return name

        dim = int(self.embedder.dim)
        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field(
            "text",
            DataType.VARCHAR,
            max_length=8000,
            enable_analyzer=True,
            analyzer_params={"type": "chinese"},
        )
        schema.add_field("doc_id", DataType.VARCHAR, max_length=64)
        schema.add_field("version", DataType.INT64)
        schema.add_field("content_type", DataType.VARCHAR, max_length=32)
        schema.add_field("section_path", DataType.VARCHAR, max_length=1024)
        schema.add_field("page", DataType.INT64)
        schema.add_field("page_end", DataType.INT64)
        schema.add_field("bbox", DataType.VARCHAR, max_length=64)
        schema.add_field("source", DataType.VARCHAR, max_length=1024)
        schema.add_field("sha256", DataType.VARCHAR, max_length=64)
        schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field("dense", DataType.FLOAT_VECTOR, dim=dim)

        # 服务端 BM25：text 字段分词后自动生成 sparse 向量，入库只需给 text。
        schema.add_function(
            Function(
                name="text_bm25",
                function_type=FunctionType.BM25,
                input_field_names=["text"],
                output_field_names=["sparse"],
            )
        )

        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="dense",
            index_type="HNSW",
            metric_type=self.metric_type,
            params={"M": 16, "efConstruction": 64},
        )
        index_params.add_index(
            field_name="sparse",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
            params={
                "inverted_index_algo": "DAAT_MAXSCORE",
                "bm25_k1": 1.2,
                "bm25_b": 0.75,
            },
        )
        client.create_collection(
            collection_name=name, schema=schema, index_params=index_params
        )
        return name

    def ensure_ready(self) -> str:
        self.ensure_database()
        return self.ensure_collection()

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    # ---------------------------------------------------------------- 入库

    @staticmethod
    def _as_dict(chunk: Chunk | dict[str, Any]) -> dict[str, Any]:
        if isinstance(chunk, Chunk):
            return chunk.as_dict()
        return dict(chunk)

    def ingest(
        self,
        doc_id: str,
        chunks: Iterable[Chunk | dict[str, Any]],
        *,
        version: int = 1,
        source: str = "",
        sha256: str = "",
        batch_size: int = 256,
    ) -> dict[str, Any]:
        """按 doc_id 先删后插，保证重跑幂等。"""
        rows_in = [self._as_dict(chunk) for chunk in chunks]
        rows_in = [row for row in rows_in if str(row.get("text", "")).strip()]
        if not rows_in:
            raise ValueError("chunks 为空：没有可入库的内容")

        name = self.ensure_ready()
        client = self.client()

        # 向量输入带章节上下文：表格块只有数据、没有“这是哪一节”时，
        # 语义检索很难排上来。存进 Milvus 的仍是原文，只有向量看到前缀。
        embed_inputs = [
            f"{row.get('section_path')}\n{row['text']}"
            if row.get("section_path")
            else str(row["text"])
            for row in rows_in
        ]
        vectors = self.embedder.embed(embed_inputs, query=False)
        if len(vectors) != len(rows_in):
            raise RuntimeError(
                f"嵌入数量不匹配: {len(vectors)} != {len(rows_in)}"
            )

        rows = [
            {
                "id": str(row["id"]),
                "text": str(row["text"]),
                "doc_id": doc_id,
                "version": int(version),
                "content_type": str(row.get("content_type", "text"))[:32],
                "section_path": str(row.get("section_path", ""))[:1000],
                "page": int(row.get("page", 0) or 0),
                "page_end": int(row.get("page_end", 0) or 0),
                "bbox": str(row.get("bbox", ""))[:60],
                "source": str(source or row.get("source", ""))[:1000],
                "sha256": str(sha256)[:64],
                "dense": vector,
            }
            for row, vector in zip(rows_in, vectors)
        ]

        client.delete(collection_name=name, filter=f'doc_id == "{doc_id}"')
        for start in range(0, len(rows), batch_size):
            client.insert(name, rows[start : start + batch_size])
        client.flush(name)
        return {
            "doc_id": doc_id,
            "version": version,
            "collection": name,
            "chunks": len(rows),
        }

    def delete(self, doc_id: str) -> dict[str, Any]:
        client = self.client()
        if not client.has_collection(self.collection):
            return {"doc_id": doc_id, "deleted": False, "reason": "collection 不存在"}
        client.delete(collection_name=self.collection, filter=f'doc_id == "{doc_id}"')
        client.flush(self.collection)
        return {"doc_id": doc_id, "deleted": True}

    def drop_collection(self) -> dict[str, Any]:
        """删除整个集合。演示/测试清理用，生产环境慎用。"""
        client = self.client()
        existed = client.has_collection(self.collection)
        if existed:
            client.drop_collection(self.collection)
        return {"collection": self.collection, "dropped": existed}

    # ---------------------------------------------------------------- 检索

    def _count(self) -> int:
        """精确行数。get_collection_stats 的 row_count 会把未 compact 的删除行也算进去。"""
        client = self.client()
        if not client.has_collection(self.collection):
            return 0
        try:
            rows = list(
                client.query(
                    collection_name=self.collection,
                    filter="",
                    output_fields=["count(*)"],
                )
            )
            return int(rows[0]["count(*)"]) if rows else 0
        except Exception:
            return -1

    def query(
        self,
        question: str,
        top_k: int = 5,
        *,
        doc_ids: list[str] | None = None,
        fusion: str = "weighted",
        expand: ExpandFn | None = None,
        rerank: RerankFn | None = None,
    ) -> dict[str, Any]:
        """混合检索：dense(IP) + 服务端 BM25，WeightedRanker 或 RRF 融合。

        - ``fusion="weighted"``：稠密权重默认 0.7，避免“顺带提到关键词”的块压过正文；
        - ``fusion="rrf"``：两路等权，适合无法校准权重的场景；
        - ``expand``：可选查询扩写（例如中文问题 → 英文检索词）；
        - ``rerank``：可选重排回调，打开后先多召回候选再重排。
        """
        if not question.strip():
            raise ValueError("question 为空")
        client = self.client()
        if not client.has_collection(self.collection):
            raise RuntimeError(f"集合 {self.collection} 不存在，请先入库")

        expanded = expand(question) if expand else ""
        queries = [q for q in (question, expanded) if q]
        limit = max(top_k * 4, 20) if rerank else top_k * 4

        expr = None
        if doc_ids:
            quoted = ", ".join(f'"{doc_id}"' for doc_id in doc_ids)
            expr = f"doc_id in [{quoted}]"

        merged: dict[str, dict[str, Any]] = {}
        for text_query in queries:
            query_vector = self.embedder.embed([text_query], query=True)[0]
            dense_req = AnnSearchRequest(
                data=[query_vector],
                anns_field="dense",
                param={"metric_type": self.metric_type, "params": {"ef": 64}},
                limit=limit,
                expr=expr,
            )
            # BM25 直接用查询文本，分词在 Milvus 服务端做
            sparse_req = AnnSearchRequest(
                data=[text_query],
                anns_field="sparse",
                param={"metric_type": "BM25"},
                limit=limit,
                expr=expr,
            )
            ranker = (
                WeightedRanker(self.dense_weight, self.sparse_weight)
                if fusion == "weighted"
                else RRFRanker(100)
            )
            results = client.hybrid_search(
                collection_name=self.collection,
                reqs=[dense_req, sparse_req],
                ranker=ranker,
                limit=limit,
                output_fields=OUTPUT_FIELDS,
            )
            for rank, hit in enumerate(results[0] if results else []):
                hit_id = hit.get("id")
                entity = hit.get("entity", {})
                contribution = 1.0 / (60 + rank + 1)
                if hit_id in merged:
                    merged[hit_id]["score"] += contribution
                else:
                    merged[hit_id] = {
                        "id": hit_id,
                        "score": contribution,
                        "doc_id": entity.get("doc_id"),
                        "version": entity.get("version"),
                        "page": entity.get("page"),
                        "page_end": entity.get("page_end"),
                        "bbox": entity.get("bbox"),
                        "content_type": entity.get("content_type"),
                        "section_path": entity.get("section_path"),
                        "source": entity.get("source"),
                        "text": entity.get("text"),
                    }

        hits = sorted(merged.values(), key=lambda item: item["score"], reverse=True)
        hits = hits[: max(limit, 20)]
        for rank, hit in enumerate(hits, start=1):
            hit["rank"] = rank

        if rerank and hits:
            order = rerank(expanded or question, [hit["text"] for hit in hits], top_k)
            hits = [hits[index] for index in order]
            for rank, hit in enumerate(hits, start=1):
                hit["rank"] = rank
        hits = hits[:top_k]

        return {
            "question": question,
            "expanded_query": expanded,
            "collection": self.collection,
            "fusion": (
                f"weighted(dense={self.dense_weight},sparse={self.sparse_weight})"
                if fusion == "weighted"
                else "rrf"
            ),
            "hits": hits,
        }

    def similar(self, text: str, top_k: int = 5) -> dict[str, Any]:
        """纯稠密相似度查询，给跨文档去重或“找相似段落”用。"""
        client = self.client()
        name = self.ensure_collection()
        vector = self.embedder.embed([text], query=False)[0]
        results = client.search(
            collection_name=name,
            data=[vector],
            anns_field="dense",
            search_params={"metric_type": self.metric_type, "params": {"ef": 64}},
            limit=top_k,
            output_fields=["doc_id", "page", "content_type"],
        )
        hits = []
        for row in results[0] if results else []:
            entity = row.get("entity", {})
            hits.append(
                {
                    "id": row.get("id"),
                    "doc_id": entity.get("doc_id"),
                    "page": entity.get("page"),
                    "content_type": entity.get("content_type"),
                    "score": round(float(row.get("distance", 0)), 4),
                }
            )
        return {"hits": hits}

    def stats(self) -> dict[str, Any]:
        return {
            "collection": self.collection,
            "exists": self.client().has_collection(self.collection),
            "rows": self._count(),
        }


__all__ = ["MilvusHybridIndex", "OUTPUT_FIELDS"]
