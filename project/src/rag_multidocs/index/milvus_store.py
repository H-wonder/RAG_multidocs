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

import json
import hashlib
import base64
import zlib
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

from ..model import Chunk
from ..providers.embedding import Embedder

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
    "chunk_index",
    "block_ids",
    "ocr_mode",
    "needs_review",
    "origin",
    "warnings",
    "metadata",
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
        legacy_readonly: bool = False,
    ):
        if _IMPORT_ERROR is not None:
            raise RuntimeError(
                "未安装 pymilvus；请先 pip install 'pymilvus>=2.5'"
            ) from _IMPORT_ERROR
        self.legacy_readonly = legacy_readonly
        self.available_fields = None
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
        from .contracts import model_identity

        identity = (
            "rag-pure-milvus-v1:"
            + hashlib.sha256(
                json.dumps(model_identity(self.embedder), sort_keys=True).encode()
            ).hexdigest()
        )
        if client.has_collection(name):
            existing = client.describe_collection(name)
            self.available_fields = {f["name"] for f in existing["fields"]}
            if self.legacy_readonly:
                dense = next(f for f in existing["fields"] if f["name"] == "dense")
                if int(dense["params"]["dim"]) != self.embedder.dim:
                    raise ValueError("INDEX_DIMENSION_MISMATCH")
                return name
            if existing.get("description") != identity:
                raise ValueError(
                    "INDEX_MODEL_MISMATCH: existing/legacy collection identity differs; choose a new collection"
                )
            dense = next(f for f in existing["fields"] if f["name"] == "dense")
            if int(dense["params"]["dim"]) != self.embedder.dim:
                raise ValueError("INDEX_DIMENSION_MISMATCH")
            if not set(OUTPUT_FIELDS + ["id", "dense", "sparse"]) <= self.available_fields:
                raise ValueError("INDEX_SCHEMA_MISMATCH")
            return name

        if self.legacy_readonly:
            raise ValueError("LEGACY_COLLECTION_NOT_FOUND")
        dim = int(self.embedder.dim)
        schema = client.create_schema(
            auto_id=False, enable_dynamic_field=False, description=identity
        )
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
        schema.add_field("metadata", DataType.JSON)
        schema.add_field("content_type", DataType.VARCHAR, max_length=32)
        schema.add_field("section_path", DataType.VARCHAR, max_length=1024)
        schema.add_field("page", DataType.INT64)
        schema.add_field("page_end", DataType.INT64)
        schema.add_field("bbox", DataType.VARCHAR, max_length=256)
        schema.add_field("chunk_index", DataType.INT64)
        schema.add_field("block_ids", DataType.VARCHAR, max_length=4096)
        schema.add_field("ocr_mode", DataType.VARCHAR, max_length=32)
        schema.add_field("needs_review", DataType.BOOL)
        schema.add_field("origin", DataType.VARCHAR, max_length=64)
        schema.add_field("warnings", DataType.VARCHAR, max_length=8192)
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
        client.create_collection(collection_name=name, schema=schema, index_params=index_params)
        return name

    def ensure_ready(self) -> str:
        prepare = getattr(self.embedder, "prepare", None)
        if prepare:
            prepare()
        if not self.legacy_readonly:
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
        replace_document: bool = True,
        document_metadata: dict | None = None,
        document_payload: dict | None = None,
    ) -> dict[str, Any]:
        """Replace one current document in Milvus; no catalog/version activation."""
        if self.legacy_readonly:
            raise ValueError("INDEX_READ_ONLY: explicit reingestion into a new collection required")
        rows_in = [self._as_dict(chunk) for chunk in chunks]
        rows_in = [row for row in rows_in if str(row.get("text", "")).strip()]
        if not rows_in:
            raise ValueError("chunks 为空：没有可入库的内容")

        name = self.ensure_ready()
        client = self.client()

        # 向量输入带章节上下文：表格块只有数据、没有“这是哪一节”时，
        # 语义检索很难排上来。存进 Milvus 的仍是原文，只有向量看到前缀。
        from ..model import embedding_text

        embed_inputs = [embedding_text(row) for row in rows_in]
        vectors = self.embedder.embed(embed_inputs, query=False)
        if len(vectors) != len(rows_in):
            raise RuntimeError(f"嵌入数量不匹配: {len(vectors)} != {len(rows_in)}")

        rows = [
            {
                "id": str(row["id"]),
                "text": str(row["text"]),
                "doc_id": doc_id,
                "version": int(version),
                "content_type": str(row.get("content_type", "text"))[:32],
                "section_path": str(row.get("section_path", "")),
                "page": int(row.get("page", 0) or 0),
                "page_end": int(row.get("page_end", 0) or 0),
                "bbox": str(row.get("bbox", "")),
                "origin": str(
                    row.get("origin", row.get("metadata", {}).get("origin", "connector"))
                ),
                "warnings": json.dumps(
                    row.get("warnings", row.get("metadata", {}).get("warnings", [])),
                    ensure_ascii=False,
                ),
                "chunk_index": int(row.get("index", 0)),
                "block_ids": json.dumps(row.get("block_ids", []), ensure_ascii=False),
                "metadata": {**row.get("metadata", {}), "_document": dict(document_metadata or {})},
                "ocr_mode": str(row.get("ocr_mode", row.get("metadata", {}).get("ocr_mode", ""))),
                "needs_review": bool(
                    row.get("needs_review", row.get("metadata", {}).get("needs_review", False))
                ),
                "source": str(source or row.get("source", "")),
                "sha256": str(sha256)[:64],
                "dense": vector,
            }
            for row, vector in zip(rows_in, vectors)
        ]

        self.pack_document_metadata(rows, document_metadata or {}, document_payload)

        limits = {
            "id": 64,
            "doc_id": 64,
            "text": 8000,
            "source": 1024,
            "section_path": 1024,
            "content_type": 32,
            "bbox": 256,
            "block_ids": 4096,
            "ocr_mode": 32,
            "sha256": 64,
            "origin": 64,
            "warnings": 8192,
        }
        if any(
            len(str(row[k]).encode("utf8")) > limit for row in rows for k, limit in limits.items()
        ):
            raise ValueError("VARCHAR_BYTE_LIMIT: no silent truncation")
        if any(len(row["text"].encode()) > 8000 for row in rows):
            raise ValueError("chunk exceeds 8000 bytes")
        if any(len(v) != self.embedder.dim for v in vectors):
            raise ValueError("embedding dimension mismatch")
        expression = f"doc_id == {json.dumps(doc_id)}"
        if not replace_document:
            expression += f" and version == {int(version)}"
        old_rows = self.document_rows(doc_id, include_dense=True)
        old_ids, new_ids = {r["id"] for r in old_rows}, {r["id"] for r in rows}
        try:
            for start in range(0, len(rows), batch_size):
                client.upsert(name, rows[start:start + batch_size])
            # Remove only rows left over from the replaced document, after the
            # complete replacement is accepted. Embed/validation failures above
            # cannot remove the old document.
            stale = old_ids - new_ids
            if stale:
                client.delete(name, ids=sorted(stale))
            client.flush(name)
        except Exception:
            # Recover ordinary write failures using the same Milvus records.
            # This is not a crash-atomic multi-request transaction.
            for start in range(0, len(old_rows), batch_size):
                client.upsert(name, old_rows[start:start + batch_size])
            added = new_ids - old_ids
            if added:
                client.delete(name, ids=sorted(added))
            client.flush(name)
            raise
        return {
            "doc_id": doc_id,
            "version": version,
            "collection": name,
            "chunks": len(rows),
        }

    @staticmethod
    def pack_document_metadata(rows, metadata, payload):
        encoded = (base64.b64encode(zlib.compress(json.dumps(payload, ensure_ascii=False).encode())).decode()
                   if payload is not None else '')
        pieces = [encoded[i:i + 8192] for i in range(0, len(encoded), 8192)]
        if len(pieces) > len(rows):
            raise ValueError('DOCUMENT_METADATA_EXCEEDS_CHUNK_CAPACITY')
        checksum = hashlib.sha256(encoded.encode()).hexdigest()
        for i, row in enumerate(sorted(rows, key=lambda r: r.get('chunk_index', 0))):
            record = dict(metadata)
            if i < len(pieces):
                record['_payload'] = {'part': i, 'parts': len(pieces), 'data': pieces[i], 'sha256': checksum}
            row['metadata'] = {**row.get('metadata', {}), '_document': record}

    def update_document_metadata(self, doc_id, metadata, payload):
        rows = self.document_rows(doc_id, include_dense=True)
        self.pack_document_metadata(rows, metadata, payload)
        for start in range(0, len(rows), 128):
            self.client().upsert(self.collection, rows[start:start + 128])
        self.client().flush(self.collection)

    def document_rows(self, doc_id=None, *, include_dense=False):
        fields = ["id"] + OUTPUT_FIELDS + (["dense"] if include_dense else [])
        expr = f"doc_id == {json.dumps(doc_id)}" if doc_id is not None else ""
        rows = []
        iterator = self.client().query_iterator(self.collection, filter=expr,
                                               output_fields=fields, batch_size=512,
                                               consistency_level="Strong")
        try:
            while True:
                batch = iterator.next()
                if not batch:
                    break
                rows.extend({**row, **({"dense": list(row["dense"])} if include_dense else {})}
                            for row in batch)
        finally:
            iterator.close()
        return rows

    def delete(self, doc_id: str) -> dict[str, Any]:
        if self.legacy_readonly:
            raise ValueError("INDEX_READ_ONLY")
        client = self.client()
        if not client.has_collection(self.collection):
            return {"doc_id": doc_id, "deleted": False, "reason": "collection 不存在"}
        client.delete(collection_name=self.collection, filter=f'doc_id == "{doc_id}"')
        client.flush(self.collection)
        return {"doc_id": doc_id, "deleted": True}

    def drop_collection(self) -> dict[str, Any]:
        if self.legacy_readonly:
            raise ValueError("INDEX_READ_ONLY")
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
        filter_expr: str | None = None,
        ann_limit: int | None = None,
        timeout_s: float | None = None,
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
        limit = (
            ann_limit if ann_limit is not None else (max(top_k * 4, 20) if rerank else top_k * 4)
        )
        if limit < top_k or limit > 10000:
            raise ValueError("invalid ANN candidate limit")

        expr = filter_expr
        if doc_ids:
            clause = f"doc_id in {json.dumps(doc_ids)}"
            expr = f"({expr}) and ({clause})" if expr else clause

        merged: dict[str, dict[str, Any]] = {}
        for text_query in queries:
            query_vector = self.embedder.embed([text_query], query=True)[0]
            dense_req = AnnSearchRequest(
                data=[query_vector],
                anns_field="dense",
                param={"metric_type": self.metric_type, "params": {"ef": max(64, limit)}},
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
                output_fields=[
                    f
                    for f in OUTPUT_FIELDS
                    if self.available_fields is None or f in self.available_fields
                ],
                consistency_level="Strong",
                **({"timeout": timeout_s} if timeout_s is not None else {}),
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
                        **{
                            key: entity.get(key)
                            for key in [
                                "chunk_index",
                                "block_ids",
                                "ocr_mode",
                                "needs_review",
                                "origin",
                                "warnings",
                                "metadata",
                            ]
                        },
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
