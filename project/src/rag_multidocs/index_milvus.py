"""Milvus index backend.

Wraps ``MilvusHybridIndex`` behind the same interface as ``KeywordIndex`` so
``IngestService`` / ``SearchService`` do not branch on storage type.

Requirements:
    pip install "rag-multidocs[milvus]"
    Milvus 2.5+ running (Docker standalone)

If Milvus or the embedding backend is unavailable, the CLI can fall back to
``KeywordIndex`` when ``fallback_to_keyword`` is true.
"""

from __future__ import annotations

from typing import Any, Iterable

from .embedding import Embedder
from .milvus_store import MilvusHybridIndex
from .model import Chunk, Document


class MilvusIndex:
    name = "milvus"

    def __init__(
        self,
        embedder: Embedder,
        *,
        uri: str = "http://localhost:19530",
        db_name: str = "rag",
        collection: str = "rag_chunks",
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
    ):
        self.embedder = embedder
        self.store = MilvusHybridIndex(
            embedder,
            uri=uri,
            db_name=db_name,
            collection=collection,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
        )

    def ensure_ready(self) -> None:
        self.store.ensure_ready()

    def add(
        self,
        chunks: Iterable[Chunk | dict],
        document: Document | None = None,
    ) -> int:
        items = list(chunks)
        if not items:
            return 0
        first = items[0] if isinstance(items[0], Chunk) else Chunk.from_dict(items[0])
        version = 1
        sha256 = ""
        if document is not None:
            version = int(document.metadata.get("version", 1) or 1)
            sha256 = str(document.metadata.get("sha256", ""))
        self.store.ingest(
            first.doc_id,
            items,
            version=version,
            source=first.source,
            sha256=sha256,
        )
        return len(items)

    def remove_doc(self, doc_id: str) -> int:
        result = self.store.delete(doc_id)
        return 1 if result.get("deleted") else 0

    def search(
        self,
        query: str,
        top_k: int = 5,
        doc_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        result = self.store.query(query, top_k=top_k, doc_ids=doc_ids)
        hits = []
        for hit in result.get("hits", []):
            hits.append(
                {
                    "chunk_id": hit.get("id"),
                    "score": hit.get("score", 0.0),
                    "doc_id": hit.get("doc_id"),
                    "source": hit.get("source", ""),
                    "text": hit.get("text", ""),
                    "page": hit.get("page", 0),
                    "section_path": hit.get("section_path", ""),
                    "content_type": hit.get("content_type", "text"),
                }
            )
        return hits

    def list_docs(self) -> list[dict[str, Any]]:
        client = self.store.client()
        if not client.has_collection(self.store.collection):
            return []
        try:
            rows = client.query(
                collection_name=self.store.collection,
                filter="",
                output_fields=["doc_id", "source", "page"],
                limit=16384,
            )
        except Exception:
            return []
        grouped: dict[str, dict[str, Any]] = {}
        for row in rows:
            doc_id = str(row.get("doc_id", ""))
            entry = grouped.setdefault(
                doc_id,
                {
                    "doc_id": doc_id,
                    "source": row.get("source", ""),
                    "chunks": 0,
                    "pages": 0,
                },
            )
            entry["chunks"] += 1
            entry["pages"] = max(entry["pages"], int(row.get("page", 0) or 0))
        return sorted(grouped.values(), key=lambda item: item["source"])

    def persist(self) -> None:
        # Milvus 自己持久化；不需要写 index.json
        return None

    def stats(self) -> dict[str, Any]:
        data = self.store.stats()
        data["backend"] = self.name
        return data

    def close(self) -> None:
        self.store.close()

    def drop(self) -> None:
        """删除测试集合；仅测试/演示使用。"""
        self.store.drop_collection()


__all__ = ["MilvusIndex"]
