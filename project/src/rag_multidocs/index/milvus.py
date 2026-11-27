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

import json
import base64
import hashlib
import zlib
from typing import Any, Iterable

from ..providers.embedding import Embedder
from .milvus_store import MilvusHybridIndex
from ..model import Chunk, Document


class MilvusIndex:
    name = "milvus"
    replaces_document = True
    versioned_publication = False

    def __init__(
        self,
        embedder: Embedder,
        *,
        uri: str = "http://localhost:19530",
        db_name: str = "rag",
        collection: str = "rag_chunks",
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        fusion: str = "weighted",
        legacy_readonly: bool = False,
    ):
        self.version_catalog = None
        self.fusion = fusion
        self.legacy_readonly = legacy_readonly
        self.embedder = embedder
        self.store = MilvusHybridIndex(
            embedder,
            uri=uri,
            db_name=db_name,
            collection=collection,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            legacy_readonly=legacy_readonly,
        )

    def ensure_ready(self) -> None:
        self.store.ensure_ready()

    def add(
        self,
        chunks: Iterable[Chunk | dict],
        document: Document | None = None,
    ) -> int:
        if self.legacy_readonly:
            raise ValueError("INDEX_READ_ONLY")
        items = list(chunks)
        if not items:
            return 0
        first = items[0] if isinstance(items[0], Chunk) else Chunk.from_dict(items[0])
        version = 1
        sha256 = ""
        if document is not None:
            version = int(document.metadata.get("version", 1) or 1)
            sha256 = str(document.metadata.get("sha256", ""))
        metadata = ({k: document.metadata.get(k) for k in (
            "sha256", "raw_sha256", "content_sha256", "processing_fingerprint", "ingest_key", "ingest_signature", "encoding", "connector")}
                    if document else {})
        metadata.update(title=document.title if document else "", chunk_count=len(items),
                        media_type=document.media_type if document else "")
        current = self.document(first.doc_id)
        if (current and metadata.get("processing_fingerprint")
                and current.get("metadata", {}).get("processing_fingerprint") == metadata["processing_fingerprint"]
                and len(self.store.document_rows(first.doc_id)) == len(items)):
            if (current.get('metadata', {}).get('ingest_key') != metadata.get('ingest_key')
                    or current.get('metadata', {}).get('ingest_signature') != metadata.get('ingest_signature')
                    or 'content' not in current):
                self.store.update_document_metadata(first.doc_id, metadata, document.to_dict() if document else None)
            return 0
        self.store.ingest(
            first.doc_id,
            items,
            version=version,
            source=first.source,
            sha256=sha256,
            document_metadata=metadata,
            document_payload=document.to_dict() if document else None,
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
        filters: dict | None = None,
        snapshot=None,
        fusion=None,
        ann_limit=None,
        timeout_s=None,
    ) -> list[dict[str, Any]]:
        expression = self.version_catalog.filter_expr(snapshot) if self.version_catalog else None
        for key, value in (filters or {}).items():
            if key not in {"page", "content_type", "doc_id"}:
                raise ValueError("unsupported filter")
            clause = f"{key} == {json.dumps(value)}"
            expression = f"({expression}) and ({clause})" if expression else clause
        result = self.store.query(
            query,
            top_k=top_k,
            doc_ids=doc_ids,
            fusion=fusion or self.fusion,
            filter_expr=expression,
            ann_limit=ann_limit,
            timeout_s=timeout_s,
        )
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
                    **{
                        key: hit.get(key)
                        for key in [
                            "version",
                            "page_end",
                            "bbox",
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
            )
        for hit in hits:
            for key in ("block_ids", "warnings"):
                if isinstance(hit.get(key), str):
                    hit[key] = json.loads(hit[key])
        return hits

    def identity(self):
        from .contracts import model_identity

        return {
            "backend": "milvus",
            "schema": "rag-pure-milvus-v1",
            "uri": self.store.uri,
            "database": self.store.db_name,
            "collection": self.store.collection,
            "embedding": model_identity(self.embedder),
        }

    def verify_version(self, doc_id, version, expected):
        if len(expected) > 16384:
            raise ValueError("PUBLICATION_TOO_LARGE: verification limit 16384 chunks")
        rows = self.store.client().query(
            self.store.collection,
            filter=f"doc_id == {json.dumps(doc_id)} and version == {int(version)}",
            output_fields=["id", "text", "source"],
            limit=len(expected),
            consistency_level="Strong",
        )
        return {r["id"]: (r["text"], r["source"]) for r in rows} == {
            r["id"]: (r["text"], r["source"]) for r in expected
        }

    def stage_version(self, document, chunks, version):
        if self.legacy_readonly:
            raise ValueError("INDEX_READ_ONLY")
        self.store.ingest(
            document.doc_id,
            chunks,
            version=version,
            source=document.source,
            sha256=document.metadata.get("sha256", ""),
            replace_document=False,
        )

    def count_version(self, doc_id, version):
        return self.store.client().query(
            self.store.collection,
            filter=f"doc_id == {json.dumps(doc_id)} and version == {int(version)}",
            output_fields=["count(*)"],
            consistency_level="Strong",
        )[0]["count(*)"]

    def remove_version(self, doc_id, version):
        if self.legacy_readonly:
            raise ValueError("INDEX_READ_ONLY")
        self.store.client().delete(
            self.store.collection,
            filter=f"doc_id == {json.dumps(doc_id)} and version == {int(version)}",
        )
        self.store.client().flush(self.store.collection)

    def list_docs(self):
        grouped = {}
        for row in self.store.document_rows():
            meta = row.get("metadata", {}).get("_document", {})
            entry = grouped.setdefault(row["doc_id"], {
                "doc_id": row["doc_id"], "source": row.get("source", ""),
                "title": meta.get("title", ""), "chunks": 0, "pages": 0,
            })
            entry["chunks"] += 1
            entry["pages"] = max(entry["pages"], row.get("page", 0))
        return sorted(grouped.values(), key=lambda row: row["source"])

    def source_records(self):
        return self.list_docs()

    def document(self, doc_id):
        rows = self.store.document_rows(doc_id)
        if not rows:
            return None
        row = rows[0]
        metadata = {k: v for k, v in row.get('metadata', {}).get('_document', {}).items() if k != '_payload'}
        parts = [r.get('metadata', {}).get('_document', {}).get('_payload') for r in rows]
        parts = sorted((part for part in parts if part), key=lambda part: part['part'])
        if parts:
            encoded = ''.join(part['data'] for part in parts)
            if (len(parts) != parts[0]['parts'] or [part['part'] for part in parts] != list(range(len(parts)))
                    or hashlib.sha256(encoded.encode()).hexdigest() != parts[0]['sha256']):
                raise ValueError('DOCUMENT_METADATA_INCOMPLETE')
            payload = json.loads(zlib.decompress(base64.b64decode(encoded)))
            return {**payload, 'metadata': {**payload.get('metadata', {}), **metadata}, 'version': None}
        return {"doc_id": doc_id, "source": row.get("source", ""),
                "title": metadata.get('title', ''), "metadata": metadata, "version": None}

    def cached_ingest(self, key, signature):
        found = {}
        for row in self.store.document_rows():
            meta = row.get("metadata", {}).get("_document", {})
            if meta.get("ingest_key") == key and meta.get("ingest_signature") == signature:
                found.setdefault(row["doc_id"], []).append(row)
        if not found or any(len(rows) != rows[0]["metadata"]["_document"].get("chunk_count")
                            for rows in found.values()):
            return None
        return [{"doc_id": did, "source": rows[0]["source"], "chunks": 0,
                 "status": "unchanged", "version": None, "warnings": []}
                for did, rows in found.items()]

    def neighbors(self, chunk_ids, snapshot=None):
        # The isolated pure collection contains current rows only.
        wanted = set(chunk_ids)
        rows = self.store.document_rows()
        seeds = [(r["doc_id"], r.get("chunk_index", 0)) for r in rows if r["id"] in wanted]
        return [{**r, "chunk_id": r["id"]} for r in rows
                if any(r["doc_id"] == did and abs(r.get("chunk_index", 0) - pos) <= 1
                       for did, pos in seeds)]

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
