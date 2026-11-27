"""轻量关键词索引（BM25），纯标准库。

用途：
- 课程项目的最低检索能力，不依赖 Milvus/嵌入模型；
- Milvus 不可用时的离线回退；
- 测试里作为确定性的检索后端。

分词策略：
- 英文/数字：小写词；
- 中文：单字 + 连续中文串的二元组（bigram）。
这样不需要第三方分词器，也能覆盖中文关键词查询。

数据只存在内存；`save()` / `load()` 用 JSON 落盘，适合课程规模的数据集。

实现的是 BM25（k1=1.2, b=0.75）：
    score(q, d) = Σ idf(t) * tf(t,d)*(k1+1) / (tf(t,d) + k1*(1-b+b*|d|/avgdl))
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import threading
from functools import wraps
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from ..model import Chunk, embedding_text

_CJK_RUN = re.compile(r"[一-鿿]+")
_WORD = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """英文按词，中文按单字 + 二元组。"""
    text = (text or "").lower()
    tokens: list[str] = _WORD.findall(text)
    for run in _CJK_RUN.findall(text):
        tokens.extend(run)
        tokens.extend(run[index : index + 2] for index in range(len(run) - 1))
    return tokens


def _synchronized(method):
    @wraps(method)
    def guarded(self, *args, **kwargs):
        catalog = self.version_catalog
        with catalog.lock if catalog else self._lock:
            return method(self, *args, **kwargs)

    return guarded


class KeywordIndex:
    name = "keyword"

    def __init__(self, k1: float = 1.2, b: float = 0.75, path: str | Path | None = None):
        self.k1 = k1
        self.b = b
        self.path = Path(path).resolve() if path else None
        self.version_catalog = None
        self._lock = threading.RLock()
        self.chunks: dict[str, dict] = {}
        self.postings: dict[str, dict[str, int]] = {}
        self.doc_len: dict[str, int] = {}

    # ---------------------------------------------------------------- 写入

    @staticmethod
    def _as_dict(chunk: Chunk | dict) -> dict:
        if isinstance(chunk, Chunk):
            return chunk.as_dict()
        return dict(chunk)

    @_synchronized
    def add(
        self,
        chunks: Iterable[Chunk | dict],
        document=None,
    ) -> int:
        """加入一批块；同 id 的块会覆盖旧内容。返回实际加入数量。

        ``document`` 仅为与 Milvus 后端保持同一接口；关键词索引不需要它。
        """
        added = 0
        for chunk in chunks:
            data = self._as_dict(chunk)
            chunk_id = str(data["id"])
            if not str(data.get("text", "")).strip():
                continue
            if chunk_id in self.chunks:
                self._remove_chunk(chunk_id)

            # 章节路径也参与索引：命中“表格”时，即使正文没写“切块”，
            # 也能通过 section_path（如“切块策略 > 表格”）召回。
            searchable = embedding_text(data)
            tokens = tokenize(searchable)
            counts = Counter(tokens)
            self.chunks[chunk_id] = data
            self.doc_len[chunk_id] = len(tokens)
            for token, frequency in counts.items():
                self.postings.setdefault(token, {})[chunk_id] = frequency
            added += 1
        return added

    @_synchronized
    def _remove_chunk(self, chunk_id: str) -> None:
        self.chunks.pop(chunk_id, None)
        self.doc_len.pop(chunk_id, None)
        for token in list(self.postings):
            self.postings[token].pop(chunk_id, None)
            if not self.postings[token]:
                del self.postings[token]

    @_synchronized
    def remove_doc(self, doc_id: str) -> int:
        """删除一篇文档的所有块。返回删除数量。"""
        ids = [chunk_id for chunk_id, data in self.chunks.items() if data.get("doc_id") == doc_id]
        for chunk_id in ids:
            self._remove_chunk(chunk_id)
        return len(ids)

    @_synchronized
    def clear(self) -> None:
        self.chunks.clear()
        self.postings.clear()
        self.doc_len.clear()

    @_synchronized
    def _refresh(self):
        if self.path and self.path.exists():
            loaded = type(self).load(self.path)
            self.chunks, self.postings, self.doc_len = (
                loaded.chunks,
                loaded.postings,
                loaded.doc_len,
            )

    @_synchronized
    def stage_version(self, document, chunks, version):
        self._refresh()  # writer lock is owned by the catalog
        self.add(chunks, document)
        self.persist()  # persist rows before SQLite publication

    @_synchronized
    def count_version(self, doc_id, version):
        return sum(
            d.get("doc_id") == doc_id and (d.get("version") or 0) == version
            for d in self.chunks.values()
        )

    @_synchronized
    def remove_version(self, doc_id, version):
        self._refresh()
        for key in [
            key
            for key, d in self.chunks.items()
            if d.get("doc_id") == doc_id and (d.get("version") or 0) == version
        ]:
            self._remove_chunk(key)
        self.persist()

    def identity(self):
        return {
            "backend": "keyword",
            "schema": "keyword-v2",
            "path": str(self.path) if self.path else None,
            "tokenizer": "word-cjk-bigram-v1",
        }

    @_synchronized
    def verify_version(self, doc_id, version, expected):
        stored = self.load(self.path).chunks if self.path else self.chunks
        actual = {
            key: d
            for key, d in stored.items()
            if d.get("doc_id") == doc_id and d.get("version") == version
        }
        return actual == {d["id"]: d for d in expected}

    @_synchronized
    def _visible(self, snapshot=None):
        active = (
            snapshot
            if snapshot is not None
            else (self.version_catalog.active() if self.version_catalog else None)
        )
        if active is None:
            if any(d.get("version") is not None for d in self.chunks.values()):
                raise ValueError("CATALOG_REQUIRED: versioned rows cannot be read without catalog")
            return self.chunks
        return {
            key: d
            for key, d in self.chunks.items()
            if active.get(d.get("doc_id"))
            == (d.get("version") if d.get("version") is not None else 0)
        }

    # ---------------------------------------------------------------- 查询

    @_synchronized
    def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        doc_ids: list[str] | None = None,
        filters: dict | None = None,
        snapshot=None,
    ) -> list[dict]:
        """返回按 BM25 分数降序的命中；没有命中返回空列表。"""
        query_tokens = tokenize(query)
        if not query_tokens or not self.chunks:
            return []

        allowed = set(doc_ids) if doc_ids else None
        visible = self._visible(snapshot)
        if not visible:
            return []
        filters = filters or {}
        total = len(visible)
        avgdl = sum(self.doc_len[k] for k in visible) / total
        scores: dict[str, float] = defaultdict(float)

        for token in set(query_tokens):
            posting = self.postings.get(token)
            if not posting:
                continue
            df = sum(k in visible for k in posting)
            idf = math.log(1 + (total - df + 0.5) / (df + 0.5))
            for chunk_id, frequency in posting.items():
                if chunk_id not in visible:
                    continue
                data = visible[chunk_id]
                if any(data.get(k) != v for k, v in filters.items()):
                    continue
                if allowed is not None and data.get("doc_id") not in allowed:
                    continue
                length = self.doc_len.get(chunk_id, 0)
                denominator = frequency + self.k1 * (1 - self.b + self.b * length / avgdl)
                scores[chunk_id] += idf * frequency * (self.k1 + 1) / denominator

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        hits = []
        for chunk_id, score in ranked[:top_k]:
            data = self.chunks[chunk_id]
            hits.append(
                {
                    "chunk_id": chunk_id,
                    "score": round(score, 4),
                    "doc_id": data.get("doc_id"),
                    "source": data.get("source", ""),
                    "text": data.get("text", ""),
                    "page": data.get("page", 0),
                    "section_path": data.get("section_path", ""),
                    "content_type": data.get("content_type", "text"),
                    **{
                        key: data.get(key, "")
                        for key in ["version", "page_end", "bbox", "block_ids"]
                    },
                    "ocr_mode": data.get("ocr_mode", data.get("metadata", {}).get("ocr_mode", "")),
                    "warnings": data.get("warnings", data.get("metadata", {}).get("warnings", [])),
                    "origin": data.get(
                        "origin", data.get("metadata", {}).get("origin", "connector")
                    ),
                    "reading_order": data.get(
                        "reading_order", data.get("metadata", {}).get("reading_order", 0)
                    ),
                    "needs_review": data.get(
                        "needs_review", data.get("metadata", {}).get("needs_review", False)
                    ),
                }
            )
        return hits

    # ---------------------------------------------------------------- 持久化

    @_synchronized
    def to_dict(self) -> dict:
        return {
            "k1": self.k1,
            "b": self.b,
            "chunks": self.chunks,
            "postings": self.postings,
            "doc_len": self.doc_len,
        }

    @_synchronized
    def save(self, path: str | Path) -> Path:
        out = Path(path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_dict(), ensure_ascii=False).encode("utf-8")
        descriptor, temporary = tempfile.mkstemp(
            prefix=out.name + ".", suffix=".tmp", dir=out.parent
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            if Path(temporary).read_bytes() != payload:
                raise OSError("persisted JSON verification failed")
            os.replace(temporary, out)
            directory = os.open(out.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            Path(temporary).unlink(missing_ok=True)
        self.path = out
        return out

    @_synchronized
    def persist(self) -> None:
        if self.path is not None:
            self.save(self.path)

    def close(self) -> None:
        return None

    @classmethod
    def load(cls, path: str | Path) -> "KeywordIndex":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        index = cls(k1=data.get("k1", 1.2), b=data.get("b", 0.75), path=path)
        index.chunks = dict(data.get("chunks") or {})
        index.postings = {
            token: dict(posting) for token, posting in (data.get("postings") or {}).items()
        }
        index.doc_len = {key: int(value) for key, value in (data.get("doc_len") or {}).items()}
        return index

    @_synchronized
    def list_docs(self) -> list[dict]:
        """按 doc_id 汇总已索引文档。"""
        grouped: dict[str, dict] = {}
        for data in self._visible().values():
            doc_id = data.get("doc_id", "")
            entry = grouped.setdefault(
                doc_id,
                {
                    "doc_id": doc_id,
                    "source": data.get("source", ""),
                    "chunks": 0,
                    "pages": 0,
                },
            )
            entry["chunks"] += 1
            entry["pages"] = max(entry["pages"], int(data.get("page", 0) or 0))
        return sorted(grouped.values(), key=lambda item: item["source"])

    @_synchronized
    def stats(self) -> dict:
        return {
            "backend": self.name,
            "chunks": sum(d["chunks"] for d in self.list_docs()),
            "tokens": len({t for d in self._visible().values() for t in tokenize(d["text"])}),
            "docs": len(self.list_docs()),
        }


__all__ = ["KeywordIndex", "tokenize"]
