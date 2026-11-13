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
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from .model import Chunk

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


class KeywordIndex:
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: dict[str, dict] = {}
        self.postings: dict[str, dict[str, int]] = {}
        self.doc_len: dict[str, int] = {}

    # ---------------------------------------------------------------- 写入

    @staticmethod
    def _as_dict(chunk: Chunk | dict) -> dict:
        if isinstance(chunk, Chunk):
            return chunk.as_dict()
        return dict(chunk)

    def add(self, chunks: Iterable[Chunk | dict]) -> int:
        """加入一批块；同 id 的块会覆盖旧内容。返回实际加入数量。"""
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
            searchable = f"{data.get('section_path', '')}\n{data['text']}"
            tokens = tokenize(searchable)
            counts = Counter(tokens)
            self.chunks[chunk_id] = data
            self.doc_len[chunk_id] = len(tokens)
            for token, frequency in counts.items():
                self.postings.setdefault(token, {})[chunk_id] = frequency
            added += 1
        return added

    def _remove_chunk(self, chunk_id: str) -> None:
        self.chunks.pop(chunk_id, None)
        self.doc_len.pop(chunk_id, None)
        for token in list(self.postings):
            self.postings[token].pop(chunk_id, None)
            if not self.postings[token]:
                del self.postings[token]

    def remove_doc(self, doc_id: str) -> int:
        """删除一篇文档的所有块。返回删除数量。"""
        ids = [
            chunk_id
            for chunk_id, data in self.chunks.items()
            if data.get("doc_id") == doc_id
        ]
        for chunk_id in ids:
            self._remove_chunk(chunk_id)
        return len(ids)

    def clear(self) -> None:
        self.chunks.clear()
        self.postings.clear()
        self.doc_len.clear()

    # ---------------------------------------------------------------- 查询

    def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        doc_ids: list[str] | None = None,
    ) -> list[dict]:
        """返回按 BM25 分数降序的命中；没有命中返回空列表。"""
        query_tokens = tokenize(query)
        if not query_tokens or not self.chunks:
            return []

        allowed = set(doc_ids) if doc_ids else None
        total = len(self.chunks)
        avgdl = sum(self.doc_len.values()) / total
        scores: dict[str, float] = defaultdict(float)

        for token in set(query_tokens):
            posting = self.postings.get(token)
            if not posting:
                continue
            df = len(posting)
            idf = math.log(1 + (total - df + 0.5) / (df + 0.5))
            for chunk_id, frequency in posting.items():
                data = self.chunks[chunk_id]
                if allowed is not None and data.get("doc_id") not in allowed:
                    continue
                length = self.doc_len.get(chunk_id, 0)
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * length / avgdl
                )
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
                }
            )
        return hits

    # ---------------------------------------------------------------- 持久化

    def to_dict(self) -> dict:
        return {
            "k1": self.k1,
            "b": self.b,
            "chunks": self.chunks,
            "postings": self.postings,
            "doc_len": self.doc_len,
        }

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False), encoding="utf-8"
        )
        return out

    @classmethod
    def load(cls, path: str | Path) -> "KeywordIndex":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        index = cls(k1=data.get("k1", 1.2), b=data.get("b", 0.75))
        index.chunks = dict(data.get("chunks") or {})
        index.postings = {
            token: dict(posting)
            for token, posting in (data.get("postings") or {}).items()
        }
        index.doc_len = {key: int(value) for key, value in (data.get("doc_len") or {}).items()}
        return index

    def list_docs(self) -> list[dict]:
        """按 doc_id 汇总已索引文档。"""
        grouped: dict[str, dict] = {}
        for data in self.chunks.values():
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

    def stats(self) -> dict:
        return {
            "chunks": len(self.chunks),
            "tokens": len(self.postings),
            "docs": len({data.get("doc_id") for data in self.chunks.values()}),
        }


__all__ = ["KeywordIndex", "tokenize"]
