"""统一文档模型：接入层与检索层之间唯一的公共数据结构。

设计来源：
- 原项目 ``任务二/vendor/docparse/docparse/ir.py`` 的 Block/Page/Document 思路；
- 课程项目只需要“文档标识 + 正文 + 来源 + 可追溯的片段”，因此裁掉 OCR 专属字段，
  保留 page / bbox / section_path 这类可追溯信息。

约定：
1. Connector 只负责把一种数据源变成 ``Document``，不负责切块和检索。
2. ``Document.doc_id`` 由来源路径稳定派生；内容变化只改 metadata["sha256"]，
   doc_id 不变，这样增量更新不会产生重复文档。
3. ``Chunk`` 是检索和引用的最小单位，``source``、``page``、``section_path``
   必须能从命中结果一路回显到原文。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any


def stable_doc_id(source: str, length: int = 12) -> str:
    """从逻辑来源（相对路径或 URI）派生稳定 id。

    注意：不要用绝对路径。绝对路径换机器就变，doc_id 跟着变，增量索引会失效。
    """
    normalized = source.replace("\\", "/").strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:length]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class Document:
    """一篇已接入的文档。

    ``content`` 是归一化后的正文；版面结构信息放在 ``segments`` / metadata 里，
    由切块器决定怎么用，接入层不预先切块。
    """

    doc_id: str
    source: str
    content: str
    title: str = ""
    media_type: str = "text/plain"
    metadata: dict[str, Any] = field(default_factory=dict)
    segments: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_text(
        cls,
        content: str,
        source: str,
        *,
        title: str = "",
        media_type: str = "text/plain",
        metadata: dict[str, Any] | None = None,
    ) -> "Document":
        return cls(
            doc_id=stable_doc_id(source),
            source=source,
            content=content,
            title=title,
            media_type=media_type,
            metadata={**(metadata or {}), "sha256": sha256_text(content)},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "source": self.source,
            "content": self.content,
            "title": self.title,
            "media_type": self.media_type,
            "metadata": dict(self.metadata),
            "segments": list(self.segments),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Document":
        return cls(
            doc_id=str(data["doc_id"]),
            source=str(data.get("source", "")),
            content=str(data.get("content", "")),
            title=str(data.get("title", "")),
            media_type=str(data.get("media_type", "text/plain")),
            metadata=dict(data.get("metadata") or {}),
            segments=list(data.get("segments") or []),
        )


@dataclass
class Chunk:
    """可检索、可引用的文本块。

    ``id`` 由 (doc_id, content_type, index) 决定，重复导入同一文档会得到同一批 id，
    便于索引层做“先删后插”的幂等更新。
    """

    id: str
    doc_id: str
    text: str
    source: str
    index: int
    content_type: str = "text"
    section_path: str = ""
    page: int = 0
    page_end: int = 0
    bbox: str = ""
    block_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "doc_id": self.doc_id,
            "text": self.text,
            "source": self.source,
            "index": self.index,
            "content_type": self.content_type,
            "section_path": self.section_path,
            "page": self.page,
            "page_end": self.page_end,
            "bbox": self.bbox,
            "block_ids": ",".join(self.block_ids),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Chunk":
        block_ids = data.get("block_ids") or ""
        if isinstance(block_ids, str):
            block_ids = [item for item in block_ids.split(",") if item]
        return cls(
            id=str(data["id"]),
            doc_id=str(data["doc_id"]),
            text=str(data.get("text", "")),
            source=str(data.get("source", "")),
            index=int(data.get("index", 0)),
            content_type=str(data.get("content_type", "text")),
            section_path=str(data.get("section_path", "")),
            page=int(data.get("page", 0) or 0),
            page_end=int(data.get("page_end", 0) or 0),
            bbox=str(data.get("bbox", "")),
            block_ids=list(block_ids),
            metadata=dict(data.get("metadata") or {}),
        )


def chunk_id(doc_id: str, content_type: str, index: int) -> str:
    """确定性块 id。不要用随机 uuid，否则重跑会不断产生重复向量。"""
    raw = f"{doc_id}:{content_type}:{index}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


__all__ = [
    "Document",
    "Chunk",
    "stable_doc_id",
    "chunk_id",
    "sha256_text",
    "sha256_file",
]
