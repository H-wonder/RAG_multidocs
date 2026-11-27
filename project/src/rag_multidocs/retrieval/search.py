"""Search service over the configured index."""

from __future__ import annotations

from typing import Any

from ..index import KeywordIndex
from .controller import RetrievalController


class SearchService:
    def __init__(self, index: KeywordIndex):
        self.index = index

    def search(
        self,
        query: str,
        top_k: int = 5,
        doc_ids: list[str] | None = None,
        filters: dict | None = None,
    ) -> list[dict[str, Any]]:
        return RetrievalController.search_hits(self.index, query, top_k=top_k, doc_ids=doc_ids, filters=filters)


__all__ = ["SearchService"]
