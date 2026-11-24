"""Index backends.

``MilvusIndex`` is imported from ``rag_multidocs.index.milvus`` only when the
Milvus backend is selected, so installations without pymilvus can still use the
keyword index.
"""

from .keyword import KeywordIndex

__all__ = ["KeywordIndex"]
