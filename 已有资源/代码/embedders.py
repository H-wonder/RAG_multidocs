"""嵌入后端：把文本变成向量，供 Milvus 索引/检索使用。

来源：``任务二/docflow/services/rag_service.py`` 的 local / dashscope 两套实现。
这里抽成可替换的 ``Embedder`` 协议，核心索引代码只依赖协议，不依赖具体模型。

三种实现：
- ``SentenceTransformerEmbedder``：本地 bge-small-zh-v1.5（512 维，CPU 可跑）；
- ``OpenAICompatibleEmbedder``：阿里 DashScope text-embedding-v4 等兼容接口；
- ``HashingEmbedder``：纯标准库的确定性向量，仅用于离线演示/测试，不具备语义能力。

课程项目如果只做关键词检索，不需要本文件；选了 Milvus 方案再引入。
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
import urllib.error
import urllib.request
from typing import Protocol, runtime_checkable

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[一-鿿]")


@runtime_checkable
class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str], query: bool = False) -> list[list[float]]:
        ...


class SentenceTransformerEmbedder:
    """本地 sentence-transformers 模型。

    bge 系列建议给查询加指令前缀，文档不加；实测能提升中文短查询的召回。
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-zh-v1.5",
        dim: int = 512,
        device: str = "cpu",
        query_instruction: str = "为这个句子生成表示以用于检索相关文章：",
    ):
        self.model_name = model_name
        self.dim = dim
        self.device = device
        self.query_instruction = query_instruction
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def embed(self, texts: list[str], query: bool = False) -> list[list[float]]:
        if not texts:
            return []
        if query and self.query_instruction:
            texts = [self.query_instruction + text for text in texts]
        vectors = self._load().encode(
            texts, normalize_embeddings=True, batch_size=32
        )
        return [vector.tolist() for vector in vectors]


class OpenAICompatibleEmbedder:
    """OpenAI 兼容的 embeddings 接口（DashScope compatible-mode 等）。

    阿里官方单次最多 10 条，所以按 batch_size=10 分批；网络抖动重试 3 次。
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        dim: int = 1024,
        batch_size: int = 10,
        timeout: int = 60,
        retries: int = 3,
    ):
        if not api_key:
            raise ValueError("api_key 为空：请通过环境变量注入，不要写进仓库")
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.dim = dim
        self.batch_size = batch_size
        self.timeout = timeout
        self.retries = retries

    def _post(self, batch: list[str]) -> list[list[float]]:
        payload = json.dumps(
            {
                "model": self.model,
                "input": batch,
                "dimensions": self.dim,
                "encoding_format": "float",
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            self.base_url,
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
                rows = sorted(data["data"], key=lambda row: row.get("index", 0))
                return [row["embedding"] for row in rows]
            except Exception as exc:  # 网络/限流/5xx 都按可重试处理
                last_error = exc
                if attempt + 1 < self.retries:
                    time.sleep(1 + attempt)
        raise RuntimeError(f"嵌入请求失败: {last_error}") from last_error

    def embed(self, texts: list[str], query: bool = False) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            vectors.extend(self._post(texts[start : start + self.batch_size]))
        return vectors


class HashingEmbedder:
    """确定性哈希向量，仅用于离线演示与单元测试。

    它没有语义能力，但能让“接入→索引→检索”链路不依赖模型下载或 API Key。
    不要把它的检索效果当成产品效果。
    """

    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed(self, texts: list[str], query: bool = False) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self.dim
            for token in _TOKEN_RE.findall(text.lower()):
                digest = hashlib.md5(token.encode("utf-8")).digest()
                bucket = int.from_bytes(digest[:4], "big") % self.dim
                sign = 1.0 if digest[4] & 1 else -1.0
                vector[bucket] += sign
            norm = math.sqrt(sum(value * value for value in vector)) or 1.0
            vectors.append([value / norm for value in vector])
        return vectors


__all__ = [
    "Embedder",
    "SentenceTransformerEmbedder",
    "OpenAICompatibleEmbedder",
    "HashingEmbedder",
]
