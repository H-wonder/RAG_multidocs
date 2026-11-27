import hashlib
import json

import pytest

from rag_multidocs.execution import request_scope
from rag_multidocs.ingestion.catalog import model_identity
from rag_multidocs.providers.embedding import HashingEmbedder
from rag_multidocs.qa.grounding import GroundedQA
from rag_multidocs.qa.llm import OpenAICompatibleChat


def test_provider_retries_respect_remaining_budget_and_are_counted(monkeypatch):
    import urllib.request
    import time

    attempts = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def read(self):
            return json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {"status": "insufficient_evidence", "claims": []}
                                )
                            }
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                }
            ).encode()

    def post(request, timeout):
        attempts.append(timeout)
        if len(attempts) == 1:
            raise OSError("temporary")
        return Response()

    monkeypatch.setattr(urllib.request, "urlopen", post)
    monkeypatch.setattr(time, "sleep", lambda _: None)
    c = OpenAICompatibleChat("http://localhost:1/v1", "mock", timeout=60, retries=2)
    with request_scope(5) as budget:
        result = GroundedQA(c).generate("q", [{"chunk_id": "e", "text": "known fact"}])
        assert result["status"] == "insufficient_evidence"
        assert (
            budget.counters["transport_attempts"] == 2 and budget.counters["transport_retries"] == 1
        )
        assert (
            budget.counters["generation_calls"] == 1
            and budget.counters["reported_prompt_tokens"] == 10
        )
    assert len(attempts) == 2 and all(0 < t <= 5 for t in attempts)
    assert "response_format" not in json.loads(c._payload([], False))
    c.response_format = {"type": "json_object"}
    assert json.loads(c._payload([], False))["response_format"] == {"type": "json_object"}


def test_legacy_collection_explicit_readonly_rejects_all_writes(monkeypatch):
    from rag_multidocs.index import milvus_store
    from rag_multidocs.index.milvus import MilvusIndex

    monkeypatch.setattr(milvus_store, "_IMPORT_ERROR", None)

    class Client:
        def has_collection(self, name):
            return True

        def describe_collection(self, name):
            return {
                "description": "old-unverified",
                "fields": [{"name": "dense", "params": {"dim": 8}}],
            }

    i = MilvusIndex(HashingEmbedder(8), legacy_readonly=True)
    monkeypatch.setattr(i.store, "client", lambda: Client())
    monkeypatch.setattr(
        i.store, "ensure_database", lambda: (_ for _ in ()).throw(AssertionError("no create DB"))
    )
    i.ensure_ready()
    for operation in [
        lambda: i.remove_version("x", 1),
        lambda: i.remove_doc("x"),
        lambda: i.drop(),
        lambda: i.stage_version(None, [], 1),
    ]:
        with pytest.raises(ValueError, match="READ_ONLY"):
            operation()


def test_same_dim_model_and_missing_schema_are_rejected(monkeypatch):
    from rag_multidocs.index import milvus_store
    from rag_multidocs.index.milvus import MilvusIndex

    monkeypatch.setattr(milvus_store, "_IMPORT_ERROR", None)

    class Client:
        description = "another model"

        def has_collection(self, name):
            return True

        def describe_collection(self, name):
            return {
                "description": self.description,
                "fields": [{"name": "dense", "params": {"dim": 8}}],
            }

    client = Client()
    i = MilvusIndex(HashingEmbedder(8))
    monkeypatch.setattr(i.store, "client", lambda: client)
    with pytest.raises(ValueError, match="MODEL_MISMATCH"):
        i.store.ensure_collection()
    client.description = (
        "rag-pure-milvus-v1:"
        + hashlib.sha256(
            json.dumps(model_identity(i.embedder), sort_keys=True).encode()
        ).hexdigest()
    )
    with pytest.raises(ValueError, match="SCHEMA_MISMATCH"):
        i.store.ensure_collection()


def test_embedding_window_rejects_silent_truncation():
    from rag_multidocs.providers.embedding import SentenceTransformerEmbedder

    class Tokenizer:
        def encode(self, text):
            return list(text)

    class Model:
        tokenizer = Tokenizer()
        max_seq_length = 8

        def encode(self, *a, **kw):
            raise AssertionError("must not encode oversized input")

    embedder = SentenceTransformerEmbedder(query_instruction="")
    embedder._model = Model()
    with pytest.raises(ValueError, match="EMBEDDING_INPUT_TOO_LONG"):
        embedder.embed(["x" * 9])
