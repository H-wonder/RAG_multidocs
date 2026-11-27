"""Opt-in real integration through public API, isolated collection and directory."""

from __future__ import annotations

import json
import tempfile
import time
import uuid
from pathlib import Path

from rag_multidocs.connectors.registry import ConnectorRegistry
from rag_multidocs.connectors.txt import TxtConnector
from rag_multidocs.core import RAGCore
from rag_multidocs.index.milvus import MilvusIndex
from rag_multidocs.ingestion.catalog import DocumentCatalog
from rag_multidocs.providers.embedding import HashingEmbedder, SentenceTransformerEmbedder
from rag_multidocs.qa.llm import OpenAICompatibleChat
from rag_multidocs.rerankers import RerankerRegistry
from rag_multidocs.retrieval.controller import RetrievalPolicy

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT.parent / "data/a_plus_acceptance"


def live():
    OUT.mkdir(parents=True, exist_ok=True)
    collection = "a_plus_" + uuid.uuid4().hex[:12]
    result = {
        "collection": collection,
        "status": "failed",
        "scope": "real Milvus/BGE/Ollama on isolated collection; not general accuracy validation",
    }
    index = None
    try:
        index = MilvusIndex(
            SentenceTransformerEmbedder(local_files_only=True),
            uri="http://127.0.0.1:19530",
            db_name="a_plus_validation",
            collection=collection,
        )
        for _ in range(30):
            try:
                index.ensure_ready()
                break
            except Exception:
                time.sleep(1)
        else:
            raise RuntimeError("Milvus startup or schema validation failed")
        registry = ConnectorRegistry(
            PROJECT / "connectors",
            context={
                "settings": {
                    "docparse_repo": str(PROJECT.parent / "已有资源/可选组件/docparse"),
                    "docparse_python": "/home/genevievelin/anaconda3/envs/docparse/bin/python",
                    "docparse_mode": "auto",
                }
            },
        )
        registry.register(TxtConnector())
        DocumentCatalog(index, OUT / f"{collection}.sqlite")
        core = RAGCore(
            registry,
            index,
            rerankers=RerankerRegistry(PROJECT / "rerankers", timeout_s=60),
            policy=RetrievalPolicy(reranker="lightweight"),
            chat=OpenAICompatibleChat(
                "http://127.0.0.1:11434/v1",
                "gemma3:4b",
                temperature=0,
                max_tokens=512,
                retries=1,
                timeout=90,
            ),
        )
        with tempfile.TemporaryDirectory(prefix="rag_a_plus_") as tmp:
            p = Path(tmp) / "knowledge.txt"
            p.write_text("图书馆本科生借阅期限30天。图书馆研究生借阅期限60天。", encoding="utf8")
            first = core.ingest("txt", str(p))
            assert first.ok, first.as_dict()
            repeat = core.ingest("txt", str(p))
            assert repeat.ok and repeat.chunks == 0
            result["repeat_added_chunks"] = repeat.chunks
            p.write_text("图书馆本科生借阅期限45天。图书馆研究生借阅期限60天。", encoding="utf8")
            changed = core.ingest("txt", str(p))
            assert changed.ok
            hits = index.search("本科生", filters={"content_type": "text"})
            assert hits and all("45天" in h["text"] and h["version"] == 2 for h in hits)
            result["version_update"] = "passed"

            # Same dimension, different embedding model/config must not reuse collection.
            class DifferentModel(HashingEmbedder):
                model_name = "different_semantic_space"

            other = MilvusIndex(
                DifferentModel(dim=512),
                uri="http://127.0.0.1:19530",
                db_name="a_plus_validation",
                collection=collection,
            )
            try:
                try:
                    other.ensure_ready()
                except ValueError:
                    result["same_dimension_model_isolation"] = "passed"
                else:
                    raise AssertionError("model identity mismatch not rejected")
            finally:
                other.close()
            result["answer"] = core.ask("本科生借阅期限是多少天？", style="concise")
            result["crossencoder"] = RerankerRegistry(PROJECT / "rerankers", timeout_s=60).rerank(
                "crossencoder", "本科生借阅期限", hits
            )[1]
            # Real PDF and OCR through the existing independent PDF connector.
            for name in ["course-project-guide.pdf", "scanned-library-notice.png"]:
                report = core.ingest("pdf", str(PROJECT / "samples/campus-corpus/pdf" / name))
                result[name] = report.as_dict()
            result["status"] = (
                "passed"
                if result["answer"]["status"] in {"answered", "partial"}
                and all(
                    result[n]["documents"]
                    for n in ["course-project-guide.pdf", "scanned-library-notice.png"]
                )
                else "partial"
            )
        core.close()
        index = None
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {str(exc)[:250]}"
    finally:
        if index:
            index.close()
        # Keep the isolated collection for review; never drop a user's collection.
        (OUT / "live_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(
            json.dumps(
                {k: v for k, v in result.items() if k not in ["answer"]},
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )


if __name__ == "__main__":
    live()
