from dataclasses import replace
from pathlib import Path

import pytest

from rag_multidocs.config import Settings
from rag_multidocs.connectors.registry import ConnectorRegistry
from rag_multidocs.connectors.txt import TxtConnector
from rag_multidocs.core import RAGCore
from rag_multidocs.evaluation.metrics import run_suite
from rag_multidocs.ingestion.service import IngestService
from rag_multidocs.retrieval.controller import RetrievalController, RetrievalPolicy
from rag_multidocs.retrieval.search import SearchService
from rag_multidocs.services import create_core


def test_old_base_backend_without_version_or_filters_is_compatible(tmp_path):
    class BaseIndex:
        def __init__(self):
            self.rows = []

        def remove_doc(self, doc_id):
            self.rows = [r for r in self.rows if r.doc_id != doc_id]

        def add(self, chunks, document=None):
            self.rows.extend(chunks)
            return len(chunks)

        def persist(self):
            pass

        def search(self, query, top_k=5, doc_ids=None):
            return [
                dict(chunk_id=r.id, text=r.text, source=r.source, doc_id=r.doc_id)
                for r in self.rows
                if query in r.text
            ][:top_k]

        def list_docs(self):
            return []

        def stats(self):
            return {}

        def close(self):
            pass

    registry = ConnectorRegistry()
    registry.register(TxtConnector())
    index = BaseIndex()
    service = IngestService(registry, index)
    p = tmp_path / "a.txt"
    p.write_text("legacy compatible evidence")
    report = service.ingest("txt", str(p))
    assert report.ok and service.catalog is None
    assert report.documents[0]["warnings"][0]["code"] == "LEGACY_INDEX_NON_ATOMIC"
    assert SearchService(index).search("evidence")[0]["source"] == "a.txt"
    core = RAGCore(registry, index)
    assert core.search("evidence")["hits"]
    with pytest.raises(ValueError, match="CAPABILITY"):
        core.cleanup()
    with pytest.raises(ValueError, match="CAPABILITY"):
        SearchService(index).search("evidence", filters={"page": 1})


def test_factory_explicit_paths_survive_cwd_changes(tmp_path, monkeypatch):
    project = Path(__file__).resolve().parents[1]
    settings = Settings(
        index_backend="keyword",
        data_dir=str(tmp_path / "kb"),
        connectors_dir=str(project / "connectors"),
    )
    core = create_core(settings, base_dir=project)
    p = tmp_path / "a.txt"
    p.write_text("stable directory evidence")
    assert core.ingest("txt", str(p)).ok
    core.close()
    monkeypatch.chdir(tmp_path)
    again = create_core(settings, base_dir=project)
    assert again.search("evidence")["hits"][0]["source"] == "a.txt"
    again.close()


def test_baseline_keeps_candidates_without_treating_rrf_as_absolute_support():
    class Index:
        def __init__(self):
            self.limits = []

        def search(self, q, **kw):
            self.limits.append(kw)
            return [
                {"chunk_id": str(i), "text": "completely unrelated", "source": "x"}
                for i in range(100)
            ]

    index = Index()
    result = RetrievalController(index, policy=RetrievalPolicy.baseline()).retrieve("borrow limit")
    # Backend candidates are not semantic proof, but zero lexical overlap is
    # not proof of irrelevance either. The baseline retains original retrieval
    # semantics; QA still has to answer from evidence or decline.
    assert len(result['hits']) == 4
    assert all(h['support_score'] == 0 for h in result['hits'])
    assert result['evidence_support'] == 'heuristic_only'
    assert all(p["top_k"] <= 48 and p["ann_limit"] <= 48 * 4 for p in index.limits)
    assert result["retrieval_trace"][0]["candidate_count"] <= 48


def test_hand_calculable_stage_metrics_gold_rescue_and_null_reviews():
    good = {"chunk_id": "good", "source": "b", "page": 1, "text": "gold fact"}
    other = {"chunk_id": "other", "source": "a", "page": 1, "text": "other fact"}
    case = {
        "id": "q",
        "question": "q",
        "answerable": True,
        "expected_evidence": [{"source": "b", "page": 1, "contains": "gold fact"}],
        "required_subquestions": ["need"],
        "subquestion_evidence": {"need": [{"source": "b", "page": 1, "contains": "gold fact"}]},
    }

    class Core:
        def search(self, q):
            return {
                "hits": [good],
                "retrieval_trace": [
                    {"context_ids": ["other"], "candidates": [other]},
                    {
                        "base_candidates": [other, good],
                        "reranked_candidates": [good, other],
                        "context_ids": ["good"],
                    },
                ],
                "context_tokens": 20,
                "retrieval_seconds": 0.5,
                "rewrite_seconds": 0.1,
                "usage": {"embedding_calls": 2, "rewrite_calls": 1, "generation_calls": 0},
            }

    result = run_suite(Core(), [case], k=1)
    assert result["summary"]["stages"]["base"]["hit_at_k"] == 0
    assert result["summary"]["stages"]["reranked"]["hit_at_k"] == 1
    assert result["summary"]["subquestion_evidence_coverage"] == 1
    assert result["summary"]["gold_recall_rescue_rate"] == 1
    assert result["summary"]["answer_accuracy"] is None
    assert result["summary"]["citation_precision"] is None
    assert result["summary"]["retrieval_seconds_mean"] == 0.5
    assert result["summary"]["usage"]["rewrite_calls"] == 1


def test_none_stage_and_full_rrf_baseline_are_distinct():
    baseline = RetrievalPolicy.baseline()
    only_none = replace(RetrievalPolicy(), reranker="none")
    assert not baseline.use_mmr and not baseline.use_recall and not baseline.use_rewrite
    assert only_none.use_mmr and only_none.use_recall
