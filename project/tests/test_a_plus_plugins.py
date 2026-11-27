import hashlib
import json
from pathlib import Path

import pytest

from rag_multidocs.connectors.registry import ConnectorRegistry
from rag_multidocs.connectors.txt import TxtConnector
from rag_multidocs.core import RAGCore
from rag_multidocs.index import KeywordIndex
from rag_multidocs.rerankers import RerankerRegistry
from rag_multidocs.retrieval.controller import RetrievalPolicy


def hashes():
    root = Path(__file__).resolve().parents[1] / "src/rag_multidocs"
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*.py")
    }


def write_plugin(root, name, body, kind="reranker", enabled=True):
    p = root / name
    p.mkdir(parents=True)
    (p / "manifest.json").write_text(
        json.dumps(
            {
                "name": name,
                "kind": kind,
                "entry": "adapter:rank" if kind == "reranker" else "adapter:load",
                "contract_version": "1",
                "enabled": enabled,
            }
        )
    )
    (p / "adapter.py").write_text(body)
    return p


def candidates():
    return [
        {"chunk_id": "a", "text": "ordinary note", "source": "a.txt", "doc_id": "a", "score": 0.02},
        {"chunk_id": "b", "text": "表格证据", "source": "b.txt", "doc_id": "b", "score": 0.01},
    ]


def test_independent_reranker_changes_order_without_core_edits(tmp_path):
    before = hashes()
    write_plugin(
        tmp_path,
        "external.rank",
        "from rag_multidocs.contracts import RankedCandidate\ndef rank(request):\n return [RankedCandidate(h.candidate_id,i+1,1-i*.1) for i,h in enumerate(reversed(request.candidates))]\n",
    )
    registry = RerankerRegistry(tmp_path)
    ranked, warnings = registry.rerank("external.rank", "q", candidates())
    assert not warnings and ranked[0]["chunk_id"] == "b"
    assert ranked[0]["text"] == "表格证据"
    assert hashes() == before
    assert registry.rerank("none", "q", candidates()) == (candidates(), [])


@pytest.mark.parametrize(
    "body,code",
    [
        ('raise RuntimeError("boom")', "RERANKER_EXECUTION_ERROR"),
        (
            'return [RankedCandidate("unknown",1,.9),RankedCandidate("b",2,.8)]',
            "RERANKER_INVALID_OUTPUT",
        ),
        ('return [RankedCandidate("a",1,.9),RankedCandidate("a",2,.8)]', "RERANKER_INVALID_OUTPUT"),
        (
            'return [RankedCandidate("a",1,float("nan")),RankedCandidate("b",2,.8)]',
            "RERANKER_INVALID_OUTPUT",
        ),
        ('return [RankedCandidate("a",1,2),RankedCandidate("b",2,.8)]', "RERANKER_INVALID_OUTPUT"),
        ('return [RankedCandidate("a",2,.9),RankedCandidate("b",1,.8)]', "RERANKER_INVALID_OUTPUT"),
        ("import nonexistent_dependency_for_a_plus", "RERANKER_EXECUTION_ERROR"),
        ("return []", "RERANKER_INVALID_OUTPUT"),
        ('return [{"candidate_id":"a","text":"tampered"}]', "RERANKER_INVALID_OUTPUT"),
    ],
)
def test_invalid_plugins_fall_back_to_exact_rrf(tmp_path, body, code):
    write_plugin(
        tmp_path,
        "broken",
        "from rag_multidocs.contracts import RankedCandidate\ndef rank(request):\n " + body + "\n",
    )
    ranked, warnings = RerankerRegistry(tmp_path).rerank("broken", "q", candidates())
    assert ranked == candidates() and warnings[0]["code"] == code


def test_disabled_missing_and_killable_timeout(tmp_path):
    write_plugin(tmp_path, "off", "def rank(request): raise AssertionError()", enabled=False)
    assert (
        RerankerRegistry(tmp_path).rerank("off", "q", candidates())[1][0]["code"]
        == "RERANKER_DISABLED"
    )
    assert (
        RerankerRegistry(tmp_path).rerank("absent", "q", candidates())[1][0]["code"]
        == "RERANKER_NOT_FOUND"
    )
    write_plugin(tmp_path, "slow", "import time\ndef rank(request): time.sleep(10)")
    registry = RerankerRegistry(tmp_path, timeout_s=0.2)
    assert registry.rerank("slow", "q", candidates())[1][0]["code"] == "RERANKER_TIMEOUT"


def test_independent_connector_full_flow_disabled_and_failure(tmp_path):
    before = hashes()
    body = """from rag_multidocs.model import Document
from rag_multidocs.contracts import ConnectorResult

def load(payload,context):
 return ConnectorResult([Document.from_text("The library opens at 08:00.","independent.record")])
"""
    write_plugin(tmp_path / "plugins", "external.connector", body, "connector")
    registry = ConnectorRegistry(tmp_path / "plugins")
    registry.register(TxtConnector())

    class Chat:
        def complete(self, messages):
            return json.dumps(
                {
                    "status": "answered",
                    "claims": [
                        {
                            "text": "The library opens at 08:00.",
                            "citation_ids": [1],
                            "supporting_quotes": ["The library opens at 08:00."],
                        }
                    ],
                }
            )

    core = RAGCore(registry, KeywordIndex(), chat=Chat(), policy=RetrievalPolicy(reranker="none"))
    assert core.ingest("external.connector", "any").ok
    result = core.ask("When does the library open?")
    assert (
        result["status"] in {"answered", "partial"}
        and result["citations"][0]["source"] == "independent.record"
    )
    assert hashes() == before
    registry.enabled = {"txt"}
    assert core.ingest("external.connector", "any").issues[0].code == "CONNECTOR_DISABLED"
    p = tmp_path / "ok.txt"
    p.write_text("Other knowledge remains available.")
    assert core.ingest("txt", str(p)).ok
    registry.enabled = None
    write_plugin(
        tmp_path / "plugins",
        "failing.connector",
        'def load(payload,context): raise RuntimeError("bad")',
        "connector",
    )
    assert core.ingest("failing.connector", "any").issues[0].code == "CONNECTOR_ERROR"
    assert core.index.search("library")
