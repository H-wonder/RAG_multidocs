"""Programmatic acceptance harness (not a replacement CLI).

Offline: PYTHONPATH=src python tests/a_plus_acceptance.py
Live services are opt-in via RAG_A_PLUS_LIVE=1. Never downloads models.
Artifacts outside project under data/a_plus_acceptance. No commits or user collection deletion.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from rag_multidocs.connectors.registry import ConnectorRegistry
from rag_multidocs.connectors.txt import TxtConnector
from rag_multidocs.core import RAGCore
from rag_multidocs.evaluation import run_suite, write_report, ablation_policies
from rag_multidocs.index import KeywordIndex
from rag_multidocs.retrieval.controller import RetrievalPolicy

PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT.parent / "data/a_plus_acceptance"


def build_keyword():
    registry = ConnectorRegistry(PROJECT / "connectors")
    registry.register(TxtConnector())
    core = RAGCore(registry, KeywordIndex(), policy=RetrievalPolicy(reranker="lightweight"))
    corpus = PROJECT / "samples/campus-corpus"
    for name, path in [
        ("txt", corpus / "txt"),
        ("connector.markdown", corpus / "markdown"),
        ("connector.html", corpus / "html"),
        ("connector.jsonl", PROJECT / "samples/eval/normalized.jsonl"),
    ]:
        report = core.ingest(name, str(path))
        if not report.ok:
            raise AssertionError(report.as_dict())
    return core


def offline():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    # Verify reference fixtures remain tied to the actual source bytes.
    for line in (PROJECT / "samples/eval/normalized.jsonl").read_text().splitlines():
        record = json.loads(line)
        assert (
            hashlib.sha256((PROJECT / record["provenance"]).read_bytes()).hexdigest()
            == record["source_sha256"]
        )
    cases = json.loads((PROJECT / "samples/eval/questions.json").read_text())["cases"]
    core = build_keyword()
    try:
        report = run_suite(core, cases)
        write_report(report, OUTPUT / "offline_retrieval")
        ablations = {}
        for name, policy in ablation_policies().items():
            if name in {"crossencoder", "l3_off"}:
                ablations[name] = {
                    "status": "not_run_offline",
                    "reason": "crossencoder requires opt-in cached model; L3 ablation requires generated answers",
                }
                continue
            core.controller.policy = policy
            r = run_suite(core, cases)
            write_report(r, OUTPUT / f"ablation_{name}")
            ablations[name] = r["summary"]
        (OUTPUT / "ablation_summary.json").write_text(
            json.dumps(ablations, ensure_ascii=False, indent=2)
        )
        print(json.dumps(report["summary"], ensure_ascii=False, indent=2), flush=True)
    finally:
        core.close()


if __name__ == "__main__":
    offline()
