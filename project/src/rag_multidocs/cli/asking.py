"""Shared presentation of validated Core answers for CLI and REPL."""

from __future__ import annotations

import json
import sys
from dataclasses import replace

from ..retrieval.profiles import recommended_policies


def configured_policy(settings, *, top_k=None):
    policies = recommended_policies()
    if settings.rag_policy not in policies:
        raise ValueError(f"Unknown RAG policy. Choose from {', '.join(policies)}")
    if top_k is not None and top_k < 1:
        raise ValueError("--top-k must be positive")
    policy = policies[settings.rag_policy]
    if top_k is not None:
        policy = replace(policy, context_k=min(top_k, 6))
    return policy


def retrieval_only(result):
    return {
        **result,
        "status": "retrieval_failed" if result.get("retrieval_error") else "retrieval_only",
        "answer": "",
        "citations": [],
        "evidence": result["hits"],
        "used_llm": False,
    }


def _doc_ids(rows):
    return sorted({row["doc_id"] for row in rows if row.get("doc_id")})


def trace_summary(result, profile):
    rounds = result.get("retrieval_trace", [])
    constraint = result.get("source_constraints", {})
    required = set(constraint.get("doc_ids") or ())
    evidence = result.get("evidence", [])
    generation = result.get("generation_evidence", [])
    citations = result.get("citations", [])
    useful = _doc_ids(result.get("answer_evidence") or generation)
    attempts = result.get("attempts", [])
    checks = set()
    for attempt in attempts:
        for failure in attempt.get("validation_errors", []):
            code = failure.rsplit(": ", 1)[-1]
            if code.startswith(("L1_", "L2_", "L3_")):
                checks.add(code)
    return {
        "policy": profile,
        "status": result["status"],
        "answer_mode": result.get("answer_mode") or (
            "model_generation" if result.get("used_llm") else "no_generation"
        ),
        "index_calls": result.get("index_calls", 0),
        "candidate_counts": [row["candidate_count"] for row in rounds],
        "context_count": len(evidence),
        "generation_evidence_count": len(generation),
        "source_constraint": {
            "explicit": bool(constraint.get("explicit")),
            "restricted": constraint.get("doc_ids") is not None,
            "blocked": bool(constraint.get("blocked")),
            "resolved_doc_ids": sorted(required),
            "candidate_doc_ids": _doc_ids(rounds[-1].get("candidates", [])) if rounds else [],
            "context_doc_ids": _doc_ids(evidence),
            "generation_doc_ids": _doc_ids(generation),
            "cited_doc_ids": _doc_ids(citations),
            "uncovered_doc_ids": sorted(required.difference(_doc_ids(evidence))),
            "unconsulted_doc_ids": sorted(required.difference(useful)),
        },
        "secondary_retrieval": len(rounds) > 1,
        "coverage": [{
            "missing_needs": len(row.get("missing_evidence", [])),
            "finite_relations": {stage: len(items) for stage, items in
                                 row.get("finite_relation_coverage", {}).items()},
        } for row in rounds],
        "reranker": [{
            "name": row.get("resources", {}).get("reranker"),
            "mode": row.get("reranker_execution", {}).get("mode"),
            "seconds": round(row.get("rerank_seconds", 0), 3),
        } for row in rounds],
        "validation": {
            "accepted_claims": len(result.get("claims", [])),
            "citations": len(citations),
            "rejected_attempt_codes": sorted(checks),
            "fact_verified": False,
        },
        "warnings": sorted({warning["code"] for warning in result.get("warnings", [])}),
        "timing_s": {
            name: round(result.get(f"{name}_seconds", 0), 3)
            for name in ("retrieval", "generation", "total")
        },
        "usage": result.get("usage", {}),
        "token_budget": {
            **result.get("token_budget", {}),
            "retrieval_context_estimate": result.get("context_tokens", 0),
            "generation_context_estimate": result.get("generation_context_tokens_estimate", 0),
            "prompt_estimate": result.get("prompt_tokens_estimate", 0),
        },
    }


def print_trace(trace, *, file=None):
    print(json.dumps(trace, ensure_ascii=False, indent=2), file=file)


def print_answer(result, print_hits):
    print(f"状态 {result['status']}")
    if result["answer"]:
        print(result["answer"])
    for citation in result["citations"]:
        location = citation.get("location") or {}
        worksheet = f" 工作表 {location['worksheet']}" if location.get("worksheet") else ""
        section = f" §{citation['section_path']}" if citation.get("section_path") else ""
        page = f" p{citation['page']}" if citation.get("page") else ""
        print(f"[{citation['index']}] {citation['source']}{page}{section}{worksheet}")
        for quote in citation.get("excerpts", []):
            print(f"    {quote[:240]}")
    if not result["citations"] and result["evidence"]:
        print("检索候选")
        print_hits(result["evidence"])
    for warning in result.get("warnings", []):
        print(f"[{warning['code']}] {warning.get('message', '')}", file=sys.stderr)
