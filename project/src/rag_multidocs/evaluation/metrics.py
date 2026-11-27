"""Stage-specific fixed-K evidence metrics; human generation review remains null."""

from __future__ import annotations

import json
import csv
from collections import Counter
import statistics
import time
from dataclasses import replace
from pathlib import Path

from ..retrieval.controller import RetrievalPolicy, similarity


def _compact(text):
    return "".join(str(text).split())


def _match(hit, gold):
    excerpt = gold.get("contains", "")
    return (
        hit.get("source") == gold["source"]
        and hit.get("page", 0) == gold.get("page", 0)
        and bool(excerpt)
        and _compact(excerpt) in _compact(hit.get("text", ""))
    )


def _mean(values):
    return statistics.mean(values) if values else None


def _p95(values):
    import math

    return sorted(values)[max(0, math.ceil(0.95 * len(values)) - 1)] if values else None


def _ranking(hits, gold, k):
    if not gold:
        return {
            name: None for name in ["hit_at_1", "hit_at_3", "hit_at_k", "recall_at_k", "mrr_at_k"]
        }
    rank = next((i for i, h in enumerate(hits[:k], 1) if any(_match(h, g) for g in gold)), None)
    return {
        "hit_at_1": any(_match(h, g) for h in hits[:1] for g in gold),
        "hit_at_3": any(_match(h, g) for h in hits[:3] for g in gold),
        "hit_at_k": rank is not None,
        "recall_at_k": sum(any(_match(h, g) for h in hits[:k]) for g in gold) / len(gold),
        "mrr_at_k": 1 / rank if rank else 0,
    }


def run_suite(
    core, cases, *, generate=False, reviews=None, k=3, run_label="development_regression", on_row=None
):
    if not 1 <= k <= 48:
        raise ValueError("fixed K must be in 1..48")
    reviews = reviews or {}
    cases = list(cases)
    rows = []
    # A suite uses a frozen catalog. Resolve source scopes once outside every
    # timed request; list_documents may perform expensive per-document RPCs.
    scoped = any(case.get('source_allowlist') for case in cases)
    documents = core.list_documents() if scoped else []
    for case in cases:
        scope = {}
        if case.get('source_allowlist'):
            allowed=set(case['source_allowlist'])
            scope['doc_ids']=[d['doc_id'] for d in documents if d['source'] in allowed]
            if not scope['doc_ids']:
                raise ValueError('Evaluation source scope has no indexed documents')
        started = time.perf_counter()
        result = core.ask(case["question"],**scope) if generate else core.search(case["question"],**scope)
        elapsed = time.perf_counter() - started
        trace = result.get("retrieval_trace", [])
        last = trace[-1] if trace else {}
        hits = (
            result.get("answer_evidence", result.get("generation_evidence", result.get("evidence", [])))
            if generate
            else result.get("hits", [])
        )
        gold = case.get("expected_evidence", [])
        base = last.get("base_candidates", [])
        ranked = last.get("reranked_candidates", last.get("candidates", []))
        sources = {g["source"] for g in gold}
        covered_sources = {g["source"] for g in gold if any(_match(h, g) for h in hits)}
        sub_gold = case.get("subquestion_evidence", {})
        needs = case.get("required_subquestions", [])
        sub_coverage = None
        if needs and all(sub_gold.get(n) for n in needs):
            sub_coverage = sum(
                all(any(_match(h, g) for h in hits) for g in sub_gold[n]) for n in needs
            ) / len(needs)
        first_ids = set(trace[0].get("context_ids", [])) if trace else set()
        first_context = [
            h
            for h in (trace[0].get("candidates", []) if trace else [])
            if h["chunk_id"] in first_ids
        ]
        first_covered = sum(any(_match(h, g) for h in first_context) for g in gold)
        covered = sum(any(_match(h, g) for h in hits) for g in gold)
        review = reviews.get(case["id"], {})
        validated_review = review if isinstance(review, dict) and review.get("reviewer") else {}
        citation_flags = validated_review.get("citation_correct", [])
        if len(citation_flags) != len(result.get("citations", [])):
            validated_review = {
                key: value for key, value in validated_review.items() if key != "citation_correct"
            }
        pairs = [
            similarity(a["text"], b["text"]) for i, a in enumerate(hits) for b in hits[i + 1 :]
        ]
        row = {
            "id": case["id"],
            "question": case['question'],
            "reference_answer": case.get('reference_answer', ''),
            "split": case.get("split", "development"),
            "answerable": case["answerable"],
            "k": k,
            "stages": {"base": _ranking(base, gold, k), "reranked": _ranking(ranked, gold, k)},
            "subquestion_evidence_coverage": sub_coverage,
            "cross_document_evidence_coverage": len(covered_sources) / len(sources)
            if len(sources) > 1
            else None,
            "gold_evidence_coverage": covered / len(gold) if gold else None,
            "context_redundancy": _mean(pairs) if pairs else 0,
            "context_tokens_estimate": result.get(
                "generation_context_tokens_estimate", result.get("context_tokens", 0)
            )
            if generate
            else result.get("context_tokens", 0),
            "retrieval_seconds": result.get(
                "retrieval_seconds",
                sum(t.get("seconds", 0) for t in trace) + result.get("rewrite_seconds", 0),
            ),
            "rewrite_seconds": result.get("rewrite_seconds", 0),
            "rerank_seconds": sum(t.get('rerank_seconds',0) for t in trace),
            "generation_seconds": result.get("generation_seconds") if generate else None,
            "total_seconds": elapsed,
            "usage": result.get("usage", {}),
            "recall_triggered": len(trace) > 1,
            # Gold rescue means a missing gold group is actually newly present.
            "gold_recall_rescued": bool(len(trace) > 1 and gold and covered > first_covered),
            "gold_recall_complete_rescue": bool(
                len(trace) > 1 and gold and first_covered < len(gold) and covered == len(gold)
            ),
            "ineffective_recall": bool(len(trace)>1 and gold and covered<=first_covered),
            "question_type": case.get('category','unknown'),
            "literal_answer_check": all(_compact(term).casefold() in _compact(result.get('answer','')).casefold() for term in case.get('automatic_answer_contains',[])) if generate and case.get('automatic_answer_contains') else None,
            "review": validated_review,
            "result": result,
            "model_context_gold_coverage": sum(any(_match(h, g) for h in result.get('generation_evidence', [])) for g in gold) / len(gold) if gold and generate else None,
            "citation_outside_gold_sources": [c.get('source') for c in result.get('citations', []) if sources and c.get('source') not in sources],
        }
        if generate:
            row.update(
                refusal_correct=(result["status"] == "insufficient_evidence")
                if not case["answerable"]
                else None,
                false_refusal=(result["status"] == "insufficient_evidence")
                if case["answerable"]
                else None,
                generation_failed=result["status"] == "generation_failed",
            )
        rows.append(row)
        if on_row:
            on_row(row)

    def aggregate(subset):
        def values(key):
            return [r[key] for r in subset if r.get(key) is not None]

        stages = {}
        for stage in ["base", "reranked"]:
            stages[stage] = {
                metric: _mean(
                    [
                        r["stages"][stage][metric]
                        for r in subset
                        if r["stages"][stage][metric] is not None
                    ]
                )
                for metric in ["hit_at_1", "hit_at_3", "hit_at_k", "recall_at_k", "mrr_at_k"]
            }
        answers = [
            r["review"]["answer_correct"]
            for r in subset
            if type(r["review"].get("answer_correct")) is bool
        ]
        citations = [
            v for r in subset for v in r["review"].get("citation_correct", []) if type(v) is bool
        ]
        triggered = [r for r in subset if r["recall_triggered"] and r["answerable"]]
        return {
            "questions": len(subset),
            "status_counts": dict(Counter(r['result'].get('status', 'retrieval_only') for r in subset)),
            "task_incomplete_count": sum(bool(r['result'].get('unanswered_subquestions')) for r in subset),
            "citation_outside_gold_sources_count": sum(bool(r['citation_outside_gold_sources']) for r in subset),
            "model_context_gold_coverage": _mean(values('model_context_gold_coverage')),
            "fixed_k": k,
            "stages": stages,
            "hit_at_1": stages["reranked"]["hit_at_1"],
            "hit_at_3": stages["reranked"]["hit_at_3"],
            "recall_at_k": stages["reranked"]["recall_at_k"],
            "mrr": stages["reranked"]["mrr_at_k"],
            "subquestion_evidence_coverage": _mean(values("subquestion_evidence_coverage")),
            "cross_document_evidence_coverage": _mean(values("cross_document_evidence_coverage")),
            "gold_evidence_coverage": _mean(values("gold_evidence_coverage")),
            "context_redundancy": _mean(values("context_redundancy")),
            "answer_accuracy": _mean(answers),
            "answer_reviewed_count": len(answers),
            "citation_precision": _mean(citations),
            "citation_reviewed_count": len(citations),
            "refusal_accuracy": _mean(values("refusal_correct")),
            "answerable_false_refusal_rate": _mean(values("false_refusal")),
            "generation_failed_rate": _mean(values("generation_failed")),
            "context_tokens_mean": _mean(values("context_tokens_estimate")),
            "retrieval_seconds_mean": _mean(values("retrieval_seconds")),
            "retrieval_seconds_p95": _p95(values("retrieval_seconds")),
            "rerank_seconds_mean": _mean(values('rerank_seconds')),
            "rerank_seconds_p95": _p95(values('rerank_seconds')),
            "total_seconds_mean": _mean(values('total_seconds')),
            "total_seconds_p95": _p95(values('total_seconds')),
            "generation_seconds_mean": _mean(values("generation_seconds")),
            "generation_seconds_p95": _p95(values("generation_seconds")),
            "usage": {
                key: sum(r["usage"].get(key, 0) for r in subset)
                for key in sorted({key for r in subset for key in r["usage"]})
            },
            "second_recall_rate": _mean(values("recall_triggered")),
            "gold_recall_rescue_rate": _mean([r["gold_recall_rescued"] for r in triggered]),
            "gold_recall_complete_rescue_rate": _mean(
                [r["gold_recall_complete_rescue"] for r in triggered]
            ),
            "ineffective_recall_rate": _mean([r['ineffective_recall'] for r in triggered]),
            "literal_answer_check_rate": _mean(values('literal_answer_check')),
            "human_answer_completeness": _mean([r['review']['answer_complete'] for r in subset if type(r['review'].get('answer_complete')) is bool]),
            "table_numeric_error_rate": _mean([r['review']['table_numeric_error'] for r in subset if type(r['review'].get('table_numeric_error')) is bool]),
        }

    return {
        "format_version": 2,
        "run_label": run_label,
        "mode": "generated" if generate else "retrieval_only",
        "summary": aggregate(rows),
        "by_split": {
            split: aggregate([r for r in rows if r["split"] == split])
            for split in sorted({r["split"] for r in rows})
        },
        "by_type": {kind: aggregate([r for r in rows if r['question_type']==kind]) for kind in sorted({r['question_type'] for r in rows})},
        "rows": rows,
        "notes": [
            "Existing dev/test questions are seen development regressions, not untouched held-out data.",
            "K is fixed; MRR is truncated at K. Both retrieval stages reported separately.",
            "Subquestion coverage is null without explicit reviewed mapping; excerpt matching is not entailment.",
            "Human generation metrics remain null without attributed, supplied reviews.",
            "Tokens are estimates; Provider-reported tokens, when present, are separate usage counters.",
            "Cold/hot timing requires explicitly labeled repeated runs on the same environment.",
        ],
    }


def write_report(report, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    lines = [
        "# RAG评测结果",
        "",
        f"模式：{report['mode']}",
        "",
        "| 指标 | 实际结果 |",
        "| --- | --- |",
    ]
    for key, value in report["summary"].items():
        lines.append(
            f"| {key} | {json.dumps(value, ensure_ascii=False) if value is not None else '未评审/未运行'} |"
        )
    lines += [
        "",
        "## 逐题状态",
        "",
        "| 问题 | 数据划分 | 重排Hit@3 | 回答状态 |",
        "| --- | --- | --- | --- |",
    ]
    for r in report["rows"]:
        lines.append(
            f"| {r['id']} | {r['split']} | {r['stages']['reranked']['hit_at_3']} | {r['result'].get('status', '仅检索')} |"
        )
    lines += ["", "## 限制", ""] + ["- " + n for n in report["notes"]]
    destination.with_suffix(".md").write_text("\n".join(lines) + "\n")
    if report['mode'] == 'generated':
        with destination.with_suffix('.review.csv').open('w', encoding='utf-8-sig', newline='') as handle:
            fields = ['id', 'question', 'reference_answer', 'answer', 'status', 'answer_mode',
                      'quotes', 'source_locations', 'automatic_checks', 'human_answer_correct',
                      'human_citation_supported', 'error_reason', 'notes']
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in report['rows']:
                r = row['result']
                writer.writerow({'id': row['id'], 'question': row['question'], 'reference_answer': row['reference_answer'],
                    'answer': r.get('answer', ''), 'status': r['status'], 'answer_mode': r.get('answer_mode', 'model_generation'),
                    'quotes': json.dumps([c.get('excerpts', []) for c in r.get('citations', [])], ensure_ascii=False),
                    'source_locations': json.dumps([{k: c.get(k) for k in ['source', 'doc_id', 'page', 'bbox', 'location']} for c in r.get('citations', [])], ensure_ascii=False),
                    'automatic_checks': json.dumps({'warnings': r.get('warnings', []), 'gold_coverage': row['gold_evidence_coverage'], 'citation_outside_gold_sources': row['citation_outside_gold_sources'], 'unanswered_tasks': r.get('unanswered_subquestions', [])}, ensure_ascii=False),
                    'notes': 'PENDING_HUMAN; automatic fields are not semantic accuracy scores'})


def ablation_policies():
    base = RetrievalPolicy.baseline()
    lightweight = replace(base, reranker="lightweight")
    mmr = replace(lightweight, use_mmr=True, alpha=0.8, beta=0.2)
    neighbors = replace(mmr, use_neighbors=True)
    coverage = replace(neighbors,use_coverage=True,alpha=.65,beta=.20,gamma=.15)
    adaptive = replace(coverage, adaptive=True)
    directed = replace(coverage, use_recall=True)
    full = replace(adaptive,use_recall=True,use_rewrite=True,reranker=None,max_generation_calls=4,atomic_generation=True,allow_calculation=True)
    return {
        "baseline_rrf": base,
        "lightweight": lightweight,
        "crossencoder": replace(lightweight, reranker="crossencoder"),
        "mmr": mmr,
        "neighbors": neighbors,
        "adaptive": adaptive,
        "directed_recall": directed,
        "coverage": coverage,
        "full": full,
        "full_no_rewrite": replace(full,use_rewrite=False),
        "full_no_mmr": replace(full,use_mmr=False),
        "full_no_neighbors": replace(full,use_neighbors=False),
        "full_no_coverage": replace(full,use_coverage=False),
        "full_no_adaptive": replace(full,adaptive=False),
        "full_no_recall": replace(full,use_recall=False),
        "full_no_rerank": replace(full,reranker='none'),
        "l3_off": full,
    }
