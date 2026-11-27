"""评测人工标注问题集：eval_rag.py cases.json；输出证据页命中率/MRR/拒答结果。

cases JSON 示例：[{"question":"...","expected_pages":[7],"source":"relative.pdf"},
{"question":"文档外问题","unanswerable":true}]。页级标注不能代替块级/事实人工评审。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "已有资源/代码"))
from embedders import SentenceTransformerEmbedder
from milvus_hybrid_index import MilvusHybridIndex
from rag_catalog import Catalog
from rag_strategy import RagEngine, Strategy


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cases", type=Path)
    p.add_argument("--state", type=Path, default=ROOT / "data/rag_strategy")
    p.add_argument("--collection", default="rag_strategy_bge_v1")
    p.add_argument(
        "--reranker", choices=["lightweight", "crossencoder"], default="lightweight"
    )
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    cases = json.loads(args.cases.read_text())
    if not cases or any(
        not c.get("question")
        or (not c.get("unanswerable") and not c.get("expected_pages"))
        for c in cases
    ):
        p.error("问题集为空或缺少expected_pages/unanswerable标注")
    index = MilvusHybridIndex(
        SentenceTransformerEmbedder(),
        uri=os.getenv("MILVUS_URI", "http://localhost:19530"),
        db_name=os.getenv("MILVUS_DB", "course_rag"),
        collection=args.collection,
    )
    records = []
    try:
        catalog = Catalog(args.state / "catalog.sqlite", index)
        engine = RagEngine(
            index, Strategy(reranker=args.reranker), neighbor_fn=catalog.neighbors
        )
        for case in cases:
            result = engine.ask(
                case["question"],
                filter_expr=catalog.active_filter(
                    [case["source"]] if case.get("source") else None
                ),
                base_url=os.getenv("OLLAMA_URL", "http://127.0.0.1:11434"),
            )

            def relevant(h, case=case):
                return h["page"] in case.get("expected_pages", []) and (
                    not case.get("source") or h["source"] == case["source"]
                )

            candidates = result["trace"][0]["candidates"]
            context_ids = set(result["trace"][0]["context_ids"])
            context = [h for h in candidates if h["id"] in context_ids]
            record = {"case": case, "result": result}
            if case.get("unanswerable"):
                record["abstention_correct"] = (
                    result["answer"]["status"] == "INSUFFICIENT_EVIDENCE"
                )
            else:
                ranks = [i + 1 for i, h in enumerate(candidates) if relevant(h)]
                record["page_mrr"] = 1 / ranks[0] if ranks else 0
                baseline = result["trace"][0].get("recalled_candidates", [])
                baseline_ranks = [i + 1 for i, h in enumerate(baseline) if relevant(h)]
                record["rrf_baseline_page_mrr"] = (
                    1 / baseline_ranks[0] if baseline_ranks else 0
                )
                record["context_page_hit"] = any(relevant(h) for h in context)
                record["answer_generated"] = result["answer"]["status"] == "ANSWERED"
            records.append(record)
            print(case["question"], result["answer"]["status"], flush=True)
        supported = [r for r in records if not r["case"].get("unanswerable")]
        outside = [r for r in records if r["case"].get("unanswerable")]
        summary = {
            "rrf_baseline_page_mrr": sum(r["rrf_baseline_page_mrr"] for r in supported)
            / len(supported)
            if supported
            else None,
            "cases": len(records),
            "supported": len(supported),
            "page_mrr": sum(r["page_mrr"] for r in supported) / len(supported)
            if supported
            else None,
            "context_page_hit_rate": sum(r["context_page_hit"] for r in supported)
            / len(supported)
            if supported
            else None,
            "supported_answer_rate": sum(r["answer_generated"] for r in supported)
            / len(supported)
            if supported
            else None,
            "unanswerable_abstention_rate": sum(
                r["abstention_correct"] for r in outside
            )
            / len(outside)
            if outside
            else None,
            "scope": "页级小样本回归；不代表通用检索质量或回答事实准确率",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {"summary": summary, "records": records}, ensure_ascii=False, indent=2
            )
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        index.close()


if __name__ == "__main__":
    main()
