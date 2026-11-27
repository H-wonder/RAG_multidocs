"""统一入口：rag.py ingest 文件 --root 知识库根目录；rag.py ask 问题；rag.py list。"""

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=ROOT / "data/rag_strategy")
    parser.add_argument("--collection", default="rag_strategy_bge_v1")
    sub = parser.add_subparsers(dest="command", required=True)
    ingest = sub.add_parser("ingest")
    ingest.add_argument("files", nargs="+", type=Path)
    ingest.add_argument("--root", required=True, type=Path)
    ask = sub.add_parser("ask")
    ask.add_argument("question")
    ask.add_argument("--source", action="append")
    ask.add_argument(
        "--tone",
        choices=["professional", "teaching", "concise"],
        default="professional",
    )
    ask.add_argument(
        "--reranker", choices=["lightweight", "crossencoder"], default="lightweight"
    )
    ask.add_argument("--rerank-model", default="BAAI/bge-reranker-base")
    ask.add_argument("--retrieve-only", action="store_true")
    ask.add_argument(
        "--history", type=Path, help="JSON消息列表，只使用最近两条用户问题"
    )
    ask.add_argument("--min-dense", type=float, default=0.35)
    ask.add_argument("--candidate-k", type=int, default=24)
    ask.add_argument("--context-k", type=int, default=3)
    ask.add_argument("--context-tokens", type=int, default=1800)
    ask.add_argument("--no-recall", action="store_true", help="关闭第二轮召回生成")
    ask.add_argument("--model", default="gemma3:4b")
    sub.add_parser("list")
    args = parser.parse_args()
    index = MilvusHybridIndex(
        SentenceTransformerEmbedder(),
        uri=os.getenv("MILVUS_URI", "http://localhost:19530"),
        db_name=os.getenv("MILVUS_DB", "course_rag"),
        collection=args.collection,
    )
    try:
        catalog = Catalog(args.state / "catalog.sqlite", index)
        if args.command == "ingest":
            for path in args.files:
                print(
                    json.dumps(
                        catalog.ingest(
                            path,
                            args.root,
                            parser_python=os.getenv("DOCPARSE_PYTHON", ""),
                            parser_repo=os.getenv("DOCPARSE_REPO", ""),
                        ),
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
        elif args.command == "list":
            print(json.dumps(catalog.list_documents(), ensure_ascii=False, indent=2))
        else:
            config = Strategy(
                tone=args.tone,
                reranker=args.reranker,
                rerank_model=args.rerank_model,
                min_dense=args.min_dense,
                candidate_k=args.candidate_k,
                context_k=args.context_k,
                context_tokens=args.context_tokens,
                retry_recall=not args.no_recall,
            )
            history = json.loads(args.history.read_text()) if args.history else []
            engine = RagEngine(index, config, neighbor_fn=catalog.neighbors)
            if args.retrieve_only:
                engine.answer_fn = lambda question, hits, **kw: {
                    "status": "ANSWERED",
                    "answer": "仅检索模式",
                    "citations": [],
                }
            result = engine.ask(
                args.question,
                filter_expr=catalog.active_filter(args.source),
                history=history,
                model=args.model,
                base_url=os.getenv("OLLAMA_URL", "http://127.0.0.1:11434"),
            )
            import tempfile

            args.state.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".json",
                prefix="query_",
                dir=args.state,
                delete=False,
                encoding="utf-8",
            ) as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
                record = f.name
            print(result["answer"]["answer"])
            if args.retrieve_only and result["trace"]:
                step = result["trace"][-1]
                selected = set(step["context_ids"])
                for h in step["candidates"]:
                    if h["id"] in selected:
                        print(
                            f"{h['source']} p{h['page']} rerank={h['rerank_score']:.4f} | {h['text'][:180]}"
                        )
            for c in result["answer"]["citations"]:
                print(f"[{c['id']}] {c['source']} p{c['page']}")
            if result["answer"].get("discarded_claims"):
                print("部分生成段落未通过引用检查，已省略。")
            print("策略记录：", record)
            if result["answer"]["status"] == "GENERATION_FAILED":
                raise SystemExit(1)
    finally:
        index.close()


if __name__ == "__main__":
    main()
