"""真实 PDF → docparse → BGE 512 维向量 → Milvus，保留数据供再次查询。

在项目根目录加载 .env 后运行：
    python 已有资源/样例/demo_pdf_milvus.py /path/to/report.pdf
    python 已有资源/样例/demo_pdf_milvus.py /path/to/report.pdf \
        --query-only --query "如何改善少数类别的识别效果？"

固定使用 BAAI/bge-small-zh-v1.5，独立集合不会与 HashingEmbedder demo 混用。
失败返回非零退出码；不会自动删除集合。验证结果和解析产物保存在 data/ 下。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / "代码"))

from chunking import chunk_docparse_ir
from docparse_connector import DocparseConnector, DocparseOptions
from document_model import stable_doc_id
from embedders import SentenceTransformerEmbedder
from grounded_answer import generate_answer
from milvus_hybrid_index import OUTPUT_FIELDS, MilvusHybridIndex

MODEL = "BAAI/bge-small-zh-v1.5"
COLLECTION = "pdf_bge_small_zh_v15"
DEFAULT_QUERIES = [
    "如何改善少数类别的识别效果？",
    "怎样判断预测的房价偏高还是偏低？",
    "如何估计回归系数的不确定性？",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--query", action="append", help="可重复传入多个问题")
    parser.add_argument("--query-only", action="store_true", help="只查询已入库文档")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--answer", action="store_true", help="用本地 Ollama 生成带引用的回答"
    )
    parser.add_argument(
        "--llm-model", default="gemma3:4b", help="Ollama 中已安装的问答模型"
    )
    parser.add_argument(
        "--answer-retrieval",
        choices=["dense", "hybrid"],
        default="hybrid",
        help="生成回答所用的检索结果，默认 hybrid",
    )
    args = parser.parse_args()
    source = args.pdf.expanduser().resolve()
    if not source.is_file() or source.suffix.lower() != ".pdf":
        parser.error("请传入存在的 PDF 文件")
    if not 1 <= args.top_k <= 10:
        parser.error("--top-k 应为 1 到 10")
    queries = args.query or DEFAULT_QUERIES
    if any(not query.strip() for query in queries):
        parser.error("查询不能为空")

    # 与 DocparseConnector 相同，以文件名作为逻辑来源；同名文件视为同一文档。
    doc_id = stable_doc_id(source.name)
    data_root = ROOT / "data" / "semantic_demo" / doc_id
    data_root.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="run_", dir=data_root))
    report = {
        "started_at": datetime.now(UTC).isoformat(),
        "status": "RUNNING",
        "source": str(source),
        "doc_id": doc_id,
        "model": MODEL,
        "dimension": 512,
        "collection": COLLECTION,
        "query_only": args.query_only,
        "queries": [],
    }
    index = None
    try:
        print(f"[1/4] 加载真实嵌入模型 {MODEL}（首次运行需下载）", flush=True)
        embedder = SentenceTransformerEmbedder(model_name=MODEL, dim=512, device="cpu")
        probe = embedder.embed([queries[0]], query=True)
        if len(probe) != 1 or len(probe[0]) != 512:
            raise RuntimeError("模型输出维度不是 512")

        index = MilvusHybridIndex(
            embedder,
            uri=os.environ.get("MILVUS_URI", "http://localhost:19530"),
            db_name=os.environ.get("MILVUS_DB", "course_rag"),
            collection=COLLECTION,
        )
        if args.query_only:
            if not index.client().has_collection(COLLECTION):
                raise RuntimeError("尚未入库，请先不带 --query-only 运行")
        else:
            index.ensure_ready()
        schema = index.client().describe_collection(COLLECTION)
        dense = next(field for field in schema["fields"] if field["name"] == "dense")
        if int(dense["params"]["dim"]) != 512:
            raise RuntimeError("已有集合维度不兼容；未修改集合")

        if not args.query_only:
            print("[2/4] 解析 PDF 并切块", flush=True)
            repo = os.environ.get("DOCPARSE_REPO", "")
            python = os.environ.get("DOCPARSE_PYTHON", "")
            if not repo or not python or not Path(python).is_file():
                raise RuntimeError(
                    "请先加载 .env，并检查 DOCPARSE_REPO / DOCPARSE_PYTHON"
                )
            connector = DocparseConnector(
                DocparseOptions(
                    repo=str(Path(repo).resolve()),
                    python=python,
                    mode=os.environ.get("DOCPARSE_MODE", "auto"),
                    timeout_s=int(os.environ.get("DOCPARSE_TIMEOUT_S", "1800")),
                )
            )
            document, ir = connector.parse(source, run_dir / "parsed")
            chunks = chunk_docparse_ir(ir, document)
            if not chunks or document.doc_id != doc_id:
                raise RuntimeError("解析为空或文档 ID 不一致")
            report["pages"] = document.metadata["pages"]
            report["parse_modes"] = dict(
                Counter(page.get("mode", "unknown") for page in ir["pages"])
            )
            report["needs_review"] = ir.get("needs_review", False)
            print(
                f"  {report['pages']} 页，{len(chunks)} 个切块，解析方式 {report['parse_modes']}",
                flush=True,
            )
            print("[3/4] 编码并入库（重复运行更新该文档，保留集合）", flush=True)
            report["ingest"] = index.ingest(
                doc_id,
                chunks,
                source=document.source,
                sha256=document.metadata["file_sha256"],
            )
            expected_count = len(chunks)
        else:
            print("[2/4] 使用已入库文档；[3/4] 跳过解析和写入", flush=True)
            expected_count = None

        # 重新连接后读取，验证数据不依赖本次 Python 进程。
        index.close()
        client = index.client()
        expr = f'doc_id == "{doc_id}"'
        count = client.query(
            COLLECTION,
            filter=expr,
            output_fields=["count(*)"],
            consistency_level="Strong",
        )[0]["count(*)"]
        if count <= 0 or (expected_count is not None and count != expected_count):
            raise RuntimeError(f"文档入库行数异常：{count}，预期 {expected_count}")
        report["document_rows"] = count
        print(f"[4/4] 重连后可读 {count} 行；分别执行纯向量和混合检索", flush=True)
        for question in queries:
            vector = embedder.embed([question], query=True)[0]
            results = client.search(
                COLLECTION,
                data=[vector],
                anns_field="dense",
                filter=expr,
                search_params={"metric_type": "IP", "params": {"ef": 64}},
                limit=args.top_k,
                output_fields=OUTPUT_FIELDS,
                consistency_level="Strong",
            )
            dense_hits = [
                dict(hit["entity"], id=hit["id"], score=float(hit["distance"]))
                for hit in results[0]
            ]
            hybrid_hits = index.query(question, top_k=args.top_k, doc_ids=[doc_id])[
                "hits"
            ]
            print(f"\n问题：{question}", flush=True)
            for label, hits in (("纯向量", dense_hits), ("混合检索", hybrid_hits)):
                if not hits:
                    raise RuntimeError(f"{label}未返回命中：{question}")
                print(f"  {label}：", flush=True)
                for rank, hit in enumerate(hits, 1):
                    if (
                        hit["source"] != source.name
                        or hit["doc_id"] != doc_id
                        or not hit["page"]
                        or not hit["text"].strip()
                    ):
                        raise RuntimeError("命中的来源、页码或正文无效")
                    snippet = hit["text"].replace("\n", " ")[:160]
                    print(
                        f"    [{rank}] p{hit['page']} score={hit['score']:.4f} | {snippet}",
                        flush=True,
                    )
            query_result = {
                "question": question,
                "dense": dense_hits,
                "hybrid": hybrid_hits,
            }
            report["queries"].append(query_result)
            if args.answer:
                print(
                    f"\n正在用 {args.llm_model} 生成回答（首次加载可能稍慢）…",
                    flush=True,
                )
                query_result["answer_retrieval"] = args.answer_retrieval
                answer = generate_answer(
                    question,
                    query_result[args.answer_retrieval],
                    model=args.llm_model,
                    base_url=os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434"),
                )
                query_result["generated_answer"] = answer
                print("\n=== 基于文档的回答 ===\n" + answer["answer"], flush=True)
                if answer.get("discarded_claims"):
                    print(
                        "部分生成段落未通过引用校验，已省略；上方仅显示通过校验的部分。",
                        flush=True,
                    )
                for citation in answer["citations"]:
                    print(
                        f"来源 [{citation['id']}] {citation['source']}，第 {citation['page']} 页",
                        flush=True,
                    )
                if answer["status"] == "ANSWERED":
                    print(
                        "引用编号及原文摘录已校验；请对照原文核对回答内容。", flush=True
                    )
                else:
                    print("证据不足，未生成事实回答。", flush=True)
        report["status"] = "PASS"
        report["scope"] = (
            "真实向量入库、重连读取、两路检索和来源字段通过；相关性需核对原文，不代表全面质量评测。"
        )
        if args.answer:
            report["scope"] += (
                " 已完成本地回答生成及引用编号校验；不保证回答事实全部正确。"
            )
        summary = "PASS：重连读取及带来源检索通过；数据已保留。"
        if args.answer:
            summary += " 本地问答流程完成（事实正确性需核对）。"
        print("\n" + summary, flush=True)
    except Exception as exc:  # noqa: BLE001 - CLI 边界记录失败并返回非零，不把外部服务错误当作 skip。
        report["status"] = "FAIL"
        report["error"] = f"{type(exc).__name__}: {exc}"
        print(f"FAIL: {report['error']}", file=sys.stderr, flush=True)
    finally:
        if index is not None:
            index.close()
        report_path = run_dir / "result.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"验证记录：{report_path}", flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
