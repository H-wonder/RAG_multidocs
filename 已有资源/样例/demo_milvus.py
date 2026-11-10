"""可选 demo：Milvus 混合检索链路冒烟测试。

前置：
- rag conda 环境已安装 pymilvus；
- Milvus 2.5+ 已启动（默认 http://localhost:19530）。

使用 HashingEmbedder，避免下载嵌入模型；它没有语义能力，只验证
“切块 → 入库 → dense + BM25 混合检索 → 带来源命中”的链路。

运行：
    cd 已有资源/样例
    python demo_milvus.py

Milvus 不可用时脚本会打印提示并正常退出（退出码 0），方便 CI 跳过。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "代码"))

from embedders import HashingEmbedder  # noqa: E402
from milvus_hybrid_index import MilvusHybridIndex  # noqa: E402
from demo_rag import (  # noqa: E402
    load_docparse_ir,
    load_markdown_dir,
    load_txt,
)


def main() -> int:
    try:
        import pymilvus  # noqa: F401
    except ImportError:
        print("skip: 未安装 pymilvus（pip install 'pymilvus>=2.5'）")
        return 0

    uri = os.environ.get("MILVUS_URI", "http://localhost:19530")
    db_name = os.environ.get("MILVUS_DB", "course_rag")
    collection = os.environ.get("MILVUS_DEMO_COLLECTION", "demo_chunks")

    index = MilvusHybridIndex(
        HashingEmbedder(dim=256),
        uri=uri,
        db_name=db_name,
        collection=collection,
    )

    try:
        index.ensure_ready()
    except Exception as exc:
        print(f"skip: 无法连接 Milvus ({uri}): {exc}")
        print("      先启动：docker start milvus-standalone")
        return 0

    chunks = (
        load_txt(HERE / "notes.txt")
        + load_markdown_dir(HERE / "markdown")
        + load_docparse_ir(HERE / "docparse_ir_sample.json")
    )

    grouped: dict[str, list] = {}
    for chunk in chunks:
        grouped.setdefault(chunk.doc_id, []).append(chunk)

    try:
        for doc_id, group in grouped.items():
            result = index.ingest(
                doc_id,
                group,
                source=group[0].source,
                sha256="",
            )
            print(f"ingest {group[0].source}: {result['chunks']} chunks")

        result = index.query("检索器如何训练", top_k=3)
        print("\n=== Milvus 混合检索 ===")
        for hit in result["hits"]:
            section = f" §{hit['section_path']}" if hit.get("section_path") else ""
            snippet = (hit.get("text") or "").replace("\n", " ")[:70]
            print(
                f"[{hit['rank']}] {hit['score']:.4f} {hit.get('source','')}"
                f" p{hit.get('page', 0)}{section} | {snippet}"
            )
        print("\nstats:", index.stats())
    finally:
        # 清理本次 demo 的集合，避免污染真实数据
        print("cleanup:", index.drop_collection())
        index.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
