"""离线端到端 demo：TXT / Markdown / docparse IR → 统一文档 → 切块 → 关键词检索。

不依赖 Milvus、嵌入模型或 Ollama，适合验证最小 RAG 闭环和 CI。

运行：
    cd 已有资源/样例
    python demo_rag.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "代码"))

from chunking import chunk_docparse_ir, chunk_markdown, chunk_text  # noqa: E402
from dedup import hamming, simhash64  # noqa: E402
from docparse_connector import document_from_ir  # noqa: E402
from document_model import Document  # noqa: E402
from keyword_index import KeywordIndex  # noqa: E402


def load_txt(path: Path) -> list:
    text = path.read_text(encoding="utf-8")
    document = Document.from_text(text, path.name, title=path.stem)
    return chunk_text(document)


def load_markdown_dir(directory: Path) -> list:
    chunks = []
    for path in sorted(directory.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        document = Document.from_text(
            text, path.name, title=path.stem, media_type="text/markdown"
        )
        chunks.extend(chunk_markdown(document))
    return chunks


def load_docparse_ir(path: Path) -> list:
    ir = json.loads(path.read_text(encoding="utf-8"))
    document = document_from_ir(ir, path.name.replace(".json", ".pdf"))
    return chunk_docparse_ir(ir, document)


def main() -> None:
    index = KeywordIndex()

    sources = {
        "TXT": load_txt(HERE / "notes.txt"),
        "Markdown": load_markdown_dir(HERE / "markdown"),
        "docparse IR": load_docparse_ir(HERE / "docparse_ir_sample.json"),
    }

    print("=== 接入 ===")
    for name, chunks in sources.items():
        added = index.add(chunks)
        print(f"{name:12} {added:3} chunks")

    print("\n=== 索引 ===")
    print(index.stats())

    print("\n=== 检索（带来源）===")
    for query in ("检索器如何训练", "表格", "RAG 是什么"):
        print(f"\nQ: {query}")
        hits = index.search(query, top_k=3)
        if not hits:
            print("  (no hits)")
            continue
        for rank, hit in enumerate(hits, start=1):
            section = f" §{hit['section_path']}" if hit["section_path"] else ""
            snippet = hit["text"].replace("\n", " ")[:70]
            print(
                f"  [{rank}] {hit['score']:.3f}  {hit['source']}"
                f" p{hit['page']}{section} | {snippet}"
            )

    print("\n=== 去重 ===")
    original = (HERE / "notes.txt").read_text(encoding="utf-8")
    near_duplicate = original.replace("联合优化", "共同优化", 1)
    distance = hamming(simhash64(original), simhash64(near_duplicate))
    print(f"near-duplicate hamming distance = {distance} (<= 10 视为近重复)")

    print("\n=== 错误处理 ===")
    missing = HERE / "missing.txt"
    if not missing.exists():
        print(f"FILE_NOT_FOUND: {missing.name}（CLI 应返回非 0 退出码，其他文件继续）")

    print("\n=== 持久化 ===")
    out = HERE / "demo_index.json"
    index.save(out)
    reloaded = KeywordIndex.load(out)
    print(f"saved -> {out.name}; reloaded hits={len(reloaded.search('检索器'))}")
    out.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
