"""演示：TXT / Markdown / docparse IR 三种输入走同一套文档模型与切块器。

运行：
    cd 已有资源/样例
    python3 demo.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "代码"))

from chunking import chunk_docparse_ir, chunk_markdown, chunk_text  # noqa: E402
from dedup import simhash64  # noqa: E402
from docparse_connector import document_from_ir  # noqa: E402
from document_model import Document  # noqa: E402


def show(title: str, chunks) -> None:
    print(f"\n=== {title}: {len(chunks)} chunks ===")
    for chunk in chunks[:5]:
        section = f" §{chunk.section_path}" if chunk.section_path else ""
        text = chunk.text.replace("\n", " ")[:80]
        print(
            f"[{chunk.content_type:9}] p{chunk.page} {chunk.source}{section} | {text}"
        )


def main() -> None:
    txt_path = HERE / "notes.txt"
    txt = txt_path.read_text(encoding="utf-8")
    txt_doc = Document.from_text(txt, "notes.txt", title="RAG 笔记")
    print(f"TXT doc_id={txt_doc.doc_id} sha256={txt_doc.metadata['sha256'][:12]}…")
    print(f"TXT simhash={simhash64(txt)}")
    show("TXT", chunk_text(txt_doc))

    md_dir = HERE / "markdown"
    all_md_chunks = []
    for path in sorted(md_dir.glob("*.md")):
        md = path.read_text(encoding="utf-8")
        document = Document.from_text(
            md, path.name, title=path.stem, media_type="text/markdown"
        )
        all_md_chunks.extend(chunk_markdown(document))
    show("Markdown", all_md_chunks)

    ir = json.loads((HERE / "docparse_ir_sample.json").read_text(encoding="utf-8"))
    ir_doc = document_from_ir(ir, "rag-intro.pdf")
    print(f"\nIR title={ir_doc.title!r} pages={ir_doc.metadata['pages']}")
    show("docparse IR", chunk_docparse_ir(ir, ir_doc))


if __name__ == "__main__":
    main()
