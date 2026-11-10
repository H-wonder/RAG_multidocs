"""可选 demo：docparse 多模态解析 → 统一文档 → 切块。

前置：
- 解析环境 docparse 已安装；
- 设置 DOCPARSE_REPO（含 docparse/ 包的目录）；
- 设置 DOCPARSE_PYTHON（docparse 环境的 Python）；
- 扫描件/图片需要 Ollama 和 OCR 模型；纯文本 PDF 可走原生抽取，但 docparse CLI
  入口仍会 ping Ollama，见文档 07 第 5 节。

运行：
    cd 已有资源/样例
    DOCPARSE_REPO=../可选组件/docparse \
    DOCPARSE_PYTHON=$(conda run -n docparse which python) \
    python demo_docparse.py /path/to/sample.pdf

没有传入 PDF 或环境未配置时，打印说明并正常退出（退出码 0）。
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "代码"))

from chunking import chunk_docparse_ir  # noqa: E402
from docparse_connector import (  # noqa: E402
    DocparseConnector,
    DocparseError,
    DocparseOptions,
    DocparseUnavailable,
)


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: python demo_docparse.py /path/to/sample.pdf")
        print("skip: 未传入 PDF")
        return 0

    repo = os.environ.get("DOCPARSE_REPO", "")
    python = os.environ.get("DOCPARSE_PYTHON", sys.executable)
    if not repo:
        print("skip: 未设置 DOCPARSE_REPO（示例：../可选组件/docparse）")
        return 0

    source = Path(argv[1])
    if not source.is_file():
        print(f"FILE_NOT_FOUND: {source}")
        return 2

    connector = DocparseConnector(
        DocparseOptions(repo=repo, python=python, mode=os.environ.get("DOCPARSE_MODE", "auto"))
    )

    try:
        with tempfile.TemporaryDirectory(prefix="docparse_demo_") as outdir:
            document, ir = connector.parse(source, outdir)
            chunks = chunk_docparse_ir(ir, document)
    except DocparseUnavailable as exc:
        print(f"skip: {exc}")
        return 0
    except DocparseError as exc:
        print(f"PARSE_ERROR: {exc}")
        return 2

    print(f"title : {document.title!r}")
    print(f"pages : {document.metadata.get('pages')}")
    print(f"source: {document.source}")
    print(f"chunks: {len(chunks)}")
    for chunk in chunks[:5]:
        section = f" §{chunk.section_path}" if chunk.section_path else ""
        snippet = chunk.text.replace("\n", " ")[:70]
        print(f"  [{chunk.content_type:9}] p{chunk.page}{section} | {snippet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
