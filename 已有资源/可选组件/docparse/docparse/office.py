"""Native parsers for office/text formats.

These never call the VLM: the structure is already machine-readable, and OCR
would only add errors. Dependencies are imported lazily so a missing optional
package does not break the rest of the pipeline.
"""

from __future__ import annotations

import html as _html
from pathlib import Path

from .ir import Block, Document, Page
from .pdfio import sha256_file


def _table_html(rows: list[list[str]]) -> str:
    body = []
    for row in rows:
        body.append("<tr>" + "".join(
            f"<td>{_html.escape(c or '').strip()}</td>" for c in row) + "</tr>")
    return "<table>\n" + "\n".join(body) + "\n</table>"


def _emit(blocks: list[Block], page_no: int, btype: str, content: str,
          *, html: str | None = None, meta: dict | None = None) -> None:
    if not (content or html):
        return
    blocks.append(Block(id=f"p{page_no}b{len(blocks)}", page=page_no, type=btype,
                        bbox=[0, 0, 1000, 1000], reading_order=len(blocks),
                        content=content, html=html, source="native_office",
                        confidence=0.99, meta=meta or {}))


def structured_parse(path: Path, options: dict | None = None) -> dict:
    """Public source-mapped Office service used by independent adapters."""
    from . import office_structured
    parsers = {".docx": office_structured.parse_docx, ".xlsx": office_structured.parse_xlsx}
    path = Path(path)
    return parsers[path.suffix.lower()]({"path": str(path), "options": dict(options or {})})


def _structured_ir(path: Path, parsed: dict) -> Document:
    doc = Document(path=str(path), sha256=sha256_file(path), doc_type=path.suffix[1:])
    doc.meta.update(parsed.get("metadata", {}))
    doc.meta.update({"title": parsed.get("title", path.stem), "page_semantics": "unpaginated_office"})
    doc.warnings.extend(parsed.get("metadata", {}).get("warnings", []))
    page = Page(number=0, kind="native", mode="native")
    for row in parsed["blocks"]:
        meta = dict(row.get("meta", {}))
        for key in ["section_path", "level", "caption", "ocr_text", "origin", "ocr_mode", "needs_review"]:
            if key in row:
                meta[key] = row[key]
        html = _table_html(meta["table_data"]["rows"]) if row["type"] == "table" else None
        page.blocks.append(Block(id=row["id"], page=0, type=row["type"], bbox=[],
             reading_order=row["reading_order"], content=row.get("content", ""), html=html,
             source="native_office", meta=meta, warnings=row.get("warnings", [])))
    doc.pages.append(page)
    return doc


def parse_docx(path: Path) -> Document:
    return _structured_ir(Path(path), structured_parse(path))


def parse_pptx(path: Path) -> Document:
    from pptx import Presentation

    prs = Presentation(str(path))
    doc = Document(path=str(path), sha256=sha256_file(path), doc_type="pptx")
    for i, slide in enumerate(prs.slides, 1):
        page = Page(number=i, kind="native")
        blocks: list[Block] = []
        for shape in slide.shapes:
            if shape.has_text_frame and shape.text_frame.text.strip():
                _emit(blocks, i, "text", shape.text_frame.text.strip(),
                      meta={"shape": shape.name})
            if getattr(shape, "has_table", False) and shape.has_table:
                rows = [[cell.text for cell in row.cells] for row in shape.table.rows]
                _emit(blocks, i, "table", "", html=_table_html(rows))
        page.blocks = blocks
        page.mode = "native"
        doc.pages.append(page)
    return doc


def parse_xlsx(path: Path) -> Document:
    return _structured_ir(Path(path), structured_parse(path))


def parse_text(path: Path) -> Document:
    text = path.read_text(encoding="utf-8", errors="replace")
    doc = Document(path=str(path), sha256=sha256_file(path), doc_type="text")
    page = Page(number=1, kind="native", mode="native")
    page.blocks.append(Block(id="p1b0", page=1, type="text", bbox=[0, 0, 1000, 1000],
                             reading_order=0, content=text, source="native_text",
                             confidence=1.0))
    doc.pages.append(page)
    return doc
