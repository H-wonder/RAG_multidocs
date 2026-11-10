"""Renderers over the IR. Markdown/HTML/text are views; IR is the source of truth."""

from __future__ import annotations

import html as _html
import json
import re
from pathlib import Path

from .ir import Document, Page

FORMATS = ("md", "json", "html", "txt")
_EXT = {"md": ".md", "json": ".json", "html": ".html", "txt": ".txt"}

_HTML_TEMPLATE = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ margin: 0; background: Canvas; color: CanvasText;
         font: 16px/1.7 -apple-system, "Segoe UI", "Noto Sans SC", sans-serif; }}
  main {{ max-width: 920px; margin: 0 auto; padding: 40px 24px 80px; }}
  h1, h2, h3 {{ line-height: 1.3; }}
  table {{ border-collapse: collapse; margin: 1em 0; max-width: 100%; display: block; overflow-x: auto; }}
  td, th {{ border: 1px solid color-mix(in srgb, CanvasText 25%, transparent);
            padding: 6px 10px; vertical-align: top; }}
  pre {{ background: color-mix(in srgb, CanvasText 7%, transparent);
         padding: 12px; overflow-x: auto; border-radius: 6px; }}
  img {{ max-width: 100%; }}
  .meta {{ color: color-mix(in srgb, CanvasText 55%, transparent); font-size: 13px; }}
</style>
</head>
<body><main>
{body}
</main></body>
</html>
"""


def page_markdown(page: Page) -> str:
    synthetic = [b for b in page.blocks if b.type == "page_markdown"]
    if synthetic:
        return synthetic[0].content.strip()

    whole = [b for b in page.blocks if b.type == "page"]
    if whole:
        return whole[0].content.strip()

    parts: list[str] = []
    for b in sorted(page.blocks, key=lambda x: x.reading_order):
        if b.type == "figure":
            continue
        content = (b.html or b.content).strip()
        if content:
            parts.append(content)
    return "\n\n".join(parts)


def document_markdown(doc: Document) -> str:
    pages = [page_markdown(p) for p in doc.pages]
    pages = [p for p in pages if p]
    if len(pages) <= 1:
        return (pages[0] if pages else "") + "\n"
    chunks = []
    for i, text in enumerate(pages, 1):
        chunks.append(f"<!-- page {i} -->\n\n{text}")
    return "\n\n".join(chunks) + "\n"


def document_text(doc: Document) -> str:
    """Plain text: HTML tables become tab-separated rows, all tags stripped."""
    text = document_markdown(doc)
    text = re.sub(r"</t[dh]>\s*", "\t", text)
    text = re.sub(r"</tr>\s*", "\n", text)
    text = re.sub(r"<br\s*/?>", "\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = _html.unescape(text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def document_html(doc: Document, title: str | None = None) -> str:
    """Standalone HTML. Raw <table> blocks are preserved by the markdown renderer."""
    source = document_markdown(doc)
    try:
        import markdown
        body = markdown.markdown(source, extensions=["tables", "fenced_code", "sane_lists"])
    except Exception:
        body = f"<pre>{_html.escape(source)}</pre>"
    return _HTML_TEMPLATE.format(title=_html.escape(title or Path(doc.path).stem), body=body)


def write(doc: Document, outdir: Path, formats: tuple[str, ...] | list[str] = ("md", "json")) -> dict[str, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    stem = Path(doc.path).stem
    paths: dict[str, Path] = {}

    renderers = {
        "md": lambda: document_markdown(doc),
        "json": lambda: json.dumps(doc.to_dict(), ensure_ascii=False, indent=2) + "\n",
        "html": lambda: document_html(doc),
        "txt": lambda: document_text(doc),
    }
    for fmt in formats:
        if fmt not in renderers:
            raise ValueError(f"unsupported format: {fmt}; choose from {', '.join(FORMATS)}")
        path = outdir / f"{stem}{_EXT[fmt]}"
        path.write_text(renderers[fmt](), encoding="utf-8")
        paths[fmt] = path
    return paths
