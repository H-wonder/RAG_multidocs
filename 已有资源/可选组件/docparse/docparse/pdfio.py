"""PDF probing, native text extraction, and page rendering (PyMuPDF)."""

from __future__ import annotations

import hashlib
import html as _html
import math
import re
from dataclasses import dataclass
from pathlib import Path

try:
    import pymupdf as fitz          # PyMuPDF >= 1.24
except ImportError:                 # older releases
    import fitz
from PIL import Image

from .ir import Block
from .quality import warning

BAD_CHARS = set("�\u0000")
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f�]")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class PageProbe:
    number: int
    width: float
    height: float
    chars: int
    bad_ratio: float
    image_coverage: float
    kind: str = "native"          # native | scanned | hybrid | blank
    quality: float = 1.0


def probe_page(page: fitz.Page, number: int) -> PageProbe:
    text = page.get_text("text") or ""
    stripped = text.strip()
    chars = len(stripped)
    bad = len(_CTRL_RE.findall(stripped))
    bad_ratio = bad / max(1, chars)

    area = max(1.0, page.rect.width * page.rect.height)
    covered = 0.0
    try:
        for info in page.get_image_info():
            r = fitz.Rect(info["bbox"]) & page.rect
            if not r.is_empty:
                covered += r.get_area()
    except Exception:
        pass
    coverage = min(1.0, covered / area)

    if chars < 10:
        kind = "blank" if coverage < 0.05 else "scanned"
    elif bad_ratio > 0.05:
        kind = "scanned"                 # broken text layer → OCR is safer
    elif coverage > 0.6 and chars < 50:
        kind = "scanned"
    elif coverage > 0.3:
        kind = "hybrid"
    else:
        kind = "native"

    quality = 1.0
    if kind in {"scanned", "blank"}:
        quality = 0.0
    else:
        quality = max(0.0, 1.0 - bad_ratio * 10)
        if chars < 80 and coverage < 0.3:
            quality *= 0.6
    return PageProbe(number, page.rect.width, page.rect.height,
                     chars, bad_ratio, coverage, kind, quality)


def probe_pdf(path: Path) -> list[PageProbe]:
    with fitz.open(path) as doc:
        return [probe_page(doc[i], i + 1) for i in range(len(doc))]


# ── native extraction ────────────────────────────────────────────────────────

_MAX_EDGE_OVERFLOW = 0.01  # Fraction of the visible page on each axis.


def _measured_rect(bbox) -> fitz.Rect | None:
    if (not bbox or len(bbox) != 4
            or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in bbox)):
        return None
    rect = fitz.Rect(bbox)
    return rect if not rect.is_empty else None


def _native_bbox(raw, page_rect, rotation_matrix, spans=()):
    """Return geometry in unrotated points for sorting and rendered-page IR units."""
    rect = _measured_rect(raw)
    display = rect * rotation_matrix if rect is not None else None
    code = None
    if display is None or not page_rect.contains(display):
        # The text block can include an off-page whitespace-only span.
        visible = [_measured_rect(s.get("bbox")) for s in spans if s.get("text", "").strip()]
        if visible and all(r is not None for r in visible):
            rect = fitz.Rect(visible[0])
            for span_rect in visible[1:]:
                rect.include_rect(span_rect)
            display = rect * rotation_matrix
            code = "native_bbox_from_text_spans"

    bbox = []
    if display is not None and not page_rect.is_empty:
        relative = [(display.x0 - page_rect.x0) / page_rect.width,
                    (display.y0 - page_rect.y0) / page_rect.height,
                    (display.x1 - page_rect.x0) / page_rect.width,
                    (display.y1 - page_rect.y0) / page_rect.height]
        if all(-_MAX_EDGE_OVERFLOW <= v <= 1 + _MAX_EDGE_OVERFLOW for v in relative):
            bounded = [max(0.0, min(1.0, v)) for v in relative]
            if bounded[0] < bounded[2] and bounded[1] < bounded[3]:
                rounded = [round(v * 1000, 1) for v in bounded]
                # Keep precision rather than emitting a zero-area box for a tiny region.
                bbox = (rounded if rounded[0] < rounded[2] and rounded[1] < rounded[3]
                        else [v * 1000 for v in bounded])
                if bounded != relative:
                    code = "native_bbox_clipped"
    if not bbox:
        code = "native_bbox_unavailable"
        rect = None
    if not code:
        return rect, bbox, []
    messages = {
        "native_bbox_from_text_spans": "Recomputed an invalid block box from measured nonblank text spans.",
        "native_bbox_clipped": "Clipped a small native box overflow to the visible page.",
        "native_bbox_unavailable": "Retained native content without a reliable position.",
    }
    raw_points = [
        v if isinstance(v, (int, float)) and math.isfinite(v) else None for v in (raw or ())
    ]
    return rect, bbox, [warning(
        code, "medium", messages[code], raw_bbox_pt=raw_points,
        page_size_pt=[page_rect.width, page_rect.height], bbox=bbox,
        raw_coordinate_space="unrotated_page_top_left",
    )]


def _table_rows(table, rows):
    """Remove a false full-height cell that encloses independently detected rows."""
    table_bbox = fitz.Rect(table.bbox)
    row_cells = [[_measured_rect(cell) for cell in row.cells] for row in table.rows]
    if len(rows) != len(row_cells) or any(len(row) != len(cells)
                                         for row, cells in zip(rows, row_cells)):
        return rows, [table_bbox], []
    removed = set()
    for row_index, cells in enumerate(row_cells):
        for column, cell in enumerate(cells):
            if cell is None or cell.y0 != table_bbox.y0 or cell.y1 != table_bbox.y1:
                continue
            nested_rows = sum(
                any(other is not None and other != cell and cell.contains(other) for other in others)
                for index, others in enumerate(row_cells) if index != row_index
            )
            if nested_rows >= 2:
                removed.add((row_index, column))
    if not removed:
        return rows, [table_bbox], []
    retained = []
    filtered = []
    for row_index, (row, cells) in enumerate(zip(rows, row_cells)):
        kept = [None if (row_index, column) in removed else value
                for column, value in enumerate(row)]
        if any(kept):
            filtered.append(kept)
            retained.extend(cell for column, cell in enumerate(cells)
                            if cell is not None and column < len(kept) and kept[column])
    return filtered, retained, [warning(
        "native_table_enclosing_cell", "medium",
        "Ignored overlapping full-height cells; kept detected rows and uncovered native text.",
        removed_cells=len(removed),
    )]


def _rows_to_html(rows: list[list[str | None]]) -> str:
    out = ["<table>"]
    for row in rows:
        out.append("<tr>" + "".join(
            f"<td>{_html.escape(c or '').strip()}</td>" for c in row) + "</tr>")
    out.append("</table>")
    return "\n".join(out)


def _two_column(blocks: list[dict], width: float) -> bool:
    if len(blocks) < 6:
        return False
    left = sum(1 for b in blocks if (b["bbox"][0] + b["bbox"][2]) / 2 < width * 0.5)
    right = len(blocks) - left
    if left < len(blocks) * 0.3 or right < len(blocks) * 0.3:
        return False
    # A block straddling the gutter means single-column flow.
    straddle = sum(1 for b in blocks
                   if b["bbox"][0] < width * 0.4 and b["bbox"][2] > width * 0.6)
    return straddle <= max(1, len(blocks) * 0.1)


def _reading_order(blocks: list[dict], width: float) -> list[dict]:
    if _two_column(blocks, width):
        left = sorted([b for b in blocks if (b["bbox"][0] + b["bbox"][2]) / 2 < width * 0.5],
                      key=lambda b: (b["bbox"][1], b["bbox"][0]))
        right = sorted([b for b in blocks if (b["bbox"][0] + b["bbox"][2]) / 2 >= width * 0.5],
                       key=lambda b: (b["bbox"][1], b["bbox"][0]))
        return left + right
    return sorted(blocks, key=lambda b: (b["bbox"][1], b["bbox"][0]))


def extract_native(page: fitz.Page, page_no: int) -> list[Block]:
    """Vector text + ruled tables + image regions, in reading order."""
    page_rect = page.rect
    rotation_matrix = page.rotation_matrix
    unrotated_width = (page_rect * page.derotation_matrix).width
    raw_blocks: list[dict] = []
    sizes: list[float] = []

    data = page.get_text("dict")
    for b in data.get("blocks", []):
        if b.get("type") != 0:
            continue
        spans = [s for line in b.get("lines", []) for s in line.get("spans", [])]
        text = "".join(s.get("text", "") for s in spans).strip()
        if not text:
            continue
        size = max((s.get("size", 0) for s in spans), default=0)
        sizes.append(size)
        rect, bbox, warnings = _native_bbox(b.get("bbox"), page_rect, rotation_matrix, spans)
        raw_blocks.append({"bbox": rect, "normalized": bbox, "warnings": warnings,
                           "text": text, "size": size})

    table_rects: list[fitz.Rect] = []
    retained_cells: list[fitz.Rect] = []
    table_blocks: list[dict] = []
    if hasattr(page, "find_tables"):
        rotation = page.rotation
        try:
            # Extract table and text boxes in the same unrotated page frame.
            if rotation:
                page.set_rotation(0)
            for table in page.find_tables().tables:
                # Table.extract() can reorder punctuation below digit glyphs
                # (e.g. "3.0" becomes "30\n."). Use the measured cell rectangle
                # and native text stream, never guess or repair numeric strings.
                rows = [[page.get_textbox(cell).strip() if cell else None
                         for cell in row.cells] for row in table.rows]
                rows, coverage, table_warnings = _table_rows(table, rows)
                if not rows:
                    continue
                rect, bbox, warnings = _native_bbox(table.bbox, page_rect, rotation_matrix)
                if rect is not None:
                    (retained_cells if table_warnings else table_rects).extend(coverage)
                table_blocks.append({
                    "bbox": rect,
                    "normalized": bbox,
                    "warnings": warnings + table_warnings,
                    "html": _rows_to_html(rows),
                })
        except Exception:
            pass
        finally:
            if rotation:
                page.set_rotation(rotation)

    # Drop text fragments that belong to a detected table.
    kept = []
    for b in raw_blocks:
        r = b["bbox"]
        if r is not None:
            if any((r & tr).get_area() > r.get_area() * 0.5 for tr in table_rects):
                continue
            if any(cell.contains(r) for cell in retained_cells):
                continue
        kept.append(b)

    # Lower median: with only two sizes [body, title] the upper one would make
    # the title indistinguishable from body text.
    median = sorted(sizes)[(len(sizes) - 1) // 2] if sizes else 10.0
    candidates = kept + table_blocks
    ordered = _reading_order([b for b in candidates if b["bbox"] is not None], unrotated_width)
    ordered.extend(b for b in candidates if b["bbox"] is None)

    blocks: list[Block] = []
    for i, b in enumerate(ordered):
        bbox = b["normalized"]
        if "html" in b:
            blocks.append(Block(id=f"p{page_no}b{i}", page=page_no, type="table",
                                bbox=bbox, reading_order=i, content=b["html"],
                                html=b["html"], source="native_text", confidence=0.98,
                                warnings=b["warnings"]))
        else:
            text = b["text"]
            btype = "text"
            if (b["size"] > median * 1.25 and len(text) < 120) or (
                    len(text) < 60 and re.match(r"^\d{1,2}[、.]\s*[^\d]", text)):
                btype = "title"
            blocks.append(Block(id=f"p{page_no}b{i}", page=page_no, type=btype,
                                bbox=bbox, reading_order=i, content=text,
                                source="native_text", confidence=0.99,
                                warnings=b["warnings"]))

    # Image regions (as figures), appended after text in reading order.
    try:
        for info in page.get_image_info():
            raw = _measured_rect(info.get("bbox"))
            if raw is None:
                continue
            r = (raw * rotation_matrix) & page_rect
            if r.is_empty or r.get_area() < page_rect.get_area() * 0.02:
                continue
            _, bbox, _ = _native_bbox(list(r), page_rect, fitz.Matrix(1, 1))
            i = len(blocks)
            blocks.append(Block(id=f"p{page_no}b{i}", page=page_no, type="figure",
                                bbox=bbox, reading_order=i,
                                content="", source="native_text", confidence=0.9,
                                meta={"image": True}))
    except Exception:
        pass

    return blocks


def render_page(page: fitz.Page, dpi: int) -> Image.Image:
    pix = page.get_pixmap(dpi=dpi, alpha=False)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def render_pdf_page(path: Path, page_index: int, dpi: int) -> Image.Image:
    with fitz.open(path) as doc:
        return render_page(doc[page_index], dpi)
