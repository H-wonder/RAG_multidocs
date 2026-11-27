"""Structured Office parsing shared by the existing engine and thin plugins.

P1 additions extend native Office extraction with source locations, original
cell values/formula caches, merged anchors and optional existing Parser OCR.
No RAG, Index or platform dependencies. No formula execution.
"""
import datetime
import hashlib
import re
import tempfile
from pathlib import Path


def parse_docx(request):
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    document = docx.Document(request["path"])
    options = request["options"]
    blocks, warnings, sections = [], [], []
    paragraph_no, table_no, image_no = 0, 0, 0

    def emit(kind, content, location, **extra):
        b = {"id": f"b{len(blocks)}", "type": kind, "content": content, "page": 0, "bbox": [], "reading_order": len(blocks), "section_path": " > ".join(sections), "origin": "native_extraction", "ocr_mode": "native", "meta": {"location": location}, **extra}
        blocks.append(b)
        return b

    for child in document.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            paragraph_no += 1
            paragraph = Paragraph(child, document)
            text = paragraph.text.strip()
            style = paragraph.style.name if paragraph.style else ""
            heading = re.search(r"(?:Heading|标题)\s*(\d+)", style, re.IGNORECASE)
            numbered_level = None
            if text and len(text) <= 60 and not re.search(r"[。！？；]", text):
                if re.match(r"^[一二三四五六七八九十百]+、\S", text):
                    numbered_level = 1
                elif re.match(r"^[（(][一二三四五六七八九十百]+[）)]\S", text):
                    numbered_level = 2
            if heading or numbered_level:
                level = int(heading[1]) if heading else numbered_level
                sections[:] = sections[:level-1] + [text]
                emit("title", text, {"paragraph": paragraph_no}, level=level)
            elif text:
                if style.lower() == "caption" and blocks and blocks[-1]["type"] == "figure":
                    blocks[-1]["caption"] = text
                emit("text", text, {"paragraph": paragraph_no})
            for blip in child.xpath('.//*[local-name()="blip"]'):
                relation = blip.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed')
                if not relation or relation not in document.part.related_parts:
                    continue
                part = document.part.related_parts[relation]
                image_no += 1
                sha = hashlib.sha256(part.blob).hexdigest()
                location = {"paragraph": paragraph_no, "image": image_no, "archive_part": str(part.partname), "sha256": sha}
                figure = emit("figure", "", location)
                if options.get("ocr_images"):
                    try:
                        from .config import ParseOptions
                        from .pipeline import Parser
                        with tempfile.TemporaryDirectory(prefix="word_image_") as temp:
                            image = Path(temp) / Path(str(part.partname)).name
                            image.write_bytes(part.blob)
                            ir = Parser(ParseOptions(mode="page")).parse(image).to_dict()
                            figure["ocr_text"] = "\n".join(str(b.get("content") or b.get("html") or "") for p in ir["pages"] for b in p["blocks"])
                            if not figure["ocr_text"].strip():
                                raise ValueError("Embedded image OCR empty")
                            warnings.extend({**item, "image": image_no} for item in ir.get("warnings", []))
                            figure["origin"] = "ocr_result"
                            figure["ocr_mode"] = "page"
                    except (OSError, ValueError, RuntimeError, ImportError, StopIteration) as exc:
                        warnings.append({"code": "IMAGE_OCR_FAILED", "severity": "warning", "message": type(exc).__name__, "image": image_no})
                else:
                    warnings.append({"code": "IMAGE_OCR_NOT_REQUESTED", "severity": "info", "message": "Image retained by archive part and SHA; caption only", "image": image_no})
        elif tag == "tbl":
            table_no += 1
            table = Table(child, document)
            anchors, rows, mapping = {}, [], []
            for row_no, row in enumerate(table.rows, 1):
                values = []
                for col_no, cell in enumerate(row.cells, 1):
                    anchor = anchors.setdefault(cell._tc, f"R{row_no}C{col_no}")
                    values.append(cell.text.strip().replace("\n", " ").replace("|", "¦"))
                    mapping.append({"address": f"R{row_no}C{col_no}", "anchor": anchor})
                rows.append(values)
            block = emit("table", "\n".join(" | ".join(r) for r in rows), {"table": table_no, "cell_range": f"R1C1:R{len(rows)}C{max(map(len, rows), default=0)}"})
            block["meta"]["table_data"] = {"rows": rows, "header_rows": int(options.get("table_header_rows", {}).get(str(table_no),options.get("header_rows", 1))), "cells": mapping}
    return {"title": document.core_properties.title or Path(request["path"]).stem, "content": "\n\n".join(b.get("content", "") or b.get("ocr_text", "") or b.get("caption", "") for b in blocks), "blocks": blocks, "metadata": {"warnings": warnings, "needs_review": any(w["severity"] == "warning" for w in warnings), "origin": "native_extraction"}}


def scalar(value):
    return value.isoformat() if isinstance(value, (datetime.datetime, datetime.date, datetime.time)) else value


def display(value, fmt):
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (datetime.datetime, datetime.date, datetime.time)):
        return value.isoformat()
    if isinstance(value, (float, int)):
        decimals = re.search(r"0\.(0+)", fmt)
        digits = len(decimals[1]) if decimals else 0
        if "%" in fmt:
            return f"{value * 100:.{digits}f}%"
        if any(c in fmt for c in ["$", "¥", "￥"]):
            currency = next(c for c in fmt if c in "$¥￥")
            return f"{currency}{value:.{digits}f}"
        if decimals:
            return f"{value:.{digits}f}"
    return str(value)


def parse_xlsx(request):
    import openpyxl
    from openpyxl.utils import get_column_letter

    options = request["options"]
    wb = openpyxl.load_workbook(request["path"], data_only=False, keep_vba=False, keep_links=False)
    cached = openpyxl.load_workbook(request["path"], data_only=True, keep_vba=False, keep_links=False)
    blocks, warnings = [], []
    max_cells = int(options.get("max_cells", 200000))
    try:
        for sheet in wb.worksheets:
            if sheet.max_row * sheet.max_column > max_cells:
                raise ValueError("Worksheet exceeds explicit max_cells")
            occupied = [(cell.row, cell.column) for row in sheet for cell in row if cell.value is not None]
            if not occupied:
                continue
            min_col, max_col = min(c for _, c in occupied), max(c for _, c in occupied)
            merged = [str(r) for r in sheet.merged_cells.ranges]
            rows, cells, row_numbers = [], [], []
            for row_no in range(min(r for r, _ in occupied), max(r for r, _ in occupied) + 1):
                values = []
                for col in range(min_col, max_col + 1):
                    cell = sheet.cell(row_no, col)
                    anchor = cell.coordinate
                    for region in sheet.merged_cells.ranges:
                        if cell.coordinate in region:
                            anchor = sheet.cell(region.min_row, region.min_col).coordinate
                            break
                    original = sheet[anchor]
                    value = original.value
                    formula = str(value) if original.data_type == "f" else None
                    cached_value = cached[sheet.title][anchor].value if formula else None
                    shown = display(cached_value if formula else value, original.number_format)
                    if formula:
                        shown = f"{formula} [cached={shown or 'UNAVAILABLE'}]"
                        if cached_value is None:
                            warnings.append({"code": "FORMULA_UNCACHED", "severity": "warning", "message": "Formula not evaluated", "worksheet": sheet.title, "cell": anchor})
                    values.append(shown.replace("\n", " ").replace("|", "¦"))
                    if value is not None:
                        cells.append({"address": cell.coordinate, "anchor": anchor, "value": scalar(value), "data_type": original.data_type, "number_format": original.number_format, "formula": formula, "cached_value": scalar(cached_value), "display": shown})
                rows.append(values)
                row_numbers.append(row_no)
            first, last = row_numbers[0], row_numbers[-1]
            cell_range = f"{get_column_letter(min_col)}{first}:{get_column_letter(max_col)}{last}"
            location = {"worksheet": sheet.title, "cell_range": cell_range, "row_numbers": row_numbers, "column_start": min_col}
            meta = {"location": location, "table_data": {"rows": rows, "header_rows": int(options.get("sheet_header_rows", {}).get(sheet.title,options.get("header_rows", 1))), "cells": cells, "merged_ranges": merged}}
            blocks.append({"id": f"sheet{len(blocks)+1}", "type": "table", "page": 0, "bbox": [], "reading_order": len(blocks), "section_path": f"Sheet: {sheet.title} > {cell_range}", "content": "\n".join(" | ".join(row) for row in rows), "meta": meta, "origin": "native_extraction", "ocr_mode": "native", "needs_review": any(w.get("worksheet") == sheet.title for w in warnings), "warnings": [w for w in warnings if w.get("worksheet") == sheet.title]})
        return {"title": "Workbook", "content": "\n\n".join(b["section_path"] + "\n" + b["content"] for b in blocks), "blocks": blocks, "metadata": {"warnings": warnings, "needs_review": bool(warnings), "origin": "native_extraction", "sheets": wb.sheetnames}}
    finally:
        wb.close()
        cached.close()

