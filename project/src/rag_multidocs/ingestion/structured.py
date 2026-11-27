"""Content-aware chunking over public Document/Block data, never plugin names."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, replace

from ..model import Chunk, chunk_id, normalize_hit, normalize_warning
from .tables import table_groups
from .chunking import (
    _TableText,
    _SECTION_RE,
    _heading_level,
    _looks_like_heading,
    chunk_document,
    estimate_tokens,
    pack_paragraphs,
)


@dataclass
class StructuredOptions:
    max_tokens: int = 500
    min_tokens: int = 100
    overlap_tokens: int = 80
    max_table_rows: int = 100000
    table_row_sep: str = " | "
    policy_version: str = "structured-v2"
    semantic_retrieval: bool = False

    def __post_init__(self):
        if (
            not 32 <= self.max_tokens <= 4000
            or not 0 <= self.overlap_tokens < self.max_tokens
            or not 0 <= self.min_tokens <= self.max_tokens
        ):
            raise ValueError("invalid chunk budget")


def public_blocks(document):
    inherited = {
        "ocr_mode": document.metadata.get("ocr_mode", ""),
        "needs_review": document.metadata.get("needs_review", False),
        "warnings": document.metadata.get("warnings", []),
        "origin": document.metadata.get("origin", "connector"),
    }
    if document.segments:
        result = [{**inherited, **b} for b in document.segments]
    else:
        ir = document.metadata.get("docparse_ir") or document.metadata.get("ir") or {}
        result = []
        for page in ir.get("pages", []):
            if page.get("mode") == "failed":
                continue
            blocks = page.get("blocks", [])
            granular = [b for b in blocks if b.get("type") not in {"page", "page_markdown"}]
            for b in granular or blocks:
                result.append(
                    {
                        **inherited,
                        **b,
                        "page": page.get("number", 0),
                        "ocr_mode": page.get("mode", ""),
                        "warnings": inherited["warnings"]
                        + page.get("warnings", [])
                        + b.get("warnings", []),
                    }
                )
    # Whole-page OCR may contain prose and complete HTML tables. Parse the
    # table boundary before token splitting, so tags/cells cannot be cut in half.
    expanded = []
    for block in result:
        text = str(block.get("content", block.get("text", "")))
        if block.get("type") in {"page", "page_text", "page_markdown"} and re.search(r"<table\b", text, re.I):
            for part, piece in enumerate(re.split(r"(<table\b[^>]*>[\s\S]*?</table\s*>)", text, flags=re.I)):
                if piece.strip():
                    is_table = bool(re.match(r"<table\b", piece, re.I))
                    expanded.append({**block, "id": f"{block.get('id', 'page')}_{part}",
                                     "type": "table" if is_table else "text",
                                     "content": piece, "html": piece if is_table else ""})
        else:
            expanded.append(block)
    result = expanded
    for i, block in enumerate(result):
        meta = block.get("meta", block.get("metadata", {}))
        for key in ["section_path", "level", "caption", "ocr_text", "origin", "ocr_mode", "needs_review"]:
            if key in meta:
                block[key] = meta[key]
        block["reading_order"] = int(block.get("reading_order", i) or 0)
        block["page"] = int(block.get("page", 0) or 0)
        # Validate real coordinates without manufacturing a full-page box.
        block["bbox"] = normalize_hit(block)["bbox"]
        block["warnings"] = [normalize_warning(w) for w in block.get("warnings", [])]
        mode = block.get("ocr_mode", "")
        if mode == "reference_transcription":
            block["origin"] = "reference_transcription"
        elif mode in {"page", "region"}:
            block["origin"] = "ocr_result"
        elif mode == "native":
            block["origin"] = "native_extraction"
        else:
            block["origin"] = block.get("origin", block.get("source", "connector"))
    return sorted(result, key=lambda b: (b["page"], b["reading_order"]))


def _table(text):
    if "<tr" in text.lower():
        parser = _TableText()
        parser.feed(text)
        return [" | ".join(row) for row in parser.rows]
    return [
        line for line in text.splitlines() if line.strip() and not re.fullmatch(r"[| :\-]+", line)
    ]


def structured_chunks(document, options=None, token_count=estimate_tokens, model_limit=None):
    options = options or StructuredOptions()
    limit = min(options.max_tokens, model_limit or options.max_tokens)
    if limit < 32:
        raise ValueError("embedding window too small")
    options = (
        replace(
            options,
            max_tokens=limit,
            min_tokens=min(getattr(options, "min_tokens", 0), limit),
            overlap_tokens=min(options.overlap_tokens, limit // 4),
        )
        if isinstance(options, StructuredOptions)
        else StructuredOptions(
            max_tokens=limit, min_tokens=0, overlap_tokens=min(options.overlap_tokens, limit // 4)
        )
    )
    blocks = public_blocks(document)
    if not blocks:
        chunks = chunk_document(document, options)
        for c in chunks:
            c.bbox = ""  # absent means absent, never fake a full-page rectangle
        return _fit(chunks, limit, token_count, options)
    seen_pages = defaultdict(set)
    for b in blocks:
        text = re.sub(r"\s+", " ", b.get("content", b.get("text", ""))).strip()
        box = json.loads(b["bbox"]) if b.get("bbox") else None
        boundary = isinstance(box, list) and len(box) == 4 and (box[1] <= 80 or box[3] >= 920)
        if b.get("type") in {"header", "footer"} or (boundary and len(text) < 120):
            seen_pages[text].add(b.get("page", 0))
    headers = {
        s
        for s, pages in seen_pages.items()
        if len(pages) >= max(3, (len({b["page"] for b in blocks}) + 1) // 2)
    }
    out = []
    sections = []
    dedup = set()

    def emit(text, kind, b, ids=None, metadata=None):
        extra = {**b.get("meta", b.get("metadata", {})), **(metadata or {})}
        section = b.get("section_path") or " > ".join(sections)
        key = (b.get("page", 0), kind, section, json.dumps(extra.get("location", {}), sort_keys=True), re.sub(r"\s+", "", text))
        if not text.strip() or key in dedup:
            return
        dedup.add(key)
        bbox = b.get("bbox") or ""
        if not isinstance(bbox, str):
            bbox = json.dumps(bbox)
        out.append(
            Chunk(
                chunk_id(document.doc_id, kind, len(out)),
                document.doc_id,
                text,
                document.source,
                len(out),
                kind,
                section,
                int(b.get("page", 0)),
                int(b.get("page_end", b.get("page", 0))),
                bbox,
                ids or [str(b.get("id", f"b{len(out)}"))],
                {
                    **extra,
                    "ocr_mode": b.get("ocr_mode", ""),
                    "needs_review": bool(b.get("needs_review", False) or extra.get("needs_review", False)),
                    "warnings": list(b.get("warnings", [])) + list(extra.get("warnings", [])),
                    "origin": b.get("origin", b.get("source", "connector")),
                    "reading_order": b.get("reading_order", 0),
                },
            )
        )

    for i, b in enumerate(blocks):
        kind = b.get("type", "text")
        text = str(b.get("content", b.get("text", ""))).strip()
        repeated = re.sub(r"\s+", " ", text).strip() in headers
        if (kind in {"header", "footer"} or repeated) and not (
            kind == "title" and _SECTION_RE.match(text)
        ):
            continue
        if kind == "title":
            box = json.loads(b["bbox"]) if b.get("bbox") else []
            major = bool(re.match(r"^[一二三四五六七八九十]+[、．.]", text))
            if b.get("level") or major or _looks_like_heading(text, box):
                level = int(b.get("level") or (1 if major else _heading_level(text, box)))
                if (not b.get("level") and not major and _SECTION_RE.match(text)
                        and sections and re.match(r"^[一二三四五六七八九十]+[、．.]", sections[0])):
                    level += 1
                sections = sections[: max(0, level - 1)] + [text]
                continue
            # Unrecognized headings remain searchable text rather than disappearing.
            kind = "text"
        if kind == "table":
            rows = _table(b.get("html") or text)
            if not rows:
                continue
            section = b.get("section_path") or " > ".join(sections)
            budget = limit - token_count(section + "\n") - 4 if section else limit - 4
            for table_text, metadata in table_groups(b, rows, budget, token_count, semantic_retrieval=options.semantic_retrieval):
                emit(table_text, "table", b, metadata=metadata)
        elif kind == "code":
            emit(text, "code", b)
        elif kind == "formula":
            context = []
            for other in blocks[max(0, i - 1) : i] + blocks[i + 1 : i + 2]:
                if other.get("type", "text") == "text" and other.get("page", 0) == b.get("page", 0):
                    candidate = other.get("content", other.get("text", ""))
                    if token_count("\n".join(context + [text, candidate])) <= limit:
                        context.append(candidate)
            emit(
                "\n".join([text] + context),
                "formula",
                b,
                [str(b.get("id", f"b{i}"))]
                + [
                    str(other.get("id", ""))
                    for other in blocks[max(0, i - 1) : i] + blocks[i + 1 : i + 2]
                    if other.get("content", other.get("text", "")) in context
                ],
            )
        elif kind == "figure":
            values = [text, b.get("caption", ""), b.get("ocr_text", ""), b.get("description", "")]
            emit("\n".join(dict.fromkeys(v for v in values if v)), "figure", b)
        else:
            for packed in pack_paragraphs(re.split(r"\n\s*\n", text), options):
                emit(
                    packed,
                    "page_text" if kind in {"page", "page_text", "page_markdown"} else "text",
                    b,
                )
    return _fit(out, limit, token_count, options)


def _fit(chunks, limit, count, options):
    result = []
    for chunk in chunks:
        budget = limit - count(chunk.section_path + "\n") if chunk.section_path else limit
        if budget < 8:
            raise ValueError("section heading exceeds embedding budget")
        if chunk.content_type == "table" and count(chunk.text) > budget:
            rows = _table(chunk.text)
            if not rows or count(rows[0]) > budget:
                raise ValueError("table header exceeds budget")
            parts = []
            group = [rows[0]]
            for row in rows[1:]:
                if count(rows[0] + "\n" + row) > budget:
                    raise ValueError("table row exceeds budget")
                if count("\n".join(group + [row])) > budget and len(group) > 1:
                    parts.append("\n".join(group))
                    group = [rows[0]]
                group.append(row)
            parts.append("\n".join(group))
        elif count(chunk.text) > budget:
            if chunk.content_type in {"formula", "code"}:
                raise ValueError("structured block exceeds model window")
            # Real tokenizer can exceed the heuristic: split and recheck, no hidden truncation.
            pending = [chunk.text]
            parts = []
            while pending:
                text = pending.pop(0)
                if count(text) <= budget:
                    parts.append(text)
                    continue
                middle = len(text) // 2
                if middle == 0:
                    raise ValueError("cannot fit text")
                pending[:0] = [text[:middle], text[middle:]]
        else:
            parts = [chunk.text]
        for text in parts:
            c = Chunk.from_dict(chunk.as_dict())
            c.text = text
            c.index = len(result)
            c.id = chunk_id(c.doc_id, c.content_type, c.index)
            c.metadata["ocr_mode"] = c.metadata.get("ocr_mode", "")
            c.metadata["needs_review"] = bool(c.metadata.get("needs_review", False))
            if (
                result
                and c.content_type == "text"
                and result[-1].content_type == "text"
                and c.page == result[-1].page
                and c.section_path == result[-1].section_path
                and c.metadata.get("location") == result[-1].metadata.get("location")
                and (count(result[-1].text) < options.min_tokens or (
                    options.min_tokens and not re.search(r"[。！？.!?][”’\"']?$", result[-1].text)))
                and count(result[-1].text + "\n" + text) <= budget
            ):
                previous = result[-1]
                previous.text += "\n" + text
                previous.block_ids = list(dict.fromkeys(previous.block_ids + c.block_ids))
                # Multiple disjoint boxes cannot be replaced with fabricated coordinates.
                if previous.bbox != c.bbox:
                    previous.bbox = ""
            else:
                result.append(c)
    return result
