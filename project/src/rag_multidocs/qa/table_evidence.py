"""Deterministic views of flat pipe tables. Raw text remains the citation source.

Ambiguous/merged/ragged structures are deliberately not inferred. No OCR numeric
repair is performed here; suspicious values are retained and marked for review.
"""

from __future__ import annotations

import re
from datetime import date
from ..ingestion.tables import panel_width


def cell_with_unit(label, value):
    """Use only a unit explicitly declared in this cell's column header."""
    match = re.search(r'[（(]\s*(万元|元|CNY|USD|kg|千克|米|箱|件|台|张|个|天|小时|%|％)\s*[）)]$', str(label), re.I)
    if match and re.fullmatch(r'-?\d+(?:\.\d+)?', str(value).strip()):
        return str(value).strip() + match[1]
    return str(value)


def label_in(label, text, *, field_query=False):
    values = [label, str(label).split(" > ")[-1]]
    if cell_with_unit(label, '1') != '1':
        values.append(re.sub(r'[（(][^（）()]+[）)]$', '', str(label)).strip())
    if values[-1] in {"成绩", "得分", "分数"}:
        values.extend(["成绩", "得分", "分数", "多少分"])
    if field_query and str(values[-1]).endswith(("时间", "日期", "时段")) and re.search(r"时间|日期|何时|什么时候|几点|\d{1,2}[:：]\d{2}", str(text)):
        return True
    return any(re.sub(r"\s+", "", str(v)).casefold() in re.sub(r"\s+", "", str(text)).casefold() for v in values if v)


_CATEGORY_HEADER = re.compile(r"(?:级别|等级|类别|状态|类型|level|category|status|type)$", re.I)


def category_values(fields):
    return {value for label, value in fields.items() if _CATEGORY_HEADER.search(label)}


def _structured_rows(hit):
    data = hit.get('table_data', hit.get('metadata', {}).get('table_data', {}))
    values = data.get('rows')
    if not isinstance(values, list) or not values or not all(isinstance(row, list) for row in values):
        return None
    rows = [[str(cell) for cell in row] for row in values]
    headers = hit.get('table_headers', hit.get('metadata', {}).get('table_headers', rows[0]))
    if any(len(row) != len(headers) for row in rows):
        return None
    raw = [' | '.join(row) for row in rows]
    if '\n'.join(raw).strip() != hit['text'].strip():
        return None
    result, cursor = [], 0
    for cells, quote in zip(rows, raw):
        start = hit['text'].find(quote, cursor)
        span = [start, start + len(quote)] if start >= 0 else []
        cursor = span[1] if span else cursor
        result.append((cells, quote, span))
    return result


def categorical_rows(hit):
    """Read real category cells with aligned headers, including empty side cells."""
    if hit.get("content_type") != "table":
        return {}
    meta = hit.get("metadata", {})
    headers = hit.get("table_headers") or meta.get("table_headers") or []
    positions = [i for i, header in enumerate(headers) if _CATEGORY_HEADER.search(header)]
    if not positions:
        return {}
    data = hit.get("table_data") or meta.get("table_data") or {}
    structured = _structured_rows(hit)
    lines = ([(cells, quote) for cells, quote, _ in structured[int(data.get("header_rows", 1)):]]
             if structured else [(None, line) for line in hit["text"].splitlines()[int(data.get("header_rows", 1)):]])
    result = {}
    for known, line in lines:
        cells = known if known is not None else [cell.strip() for cell in line.split("|")]
        if len(cells) == len(headers) + 2 and not cells[0] and not cells[-1]:
            cells = cells[1:-1]
        if len(cells) == len(headers) and any(cells[i] for i in positions):
            result[line] = {header: cell for header, cell in zip(headers, cells) if header and cell}
    return result


def table_view(hit, question=""):
    text = hit["text"]
    meta = hit.get("metadata", {})
    mapped_headers = hit.get("table_headers", meta.get("table_headers"))
    rows = []
    offset = 0
    for line in text.splitlines(keepends=True):
        raw = line.rstrip("\r\n")
        if "|" in raw:
            # Core serialization has no Markdown border pipes. Stripping them
            # loses real empty cells at either end of a row.
            cells = [c.strip() for c in (raw if mapped_headers else raw.strip().strip("|")).split("|")]
            if not all(re.fullmatch(r":?-+:?", c or " ") for c in cells):
                rows.append((cells, raw, [offset, offset + len(raw)]))
        offset += len(line)
    rows = _structured_rows(hit) or rows
    if len(rows) < 2:
        return {"status": "unsupported", "rows": []}
    data = hit.get("table_data", meta.get("table_data", {}))
    header_count = int(data.get("header_rows", 1)) if mapped_headers else 1
    if not 1 <= header_count < len(rows):
        return {"status": "unsupported", "rows": []}
    headers = mapped_headers or rows[0][0]
    data_rows = rows[header_count:]
    if mapped_headers:
        # OCR sometimes emits extra empty cells after the final column. They
        # carry no value; retain the original quote but align the known columns.
        data_rows = [(cells[:len(headers)] if len(cells) > len(headers)
                      and not any(cells[len(headers):]) else cells, raw, span)
                     for cells, raw, span in data_rows]
    width = panel_width(headers)
    if width < len(headers) and all(len(r[0]) == len(headers) for r in data_rows):
        data_rows = [(cells[start:start + width], raw, span)
                     for cells, raw, span in data_rows
                     for start in range(0, len(headers), width)
                     if cells[start] and any(cells[start + 1:start + width])]
        headers = headers[:width]
    nonempty_headers = [c for c in headers if c]
    if (len(nonempty_headers) < 2 or (not mapped_headers and any(not c for c in headers))
            or len(set(nonempty_headers)) != len(nonempty_headers)
            or any(len(r[0]) != len(headers) or (not mapped_headers and any(not c for c in r[0])) for r in data_rows)):
        return {"status": "ambiguous", "rows": []}
    anchor = 0
    if question and data_rows:
        scores = {col: sum(any(c.isalpha() for c in row[0][col]) and label_in(row[0][col], question)
                           for row in data_rows)
                  for col in range(len(headers))
                  if len({row[0][col] for row in data_rows}) == len(data_rows)}
        if scores and max(scores.values()) > 0:
            anchor = max(scores, key=scores.get)
    if len({r[0][anchor] for r in data_rows}) != len(data_rows):
        return {"status": "ambiguous", "rows": []}
    selected = [r for r in data_rows if label_in(r[0][anchor], question)]
    selected = selected or data_rows
    result = []
    for cells, raw, span in selected:
        issues = ["UNRESOLVED_NUMERIC_FORMAT"] if any(
            re.search(r"\d\s+\d.*[%％]|[%％]\s*\.", c) for c in cells[1:]
        ) else []
        result.append({
            "row_label": cells[anchor],
            "statements": [f"{label}的{cells[anchor]}：{value}" for j, (label, value) in enumerate(zip(headers, cells)) if j != anchor and label and value],
            "cells": {label:value for j, (label,value) in enumerate(zip(headers, cells)) if j != anchor and label and value},
            "empty_cells": {label: value for j, (label, value) in enumerate(zip(headers, cells)) if j != anchor and label and value == ''},
            "supporting_quote": raw,
            "raw_span": span,
            "warnings": issues,
        })
    return {
        "status": "mapped_flat_table",
        "kind": "derived_view_not_original_quote",
        "source_chunk_id": hit["chunk_id"],
        "header": {"text": rows[0][1], "raw_span": rows[0][2]},
        "rows": result,
    }


_TIME = r"(?:上午|下午|晚上|凌晨)?\s*\d{1,2}[:：]\d{2}(?:[:：]\d{2})?"


def temporal_parts(text):
    """Canonical date/time tokens; leave unrelated numbers and units intact."""
    tokens = set()
    text = re.sub('[‐‑‒−]', '-', str(text))
    def day(match):
        try:
            value = date(*map(int, match.groups())).isoformat()
        except ValueError:
            return match[0]
        tokens.add('date:' + value)
        return ' '
    text = re.sub(r'(?<!\d)(\d{4})\s*(?:年|[-/.])\s*(\d{1,2})\s*(?:月|[-/.])\s*(\d{1,2})\s*(?:日|号)?', day, str(text))
    def clock(value, inherited=''):
        part = re.fullmatch(r'(上午|下午|晚上|凌晨)?\s*(\d{1,2})[:：](\d{2})(?:[:：](\d{2}))?', value.strip())
        if not part:
            return None
        period = part[1] or inherited
        hour, minute, second = int(part[2]), int(part[3]), int(part[4] or 0)
        if hour > 23 or minute > 59 or second > 59:
            return None
        if period in ('下午', '晚上') and hour < 12:
            hour += 12
        if period in ('上午', '凌晨') and hour == 12:
            hour = 0
        return f'{hour:02}:{minute:02}:{second:02}'
    def interval(match):
        first, last = match.groups()
        period = re.match(r'上午|下午|晚上|凌晨', first.strip())
        a, b = clock(first), clock(last, period[0] if period else '')
        if not a or not b:
            return match[0]
        tokens.update(('time:' + a, 'time:' + b, 'range:' + a + '..' + b))
        return ' '
    text = re.sub('(' + _TIME + r')\s*(?:-|—|–|－|~|～|至|到)\s*(' + _TIME + ')', interval, text)
    def single(match):
        value = clock(match[0])
        if not value:
            return match[0]
        tokens.add('time:' + value)
        return ' '
    return tokens, re.sub(_TIME, single, text)


def temporal_equal(a, b):
    aa, ar = temporal_parts(a)
    bb, br = temporal_parts(b)
    return bool(aa) and aa == bb and re.sub(r'[\W_]+', '', ar) == re.sub(r'[\W_]+', '', br)
