"""Source-mapped table views and row grouping, independent of file format."""
import copy
import re


def column_name(number):
    result = ""
    while number:
        number, remainder = divmod(number - 1, 26)
        result = chr(65 + remainder) + result
    return result


def panel_width(headers):
    """Recognize only exact, repeated horizontal schemas; no guessed columns."""
    for width in range(2, len(headers) // 2 + 1):
        if (len(headers) % width == 0 and all(headers[:width])
                and len(set(headers[:width])) == width
                and headers == headers[:width] * (len(headers) // width)):
            return width
    return len(headers)


def table_groups(block, lines, budget, count, *, semantic_retrieval=False):
    base = copy.deepcopy(block.get("meta", block.get("metadata", {})))
    table = base.get("table_data", {})
    rows = table.get("rows") or [line.split(" | ") for line in lines]
    headers_count = int(table.get("header_rows", 1))
    # A spreadsheet screenshot can include the UI column letters above the
    # real header. Keep both original rows and use the real field names too.
    if (not table.get("header_rows") and len(rows) > 1 and len(rows[0]) > 2
            and not rows[0][0].strip()
            and rows[0][1:] == [column_name(i) for i in range(1, len(rows[0]))]):
        headers_count = 2
    if not 1 <= headers_count <= len(rows):
        raise ValueError("invalid table header_rows")
    header_rows = rows[:headers_count]
    width = max(map(len, header_rows))
    headers = [" > ".join(dict.fromkeys(str(row[c]).strip() for row in header_rows if c < len(row) and str(row[c]).strip())) for c in range(width)]
    raw_lines = [" | ".join(map(str, row)) for row in rows]
    location = base.get("location", {})
    source_numbers = location.get("row_numbers") or list(range(1, len(rows) + 1))

    def make(indices):
        selected = list(range(headers_count)) + indices
        meta = copy.deepcopy(base)
        picked_rows = [rows[i] for i in selected]
        text = "\n".join(raw_lines[i] for i in selected)
        semantic = []
        panel = panel_width(headers)
        for i in indices:
            row = rows[i]
            for start in range(0, len(headers), panel):
                label = str(row[start]) if start < len(row) else ""
                for column in range(start + 1, min(start + panel, len(row))):
                    value = row[column]
                    if value != "" and label:
                        semantic.append({"row": label, "column": headers[column], "value": str(value), "quote": raw_lines[i], "source_row": source_numbers[i]})
        view = "\n".join(f"{s['row']} / {s['column']}: {s['value']}" for s in semantic)
        meta["retrieval_text"] = view if semantic_retrieval else ""
        meta["table_headers"] = headers
        meta["semantic_cells"] = semantic
        meta["table_data"] = {**table, "rows": picked_rows, "header_rows": headers_count}
        meta["source_table_id"] = str(block.get("id", ""))
        if location:
            loc = copy.deepcopy(location)
            numbers = [source_numbers[i] for i in selected]
            loc["row_numbers"] = numbers
            if "worksheet" in loc:
                first_col = int(loc.get("column_start", 1))
                last_col = first_col + max(map(len, rows)) - 1
                ranges = []
                start = end = numbers[0]
                for number in numbers[1:]:
                    if number == end + 1:
                        end = number
                    else:
                        ranges.append(f"{column_name(first_col)}{start}:{column_name(last_col)}{end}")
                        start = end = number
                ranges.append(f"{column_name(first_col)}{start}:{column_name(last_col)}{end}")
                loc["cell_range"] = ",".join(ranges)
            elif "table" in loc:
                loc["cell_range"] = ",".join(f"R{r}C1:R{r}C{max(map(len, rows))}" for r in numbers)
            meta["location"] = loc
            cells = table.get("cells", [])
            def row_of(cell):
                match = re.match(r"R(\d+)C\d+", cell.get("address", "")) or re.match(r"[A-Z]+(\d+)$", cell.get("address", ""))
                return int(match[1]) if match else None
            meta["table_data"]["cells"] = [c for c in cells if row_of(c) in numbers]
        damaged = any(re.search(r"\d\s+\d.*[%％]|[%％]\s*\.", str(value)) for row in picked_rows for value in row)
        if damaged:
            meta["needs_review"] = True
            meta["warnings"] = list(meta.get("warnings", [])) + [{"code": "UNRESOLVED_NUMERIC_FORMAT", "severity": "warning", "message": "Raw numeric formatting retained; no inferred decimal correction"}]
        return text, meta

    group = []
    for index in range(headers_count, len(rows)):
        text, meta = make(group + [index])
        if count(text + "\n" + meta["retrieval_text"]) > budget:
            if not group:
                # Never discard source cells to make room for an optional view.
                if count(text) > budget:
                    raise ValueError("single table row exceeds embedding budget")
                meta["retrieval_text"] = ""
                yield text, meta
                continue
            yield make(group)
            group = []
            text, meta = make([index])
            if count(text + "\n" + meta["retrieval_text"]) > budget:
                if count(text) > budget:
                    raise ValueError("single table row exceeds embedding budget")
                meta["retrieval_text"] = ""
                yield text, meta
                continue
        group.append(index)
    if group:
        yield make(group)
    elif len(rows) == headers_count:
        yield make([])
