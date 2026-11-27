"""Independent excel connector; only public contracts/models on the host."""
import json
import csv
import io
import os
import subprocess
import sys
from pathlib import Path
from rag_multidocs.contracts import ConnectorResult, ConnectorIssue
from rag_multidocs.model import Document, sha256_file


def parse_csv(file, options):
    raw = file.read_bytes()
    encodings = ((str(options['encoding']),) if options.get('encoding') else
                 ('utf-32',) if raw.startswith((b'\xff\xfe\x00\x00', b'\x00\x00\xfe\xff')) else
                 ('utf-16',) if raw.startswith((b'\xff\xfe', b'\xfe\xff')) else ('utf-8-sig', 'gb18030'))
    text, encoding = None, None
    for name in encodings:
        try:
            text, encoding = raw.decode(name, errors='strict'), name
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise ValueError('CSV_DECODE_ERROR')
    if options.get('delimiter'):
        reader = csv.reader(io.StringIO(text, newline=''), delimiter=options['delimiter'], strict=True)
    else:
        try:
            dialect = csv.Sniffer().sniff(text[:8192], delimiters=',;\t|')
        except csv.Error:
            dialect = csv.excel
        reader = csv.reader(io.StringIO(text, newline=''), dialect=dialect, strict=True)
    rows, records = [], []
    previous, cells_read = 0, 0
    for row in reader:
        records.append({'record': len(rows) + 1, 'line_start': previous + 1, 'line_end': reader.line_num})
        previous = reader.line_num
        if not row:
            records.pop()
            continue
        rows.append(row)
        cells_read += len(row)
        if cells_read > int(options.get('max_cells', 200000)):
            raise ValueError('CSV_CELL_LIMIT')
    if not rows or not rows[0]:
        raise ValueError('CSV_EMPTY')
    if any(len(row) != len(rows[0]) for row in rows):
        raise ValueError('CSV_COLUMN_COUNT_MISMATCH')
    count = int(options.get('header_rows', 1))
    if not 1 <= count <= len(rows):
        raise ValueError('CSV_HEADER_ROWS_INVALID')
    cells = [{'address': f'R{r}C{c}', 'anchor': f'R{r}C{c}', 'value': value,
              'display': value, 'data_type': 's', 'formula': None, 'cached_value': None}
             for r, row in enumerate(rows, 1) for c, value in enumerate(row, 1)]
    content = '\n'.join(' | '.join(row) for row in rows)
    block = {'id': 'csv1', 'type': 'table', 'page': 0, 'reading_order': 0,
             'content': content, 'origin': 'native_extraction', 'ocr_mode': 'native',
             'meta': {'location': {'table': 1, 'row_numbers': list(range(1, len(rows) + 1)), 'column_start': 1},
                      'table_data': {'rows': rows, 'header_rows': count, 'cells': cells, 'csv_record_lines': records}}}
    return {'content': content, 'title': file.stem, 'blocks': [block],
            'metadata': {'encoding': encoding, 'origin': 'native_extraction', 'warnings': []}}


def load(payload, context=None):
    options = dict(payload.get("options") or {})
    settings = (context or {}).get("settings", {})
    python = options.get("parser_python") or settings.get("docparse_python") or sys.executable
    repo = options.get("docparse_repo") or settings.get("docparse_repo") or ""
    path = Path(payload["path"]).expanduser().resolve()
    result = ConnectorResult()
    if not path.exists():
        return ConnectorResult(issues=[ConnectorIssue("FILE_NOT_FOUND", "Input does not exist", str(path))])
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in {".xlsx", ".csv"})
    if not files:
        result.issues.append(ConnectorIssue("EMPTY_SOURCE", "No supported documents", str(path)))
    for file in files:
        source = file.name if path.is_file() else file.relative_to(path).as_posix()
        if file.suffix.lower() not in {".xlsx", ".csv"}:
            result.issues.append(ConnectorIssue("UNSUPPORTED_TYPE", "Expected .xlsx or .csv", source))
            continue
        request = {"path": str(file), "options": options, "docparse_repo": repo}
        try:
            if file.suffix.lower() == '.csv':
                parsed = parse_csv(file, options)
            else:
                process = subprocess.run([str(python), str(Path(__file__).with_name("worker.py"))], input=json.dumps(request), capture_output=True, text=True, timeout=float(options.get("timeout_s", 180)), env={**os.environ, "PYTHONPATH": str(repo), "TOKENIZERS_PARALLELISM": "false"})
                if process.returncode:
                    raise RuntimeError("parser process exited")
                parsed = json.loads(process.stdout)
            if parsed.get("error"):
                result.issues.append(ConnectorIssue(parsed["error"], parsed.get("message", "Parse failed"), source))
                continue
            doc = Document.from_text(parsed["content"], source, title=parsed.get("title", file.stem), media_type="text/csv" if file.suffix.lower() == ".csv" else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", metadata=parsed.get("metadata", {}))
            doc.metadata.update({"sha256": sha256_file(file), "file_sha256": sha256_file(file), "connector": "connector.excel", "parser": "csv" if file.suffix.lower() == ".csv" else "openpyxl"})
            doc.segments = parsed["blocks"]
            result.documents.append(doc)
        except FileNotFoundError:
            result.issues.append(ConnectorIssue("CONNECTOR_UNAVAILABLE", "解析解释器不可用，请检查 DOCPARSE_PYTHON", source))
        except subprocess.TimeoutExpired:
            result.issues.append(ConnectorIssue("PARSE_TIMEOUT", "Parser exceeded deadline", source))
        except (OSError, RuntimeError, ValueError, KeyError, TypeError, csv.Error) as exc:
            code = str(exc) if isinstance(exc, ValueError) and str(exc).startswith('CSV_') else 'PARSE_ERROR'
            result.issues.append(ConnectorIssue(code, "Invalid or unsupported document", source))
    return result
