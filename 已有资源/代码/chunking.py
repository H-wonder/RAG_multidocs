"""切块：把统一文档模型或 docparse IR 切成可检索、可引用的 Chunk。

来源：``任务二/docflow/chunker.py``。做了三处适配，使其脱离 docflow 配置：
1. 参数收进 ``ChunkingOptions``，不再读全局 config；
2. 输出统一为 ``document_model.Chunk``，可直接交给任意索引后端；
3. 增加 ``chunk_markdown`` / ``chunk_text``，课程项目的 TXT 与 Markdown Connector
   不需要先构造 docparse IR 就能使用同一套切块逻辑。

保持不变的关键设计：
- 按 token 预算切，不按字符数；token 用启发式估计，避免引入 tokenizer；
- 标题层级进入 ``section_path``，表格/公式/代码块整块保留，不被字符数腰斩；
- chunk id 由 (doc_id, 类型, 序号) 决定，重跑结果可复现；
- 跨页重复的页眉/页脚会被过滤（只对多页 IR 生效）。
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable

from document_model import Chunk, Document, chunk_id

DEFAULT_MAX_TOKENS = 480
DEFAULT_OVERLAP_TOKENS = 60
DEFAULT_MAX_TABLE_ROWS = 60
TABLE_ROW_SEP = " | "

_CJK_RE = re.compile(r"[一-鿿]")
_WORD_RE = re.compile(r"[A-Za-z0-9]+")
_SENTENCE_END = re.compile(r"(?<=[。！？!?；;])")
_SECTION_RE = re.compile(
    r"^\s*(\d{1,2}(?:\.\d{1,2}){0,2})(?![\d.])[\s\.、]*[A-Za-z一-鿿]"
)
_NOT_HEADING = re.compile(r"(arxiv|doi|http|www\.|@|20\d{2}[-/年])", re.I)
_HEADING_LEVELS = ((70, 1), (45, 2), (0, 3))


@dataclass
class ChunkingOptions:
    max_tokens: int = DEFAULT_MAX_TOKENS
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS
    max_table_rows: int = DEFAULT_MAX_TABLE_ROWS
    table_row_sep: str = TABLE_ROW_SEP


# ------------------------------------------------------------------ token 预算


def estimate_tokens(text: str) -> int:
    """不引入 tokenizer 的粗略估计（带安全系数）。

    原始实现用 bge-small-zh 的 bert 分词器在 47 个真实 chunk 上校准：
    实际/估计 的 p95 是 1.21、最大 1.28，所以统一乘 1.3，保证估出来的是上界。
    经验值：中文约 1 字 1 token，英文约 4~5 字 1 token。
    """
    cjk = len(_CJK_RE.findall(text))
    words = _WORD_RE.findall(text)
    ascii_chars = sum(len(w) for w in words)
    other = max(0, len(text) - cjk - ascii_chars)
    raw = cjk + len(words) * 1.35 + other * 0.4 + 1
    return int(raw * 1.3)


def _cut_index(text: str, max_tokens: int) -> int:
    """二分找满足 token 预算的最长前缀。"""
    lo, hi = 1, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if estimate_tokens(text[:mid]) <= max_tokens:
            lo = mid
        else:
            hi = mid - 1
    return lo


def split_long_text(text: str, max_tokens: int) -> list[str]:
    """按句子边界把超长段落切开；单句仍超长时按 token 预算硬切。"""
    if estimate_tokens(text) <= max_tokens:
        return [text]
    pieces: list[str] = []
    buf = ""
    for sentence in _SENTENCE_END.split(text):
        if not sentence:
            continue
        while estimate_tokens(sentence) > max_tokens:
            cut = _cut_index(sentence, max_tokens)
            pieces.append(sentence[:cut])
            sentence = sentence[cut:]
        if not sentence:
            continue
        if estimate_tokens(buf + sentence) <= max_tokens:
            buf += sentence
        else:
            if buf:
                pieces.append(buf)
            buf = sentence
    if buf:
        pieces.append(buf)
    return pieces


def _tail_tokens(text: str, max_tokens: int) -> str:
    """取末尾约 max_tokens 个 token，用于块间重叠。"""
    if max_tokens <= 0:
        return ""
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi) // 2
        if estimate_tokens(text[mid:]) <= max_tokens:
            hi = mid
        else:
            lo = mid + 1
    return text[lo:]


def pack_paragraphs(paragraphs: Iterable[str], options: ChunkingOptions) -> list[str]:
    """把段落装成若干块，块间保留少量重叠，避免上下文被切断。"""
    blocks: list[str] = []
    buf = ""
    for text in paragraphs:
        text = (text or "").strip()
        if not text:
            continue
        for piece in split_long_text(text, options.max_tokens):
            if not buf:
                buf = piece
            elif estimate_tokens(buf + "\n" + piece) <= options.max_tokens:
                buf += "\n" + piece
            else:
                blocks.append(buf)
                tail = _tail_tokens(buf, options.overlap_tokens)
                candidate = (tail + "\n" + piece).strip() if tail else piece
                # 重叠部分可能把新块顶超预算，超了就放弃重叠
                if estimate_tokens(candidate) > options.max_tokens:
                    candidate = piece
                buf = candidate
    if buf.strip():
        blocks.append(buf)
    return blocks


# ------------------------------------------------------------------ 纯文本 / Markdown


def _paragraphs(text: str) -> list[str]:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    parts = re.split(r"\n\s*\n", text)
    return [re.sub(r"[ \t]+", " ", part).strip() for part in parts if part.strip()]


def _emit(
    chunks: list[Chunk],
    document: Document,
    text: str,
    content_type: str,
    section_path: str,
    counter: int,
    *,
    page: int = 0,
) -> int:
    """追加一个块并返回新的全局序号。"""
    chunks.append(
        Chunk(
            id=chunk_id(document.doc_id, content_type, counter),
            doc_id=document.doc_id,
            text=text,
            source=document.source,
            index=counter,
            content_type=content_type,
            section_path=section_path,
            page=page,
            page_end=page,
        )
    )
    return counter + 1


def chunk_text(
    document: Document,
    options: ChunkingOptions | None = None,
) -> list[Chunk]:
    """普通纯文本切块：按空行分段，再按 token 预算装箱。"""
    options = options or ChunkingOptions()
    chunks: list[Chunk] = []
    counter = 0
    for packed in pack_paragraphs(_paragraphs(document.content), options):
        counter = _emit(chunks, document, packed, "text", "", counter)
    return chunks


def _flush_markdown_blocks(
    blocks: list[tuple[str, str]],
    document: Document,
    chunks: list[Chunk],
    counter: int,
    options: ChunkingOptions,
    section: str,
) -> int:
    """blocks 元素是 (content_type, text)，按顺序落成 chunk。"""
    for content_type, text in blocks:
        if content_type == "table":
            lines = text.splitlines()
            if len(lines) > options.max_table_rows + 2:
                text = "\n".join(lines[: options.max_table_rows + 2])
                text += f"\n…（表格已截断，原 {len(lines)} 行）"
            counter = _emit(chunks, document, text, "table", section, counter)
        elif content_type == "code":
            for piece in split_long_text(text, options.max_tokens):
                counter = _emit(chunks, document, piece, "code", section, counter)
        else:
            for packed in pack_paragraphs([text], options):
                counter = _emit(chunks, document, packed, "text", section, counter)
    return counter


def chunk_markdown(
    document: Document,
    options: ChunkingOptions | None = None,
) -> list[Chunk]:
    """Markdown 结构化切块。

    - ``#``~``######`` 进入 section_path，标题行本身不进正文；
    - 围栏代码块整块保留，内部出现的 ``#`` 不会被当成标题；
    - 连续以 ``|`` 开头的行视为表格，整块保留；
    - 其余按空行分段并装箱。
    """
    options = options or ChunkingOptions()
    text = (document.content or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = text.splitlines()

    chunks: list[Chunk] = []
    counter = 0
    heading_stack: list[tuple[int, str]] = []
    blocks: list[tuple[str, str]] = []

    buf: list[str] = []
    buf_kind = "text"
    fence: str | None = None

    def section() -> str:
        return " > ".join(title for _, title in heading_stack if title)

    def flush_buf() -> None:
        nonlocal counter
        if not buf:
            return
        counter = _flush_markdown_blocks(
            [(buf_kind, "\n".join(buf).strip())],
            document,
            chunks,
            counter,
            options,
            section(),
        )
        buf.clear()

    for line in lines:
        stripped = line.strip()

        # 围栏代码块：内部内容原样保留
        if fence is not None:
            buf.append(line)
            if stripped.startswith(fence):
                flush_buf()
                fence = None
                buf_kind = "text"
            continue
        if stripped.startswith("```") or stripped.startswith("~~~"):
            flush_buf()
            fence = stripped[:3]
            buf_kind = "code"
            buf.append(line)
            continue

        heading = re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line)
        if heading:
            flush_buf()
            level = len(heading.group(1))
            title = heading.group(2).strip()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            continue

        if stripped.startswith("|"):
            if buf_kind != "table":
                flush_buf()
                buf_kind = "table"
            buf.append(line)
            continue

        if not stripped:
            flush_buf()
            buf_kind = "text"
            continue

        if buf_kind == "table":
            flush_buf()
            buf_kind = "text"
        buf.append(line)

    flush_buf()
    return chunks


# ------------------------------------------------------------------ docparse IR


class _TableText(HTMLParser):
    """把 docparse 输出的 HTML 表格转成 ``a | b | c`` 的行文本。"""

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th"):
            self._cell = []

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th") and self._row is not None and self._cell is not None:
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)


def render_html_table(
    html: str, max_rows: int = DEFAULT_MAX_TABLE_ROWS, row_sep: str = TABLE_ROW_SEP
) -> str:
    parser = _TableText()
    try:
        parser.feed(html)
    except Exception:
        return html.strip()
    if not parser.rows:
        return html.strip()
    lines = [row_sep.join(row) for row in parser.rows[:max_rows]]
    if len(parser.rows) > max_rows:
        lines.append(f"…（表格共 {len(parser.rows)} 行，已截断）")
    return "\n".join(lines)


def _norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def _running_headers(pages: list[dict]) -> set[str]:
    """跨页重复出现的短文本 = 页眉/页脚。"""
    from collections import defaultdict

    seen: dict[str, set[int]] = defaultdict(set)
    for page in pages:
        for block in page.get("blocks", []) or []:
            text = _norm_text(block.get("content") or "")
            if text and len(text) <= 120:
                seen[text].add(int(page.get("number", 0) or 0))
    total = len(pages)
    if total < 3:
        return set()
    threshold = max(3, int(total * 0.5))
    return {text for text, hit in seen.items() if len(hit) >= threshold}


def _looks_like_heading(text: str, bbox: list) -> bool:
    text = (text or "").strip()
    if not text or len(text) > 60:
        return False
    if _NOT_HEADING.search(text):
        return False
    if _SECTION_RE.match(text):
        return True
    height = bbox[3] - bbox[1] if len(bbox) == 4 else 0
    return height >= 60 and len(text) <= 30


def _heading_level(text: str, bbox: list) -> int:
    match = _SECTION_RE.match(text or "")
    if match:
        return min(3, match.group(1).count(".") + 1)
    height = bbox[3] - bbox[1] if len(bbox) == 4 else 0
    for threshold, level in _HEADING_LEVELS:
        if height >= threshold:
            return level
    return 3


def _push_heading(stack: list[tuple[int, str]], level: int, text: str) -> None:
    text = text.strip()
    if not text:
        return
    while stack and stack[-1][0] >= level:
        stack.pop()
    stack.append((level, text))


def _section_path(stack: list[tuple[int, str]]) -> str:
    return " > ".join(t for _, t in stack if t)


def _union_bbox(boxes: Iterable[list]) -> list:
    boxes = [b for b in boxes if len(b) == 4]
    if not boxes:
        return [0, 0, 1000, 1000]
    return [
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    ]


def _ir_chunk(
    document: Document,
    chunks: list[Chunk],
    counter: int,
    text: str,
    content_type: str,
    section: str,
    page: int,
    bbox: list,
    block_ids: list[str],
) -> int:
    chunks.append(
        Chunk(
            id=chunk_id(f"{document.doc_id}:p{page}", content_type, counter),
            doc_id=document.doc_id,
            text=text,
            source=document.source,
            index=counter,
            content_type=content_type,
            section_path=section,
            page=page,
            page_end=page,
            bbox=json.dumps(bbox),
            block_ids=block_ids,
        )
    )
    return counter + 1


def chunk_docparse_ir(
    ir: dict,
    document: Document,
    options: ChunkingOptions | None = None,
) -> list[Chunk]:
    """把 docparse IR 切成结构化块。

    IR 结构见 ``文档/02-文档模型与IR.md``。这里只依赖标准库，可在任意进程运行。
    """
    options = options or ChunkingOptions()
    chunks: list[Chunk] = []
    heading_stack: list[tuple[int, str]] = []
    counter = 0

    pages = sorted(ir.get("pages", []), key=lambda p: p.get("number", 0))
    running = _running_headers(pages)

    for page in pages:
        page_no = int(page.get("number", 0) or 0)
        raw_blocks = page.get("blocks", []) or []
        granular = [b for b in raw_blocks if b.get("type") != "page_markdown"]
        blocks = granular or raw_blocks

        paragraphs: list[tuple[str, list]] = []
        block_ids: list[str] = []

        def flush_paragraphs() -> None:
            nonlocal counter, paragraphs, block_ids
            if not paragraphs:
                return
            section = _section_path(heading_stack)
            bbox = _union_bbox([box for _, box in paragraphs])
            for packed in pack_paragraphs([text for text, _ in paragraphs], options):
                counter = _ir_chunk(
                    document, chunks, counter, packed, "text", section,
                    page_no, bbox, block_ids,
                )
            paragraphs = []
            block_ids = []

        for block in sorted(blocks, key=lambda b: (b.get("reading_order") or 0)):
            btype = block.get("type", "text")
            content = (block.get("content") or "").strip()
            html = (block.get("html") or "").strip()
            bbox = block.get("bbox") or [0, 0, 1000, 1000]
            bid = str(block.get("id", ""))

            if (
                _norm_text(content) in running
                and not (btype == "title" and _SECTION_RE.match(content))
            ):
                continue

            if btype == "title":
                if _looks_like_heading(content, bbox):
                    _push_heading(heading_stack, _heading_level(content, bbox), content)
                elif content:
                    paragraphs.append((content, bbox))
                    block_ids.append(bid)
                continue

            if btype == "page":
                text = content or html
                if text.strip():
                    counter = _ir_chunk(
                        document, chunks, counter, text, "page_text",
                        _section_path(heading_stack), page_no, bbox, [bid],
                    )
                continue

            if btype == "table":
                text = render_html_table(
                    html or content, options.max_table_rows, options.table_row_sep
                )
                if not text.strip():
                    continue
                flush_paragraphs()
                counter = _ir_chunk(
                    document, chunks, counter, text, "table",
                    _section_path(heading_stack), page_no, bbox, [bid],
                )
                continue

            if btype in ("formula", "figure"):
                if content:
                    flush_paragraphs()
                    counter = _ir_chunk(
                        document, chunks, counter, content, btype,
                        _section_path(heading_stack), page_no, bbox, [bid],
                    )
                continue

            if content:
                paragraphs.append((content, bbox))
                block_ids.append(bid)

        flush_paragraphs()

    for chunk in chunks:
        chunk.block_ids = list(dict.fromkeys(chunk.block_ids))
    return chunks


def chunk_document(
    document: Document,
    options: ChunkingOptions | None = None,
) -> list[Chunk]:
    """按 media_type 选择切块方式；docparse 结果请显式调用 ``chunk_docparse_ir``。"""
    if document.media_type in ("text/markdown", "text/x-markdown"):
        return chunk_markdown(document, options)
    return chunk_text(document, options)


# ------------------------------------------------------------------ 落盘


def write_jsonl(chunks: list[Chunk], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(chunk.as_dict(), ensure_ascii=False) + "\n")
    return out


def read_jsonl(path: str | Path) -> list[Chunk]:
    p = Path(path)
    if not p.exists():
        return []
    return [
        Chunk.from_dict(json.loads(line))
        for line in p.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


__all__ = [
    "ChunkingOptions",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_OVERLAP_TOKENS",
    "chunk_document",
    "chunk_docparse_ir",
    "chunk_markdown",
    "chunk_text",
    "estimate_tokens",
    "read_jsonl",
    "render_html_table",
    "split_long_text",
    "write_jsonl",
]
