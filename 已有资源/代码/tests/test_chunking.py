from chunking import (
    ChunkingOptions,
    chunk_docparse_ir,
    chunk_markdown,
    chunk_text,
    estimate_tokens,
)
from document_model import Document


def _doc(text: str, media_type: str = "text/plain") -> Document:
    return Document.from_text(text, "notes/sample.txt", media_type=media_type)


def test_chunk_text_respects_token_budget_and_is_deterministic():
    text = "这是一段用于测试切块预算的中文文本，包含足够的字符来触发切分。" * 60
    doc = _doc(text)
    options = ChunkingOptions(max_tokens=100, overlap_tokens=10)
    first = chunk_text(doc, options)
    second = chunk_text(doc, options)

    assert len(first) > 1
    assert [chunk.id for chunk in first] == [chunk.id for chunk in second]
    assert len({chunk.id for chunk in first}) == len(first)
    for chunk in first:
        assert estimate_tokens(chunk.text) <= options.max_tokens
        assert chunk.source == "notes/sample.txt"
        assert chunk.doc_id == doc.doc_id


def test_chunk_text_empty_document():
    assert chunk_text(_doc("   \n\n  ")) == []


def test_markdown_heading_code_and_table():
    markdown = (
        "# Intro\n\n"
        "alpha paragraph\n\n"
        "## Detail\n\n"
        "beta paragraph\n\n"
        "```python\n"
        "# 这不是标题\n"
        "print('hi')\n"
        "```\n\n"
        "| 项目 | 数量 |\n"
        "| --- | --- |\n"
        "| 螺丝 | 2 |\n\n"
        "# End\n\n"
        "gamma paragraph\n"
    )
    doc = Document.from_text(markdown, "notes/guide.md", media_type="text/markdown")
    chunks = chunk_markdown(doc)

    by_text = {chunk.text: chunk for chunk in chunks}
    assert by_text["alpha paragraph"].section_path == "Intro"
    assert by_text["beta paragraph"].section_path == "Intro > Detail"
    assert by_text["gamma paragraph"].section_path == "End"

    code = [chunk for chunk in chunks if chunk.content_type == "code"]
    assert len(code) == 1 and "# 这不是标题" in code[0].text

    tables = [chunk for chunk in chunks if chunk.content_type == "table"]
    assert len(tables) == 1
    assert "螺丝 | 2" in tables[0].text


def test_markdown_ids_are_deterministic():
    text = "# A\n\nalpha\n\n# B\n\nbeta\n"
    doc = Document.from_text(text, "notes/a.md", media_type="text/markdown")
    assert [chunk.id for chunk in chunk_markdown(doc)] == [
        chunk.id for chunk in chunk_markdown(doc)
    ]


def _block(btype, content, bbox, order=0, html="", block_id=None):
    return {
        "id": block_id or f"b{order}",
        "type": btype,
        "content": content,
        "html": html,
        "bbox": bbox,
        "reading_order": order,
        "confidence": 1.0,
    }


def _ir(pages):
    return {"sha256": "abc123", "pages": pages}


def test_ir_running_header_is_dropped_and_section_path_kept():
    pages = []
    for number in (1, 2, 3):
        pages.append(
            {
                "number": number,
                "blocks": [
                    _block("text", "arXiv:2103.15348v2 [cs.CV] 21 Jun 2021", [0, 0, 1000, 30], 0),
                    _block("title", "1 Introduction", [100, 100, 900, 170], 1),
                    _block("text", f"Body paragraph {number} with enough content.", [100, 200, 900, 400], 2),
                ],
            }
        )

    doc = Document.from_text("", "papers/a.pdf", media_type="application/pdf")
    chunks = chunk_docparse_ir(_ir(pages), doc)

    assert chunks
    assert all("arXiv:2103" not in chunk.text for chunk in chunks)
    assert all(chunk.section_path == "1 Introduction" for chunk in chunks)


def test_ir_short_document_keeps_repeated_line():
    pages = [
        {"number": n, "blocks": [_block("text", "重复的一句话", [0, 0, 100, 20], 0)]}
        for n in (1, 2)
    ]
    doc = Document.from_text("", "papers/b.pdf")
    chunks = chunk_docparse_ir(_ir(pages), doc)
    assert any("重复的一句话" in chunk.text for chunk in chunks)


def test_ir_table_is_one_chunk_and_rendered_as_rows():
    page = {
        "number": 1,
        "blocks": [
            _block("title", "2 Method", [0, 0, 500, 80], 0),
            _block(
                "table",
                "",
                [100, 100, 900, 400],
                1,
                html="<table><tr><td>项目</td><td>数量</td></tr>"
                "<tr><td>螺丝</td><td>2</td></tr></table>",
            ),
            _block("text", "After the table.", [100, 450, 900, 500], 2),
        ],
    }
    doc = Document.from_text("", "papers/c.pdf")
    chunks = chunk_docparse_ir(_ir([page]), doc)

    tables = [chunk for chunk in chunks if chunk.content_type == "table"]
    assert len(tables) == 1
    assert "项目 | 数量" in tables[0].text
    assert "螺丝 | 2" in tables[0].text


def test_ir_chunk_ids_are_deterministic_and_unique():
    page = {
        "number": 1,
        "blocks": [
            _block("text", "第一段内容。" * 20, [0, 0, 1000, 500], 0),
            _block("text", "第二段内容。" * 20, [0, 500, 1000, 900], 1),
        ],
    }
    doc = Document.from_text("", "papers/d.pdf")
    first = [chunk.id for chunk in chunk_docparse_ir(_ir([page]), doc)]
    second = [chunk.id for chunk in chunk_docparse_ir(_ir([page]), doc)]
    assert first == second
    assert len(set(first)) == len(first)


def test_ir_page_mode_yields_page_text():
    page = {"number": 1, "blocks": [_block("page", "整页 OCR 出来的内容", [0, 0, 1000, 1000], 0)]}
    doc = Document.from_text("", "papers/e.pdf")
    chunks = chunk_docparse_ir(_ir([page]), doc)
    assert len(chunks) == 1
    assert chunks[0].content_type == "page_text"
    assert chunks[0].page == 1
