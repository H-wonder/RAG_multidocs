import json

import pytest

from docparse_connector import (
    DocparseConnector,
    DocparseOptions,
    document_from_ir,
)
from embedders import HashingEmbedder


def test_hashing_embedder_is_deterministic_and_normalized():
    embedder = HashingEmbedder(dim=32)
    first = embedder.embed(["检索增强生成", "RAG"])[0]
    second = embedder.embed(["检索增强生成", "RAG"])[0]

    assert len(first) == 32
    assert first == second
    norm = sum(value * value for value in first) ** 0.5
    assert abs(norm - 1.0) < 1e-9


def _ir():
    return {
        "path": "a.pdf",
        "sha256": "f" * 64,
        "doc_type": "pdf",
        "needs_review": False,
        "warnings": [],
        "pages": [
            {
                "number": 1,
                "blocks": [
                    {"id": "p1b0", "type": "title", "content": "A Paper", "reading_order": 0},
                    {"id": "p1b1", "type": "text", "content": "body", "reading_order": 1},
                ],
            }
        ],
    }


def test_document_from_ir_maps_title_and_source():
    doc = document_from_ir(_ir(), "papers/a.pdf")
    assert doc.title == "A Paper"
    assert doc.source == "papers/a.pdf"
    assert "body" in doc.content
    assert doc.metadata["sha256"] == "f" * 64
    assert doc.metadata["pages"] == 1


def test_document_from_ir_prefers_markdown_content():
    doc = document_from_ir(_ir(), "papers/a.pdf", content="# A Paper\n\nbody\n")
    assert doc.content.startswith("# A Paper")


def test_connector_unavailable_without_repo(tmp_path):
    source = tmp_path / "a.pdf"
    source.write_bytes(b"%PDF-1.4 fake")
    connector = DocparseConnector(DocparseOptions(repo=""))
    assert connector.available() is False
    with pytest.raises(RuntimeError, match="docparse 不可用"):
        connector.parse(source, tmp_path / "out")


def test_connector_reports_missing_file(tmp_path):
    connector = DocparseConnector(DocparseOptions(repo=str(tmp_path)))
    with pytest.raises(FileNotFoundError):
        connector.parse(tmp_path / "missing.pdf", tmp_path / "out")


def test_ir_json_can_be_loaded(tmp_path):
    path = tmp_path / "a.json"
    path.write_text(json.dumps(_ir()), encoding="utf-8")
    from docparse_connector import load_ir

    assert load_ir(path)["doc_type"] == "pdf"
