import json


from rag_multidocs.connectors.registry import ConnectorRegistry
from rag_multidocs.connectors.txt import TxtConnector
from rag_multidocs.index import KeywordIndex
from rag_multidocs.ingestion.catalog import DocumentCatalog
from rag_multidocs.ingestion.service import IngestService
from rag_multidocs.ingestion.structured import StructuredOptions, structured_chunks
from rag_multidocs.model import Document


def service(tmp_path, index=None):
    r = ConnectorRegistry()
    r.register(TxtConnector())
    index = index or KeywordIndex(path=tmp_path / "index.json")
    return IngestService(r, index)


def test_unchanged_processing_change_publish_and_failure_recovery(tmp_path):
    s = service(tmp_path)
    p = tmp_path / "a.txt"
    p.write_text("保修期12个月。")
    a = s.ingest("txt", str(p))
    b = s.ingest("txt", str(p))
    assert a.chunks > 0 and b.chunks == 0 and b.documents[0]["status"] == "unchanged"
    # Same bytes, different processing configuration must create a new version.
    c = s.ingest("txt", str(p), {"processing_revision": "2"})
    assert c.documents[0]["version"] == 2
    p.write_text("保修期24个月。")
    original = s.index.stage_version

    def failed(*args):
        raise RuntimeError("index write failed")

    s.index.stage_version = failed
    assert s.ingest("txt", str(p)).issues[0].code == "INGEST_FAILED"
    assert "12个月" in s.index.search("保修期")[0]["text"]
    s.index.stage_version = original
    d = s.ingest("txt", str(p))
    assert d.documents[0]["version"] == 4
    assert all("24个月" in h["text"] for h in s.index.search("保修期"))
    assert s.catalog.cleanup() == 3
    assert s.index.stats()["chunks"] == 1


def test_sqlite_publish_failure_keeps_old_version_and_delete_cleanup(tmp_path):
    s = service(tmp_path)
    p = tmp_path / "a.txt"
    p.write_text("原始文档知识。")
    did = s.ingest("txt", str(p)).documents[0]["doc_id"]
    s.catalog.db.execute(
        "CREATE TRIGGER fail_activation BEFORE UPDATE ON documents BEGIN SELECT RAISE(ABORT,'failure'); END"
    )
    s.catalog.db.commit()
    p.write_text("替换后的知识。")
    assert not s.ingest("txt", str(p)).ok
    assert "原始" in s.index.search("知识")[0]["text"]
    s.catalog.db.execute("DROP TRIGGER fail_activation")
    s.catalog.db.commit()
    assert s.ingest("txt", str(p)).ok
    s.catalog.delete(did)
    assert s.index.search("知识") == []
    assert s.catalog.cleanup() == 3


def test_keyword_reload_and_filters_metadata(tmp_path):
    s = service(tmp_path)
    d = Document.from_text("示例正文", "doc.pdf")
    d.segments = [
        {
            "id": "b",
            "type": "text",
            "content": "示例正文",
            "page": 3,
            "page_end": 4,
            "bbox": [1, 2, 3, 4],
            "ocr_mode": "region",
            "needs_review": True,
        }
    ]
    chunks = structured_chunks(d)
    s.catalog.publish(d, chunks, "fingerprint")
    index = KeywordIndex.load(tmp_path / "index.json")
    DocumentCatalog(index)
    h = index.search("示例", filters={"page": 3, "content_type": "text"})[0]
    assert h["page_end"] == 4 and json.loads(h["bbox"]) == [1, 2, 3, 4]
    assert h["ocr_mode"] == "region" and h["needs_review"]
    assert index.search("示例", filters={"page": 9}) == []


def test_table_rows_repeated_header_formula_and_no_invented_bbox():
    d = Document.from_text("", "record")
    rows = ["名称 | 数值"] + [f"指标{i} | {i}" for i in range(30)]
    d.segments = [
        {"id": "t", "type": "table", "content": "\n".join(rows), "page": 1},
        {"id": "f", "type": "formula", "content": "E = m c^2", "page": 2},
        {"id": "g", "type": "figure", "caption": "已有图注", "ocr_text": "图中已有文字", "page": 3},
    ]
    chunks = structured_chunks(d, StructuredOptions(80, 20, 10))
    tables = [c for c in chunks if c.content_type == "table"]
    assert len(tables) > 1 and all(c.text.startswith(rows[0]) for c in tables)
    assert all(row in "\n".join(c.text for c in tables) for row in rows)
    assert any(c.content_type == "formula" and c.text == "E = m c^2" for c in chunks)
    assert any(c.content_type == "figure" and "已有图注" in c.text for c in chunks)
    assert all(not c.bbox for c in chunks)


def test_repeated_body_not_mistaken_for_running_headers():
    d = Document.from_text("", "d")
    d.segments = [
        {
            "id": str(i),
            "type": "text",
            "page": i,
            "content": "重复出现的正文结论",
            "bbox": [100, 300, 900, 500],
        }
        for i in range(1, 4)
    ]
    assert len(structured_chunks(d)) == 3


def test_no_auto_delete_near_duplicate(tmp_path):
    s = service(tmp_path)
    for name in ["a.txt", "b.txt"]:
        p = tmp_path / name
        p.write_text("相同知识内容。")
        r = s.ingest("txt", str(p))
        assert r.ok
    assert r.documents[0]["near_duplicate_candidates"]
    assert s.index.stats()["docs"] == 2
