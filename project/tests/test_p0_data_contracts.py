import pytest

from rag_multidocs.contracts import ConnectorResult
from rag_multidocs.index import KeywordIndex
from rag_multidocs.ingestion.catalog import DocumentCatalog
from rag_multidocs.ingestion.service import IngestService
from rag_multidocs.ingestion.structured import StructuredOptions, structured_chunks
from rag_multidocs.model import Document, normalize_hit
from test_p0_publication import setup


def test_input_cache_skips_connector_and_volatile_ir_does_not_change_version(tmp_path, monkeypatch):
    service = setup(tmp_path)
    p = tmp_path / "a.txt"
    p.write_text("same source knowledge")
    assert service.ingest("txt", str(p)).ok
    monkeypatch.setattr(
        service.registry,
        "load",
        lambda *args: (_ for _ in ()).throw(AssertionError("must not parse")),
    )
    assert service.ingest("txt", str(p)).documents[0]["status"] == "unchanged"

    class Registry:
        context = {}
        enabled = None

        def plugin_infos(self):
            return []

        def load(self, *a):
            d = Document.from_text("正文", "ir.pdf")
            d.metadata["docparse_ir"] = {
                "latency_ms": self.latency,
                "meta": {"unrecognized_runtime_counter": self.latency},
                "pages": [
                    {
                        "number": 1,
                        "mode": "native",
                        "usage": {"n": self.latency},
                        "blocks": [{"type": "text", "content": "正文", "bbox": [1, 2, 30, 40]}],
                    }
                ],
            }
            return ConnectorResult([d])

    r = Registry()
    r.latency = 10
    s = IngestService(r, KeywordIndex())
    assert s.ingest("x", "opaque://uri").documents[0]["version"] == 1
    r.latency = 99
    assert s.ingest("x", "opaque://uri").documents[0]["status"] == "unchanged"


def test_quality_gate_and_original_metadata_no_fabricated_bbox():
    class Registry:
        context = {}
        enabled = None

        def plugin_infos(self):
            return []

        def load(self, *a):
            return ConnectorResult([self.d])

    r = Registry()
    r.d = Document.from_text("incomplete", "scan")
    r.d.metadata["ir"] = {"pages": [{"number": 1, "mode": "failed", "blocks": []}]}
    s = IngestService(r, KeywordIndex())
    assert not s.ingest("opaque", "uri").ok and not s.catalog.active()
    r.d.metadata["ir"]["pages"].append(
        {"number": 2, "mode": "page", "blocks": [{"type": "page", "content": "可用正文"}]}
    )
    report = s.ingest("opaque", "uri")
    assert report.ok and any(w["code"] == "PARTIAL_PARSE" for w in report.documents[0]["warnings"])
    assert not s.index.search("正文")[0]["bbox"]
    strict = IngestService(r, KeywordIndex(), allow_partial=False)
    assert not strict.ingest("opaque", "uri").ok


def test_reading_order_numbered_title_code_and_hit_types():
    d = Document.from_text("", "sample")
    d.segments = [
        {
            "id": "b",
            "type": "text",
            "page": 1,
            "reading_order": 2,
            "content": "第二段结论",
            "bbox": [1, 100, 90, 200],
        },
        {"id": "h", "type": "title", "page": 1, "reading_order": 0, "content": "1.2 实验结果"},
        {"id": "a", "type": "text", "page": 1, "reading_order": 1, "content": "第一段证据"},
    ]
    chunks = structured_chunks(d, StructuredOptions(min_tokens=0))
    assert "第一段" in chunks[0].text and "第二段" in chunks[1].text
    assert all("1.2 实验结果" in c.section_path for c in chunks)
    hit = normalize_hit(chunks[0].as_dict())
    assert isinstance(hit["block_ids"], list) and not hit["bbox"]
    with pytest.raises(ValueError, match="BBOX"):
        normalize_hit({"bbox": "[0,0,2000,3000]"})
    d.segments = [{"type": "code", "content": "code line " * 1000}]
    with pytest.raises(ValueError):
        structured_chunks(d, StructuredOptions(64, 0, 0))


def test_space_identity_rejects_reuse(tmp_path):
    index = KeywordIndex(path=tmp_path / "a.json")
    c = DocumentCatalog(index, tmp_path / "catalog.sqlite")
    c.close()
    with pytest.raises(ValueError, match="MISMATCH"):
        DocumentCatalog(KeywordIndex(path=tmp_path / "b.json"), tmp_path / "catalog.sqlite")
