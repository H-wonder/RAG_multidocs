import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from rag_multidocs.connectors.registry import ConnectorRegistry
from rag_multidocs.connectors.txt import TxtConnector
from rag_multidocs.index import KeywordIndex
from rag_multidocs.ingestion.catalog import DocumentCatalog
from rag_multidocs.ingestion.service import IngestService
from rag_multidocs.ingestion.structured import structured_chunks
from rag_multidocs.model import Document


def setup(tmp_path):
    registry = ConnectorRegistry()
    registry.register(TxtConnector())
    return IngestService(registry, KeywordIndex(path=tmp_path / "index.json"))


def test_restart_does_not_activate_complete_interrupted_staging(tmp_path):
    service = setup(tmp_path)
    p = tmp_path / "source.txt"
    p.write_text("original evidence")
    did = service.ingest("txt", str(p)).documents[0]["doc_id"]
    path = service.catalog.path
    service.catalog.close()
    script = """from rag_multidocs.index import KeywordIndex
from rag_multidocs.ingestion.catalog import DocumentCatalog
import os
c=DocumentCatalog(KeywordIndex.load(os.environ['P0_INDEX']))
c.db.execute("INSERT INTO versions VALUES (?,2,'staging','hash','fingerprint','')",(os.environ['P0_DOC'],))
c.db.commit()
c.index.stage_version(None,[{'id':'interrupted-row','doc_id':os.environ['P0_DOC'],'version':2,'index':0,'text':'unfinished evidence','source':'source.txt'}],2)
os._exit(17)
"""
    env = {
        **os.environ,
        "P0_INDEX": str(tmp_path / "index.json"),
        "P0_DOC": did,
        "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    assert subprocess.run([sys.executable, "-c", script], env=env).returncode == 17
    index = KeywordIndex.load(tmp_path / "index.json")
    catalog = DocumentCatalog(index, path)
    assert catalog.active()[did] == 1
    assert catalog.recovery_warnings[0]["code"] == "STAGING_INTERRUPTED"
    assert catalog.reconcile()[0]["status"] == "failed"
    assert catalog.reconcile()[0]["index_rows"] == 1
    assert index.search("original")[0]["version"] == 1
    assert catalog.cleanup() == 1 and catalog.cleanup() == 0
    catalog.close()
    catalog = DocumentCatalog(index, path)
    assert not catalog.recovery_warnings and catalog.active()[did] == 1


def test_snapshot_pins_old_version_and_blocks_gc(tmp_path):
    service = setup(tmp_path)
    p = tmp_path / "source.txt"
    p.write_text("old evidence first paragraph")
    did = service.ingest("txt", str(p)).documents[0]["doc_id"]
    first = service.index.search("evidence")[0]
    with service.catalog.read_snapshot() as snapshot:
        p.write_text("new evidence second paragraph")
        assert service.ingest("txt", str(p)).documents[0]["version"] == 2
        assert service.index.search("evidence")[0]["version"] == 1
        assert {h["version"] for h in service.catalog.neighbors([first["chunk_id"]], snapshot)} == {
            1
        }
        assert service.catalog.cleanup() == 0
        assert service.catalog.cleanup_warnings[0]["code"] == "CLEANUP_PINNED"
    assert service.index.search("evidence")[0]["version"] == 2
    assert service.catalog.neighbors([first["chunk_id"]]) == []
    assert service.catalog.cleanup() == 1
    assert service.catalog.cleanup() == 0
    assert service.catalog.active()[did] == 2


def test_second_process_writer_rejected(tmp_path):
    service = setup(tmp_path)
    script = """from rag_multidocs.index import KeywordIndex
from rag_multidocs.ingestion.catalog import DocumentCatalog,KnowledgeBusyError
import os
try:DocumentCatalog(KeywordIndex(path=os.environ['P0_INDEX']))
except KnowledgeBusyError:raise SystemExit(23)
"""
    env = {
        **os.environ,
        "P0_INDEX": str(tmp_path / "index.json"),
        "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
    }
    with service.catalog.writer():
        result = subprocess.run([sys.executable, "-c", script], env=env)
    assert result.returncode == 23


def test_atomic_json_failure_preserves_old_bytes(tmp_path, monkeypatch):
    service = setup(tmp_path)
    p = tmp_path / "source.txt"
    p.write_text("old evidence")
    assert service.ingest("txt", str(p)).ok
    before = service.index.path.read_bytes()
    monkeypatch.setattr(
        os, "replace", lambda *args: (_ for _ in ()).throw(OSError("replace failure"))
    )
    p.write_text("new evidence")
    assert not service.ingest("txt", str(p)).ok
    assert service.index.path.read_bytes() == before
    assert json.loads(before)["chunks"]
    assert not list(tmp_path.glob("index.json.*.tmp"))
    assert "old" in service.index.search("evidence")[0]["text"]


def test_historical_rows_do_not_change_bm25_and_delete_legacy(tmp_path):
    index = KeywordIndex(path=tmp_path / "index.json")
    legacy = Document.from_text("legacy evidence", "legacy")
    index.add(structured_chunks(legacy))
    catalog = DocumentCatalog(index)
    assert index.search("legacy")
    doc = Document.from_text("current evidence", "current")
    catalog.publish(doc, structured_chunks(doc), "a")
    before = index.search("current")[0]["score"]
    index.add(
        [{"id": "unpublished", "doc_id": doc.doc_id, "text": "current " * 100, "version": 99}]
    )
    assert index.search("current")[0]["score"] == before
    catalog.delete(legacy.doc_id)
    assert index.search("legacy") == []
    assert not any(h.get("version") == 99 for h in index.search("current"))


def test_cleanup_failure_retry_and_optional_analysis_is_not_publish_failure(tmp_path, monkeypatch):
    service = setup(tmp_path)
    p = tmp_path / "source.txt"
    p.write_text("old evidence")
    assert service.ingest("txt", str(p)).ok
    monkeypatch.setattr(
        service.catalog, "near_duplicates", lambda d: (_ for _ in ()).throw(RuntimeError())
    )
    p.write_text("new evidence")
    report = service.ingest("txt", str(p))
    assert report.ok and report.documents[0]["version"] == 2
    assert report.documents[0]["warnings"][0]["code"] == "NEAR_DUPLICATE_CHECK_FAILED"
    remove = service.index.remove_version
    monkeypatch.setattr(
        service.index, "remove_version", lambda *a: (_ for _ in ()).throw(OSError())
    )
    with pytest.raises(OSError):
        service.catalog.cleanup()
    monkeypatch.setattr(service.index, "remove_version", remove)
    assert service.catalog.cleanup() == 1 and service.catalog.cleanup() == 0


def test_reader_thread_during_publish_has_consistent_rows(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    import threading

    service = setup(tmp_path)
    p = tmp_path / "source.txt"
    p.write_text("old concurrent evidence")
    assert service.ingest("txt", str(p)).ok
    entered = threading.Event()
    release = threading.Event()
    old_stage = service.index.stage_version

    def delayed(*args):
        entered.set()
        assert release.wait(3)
        return old_stage(*args)

    service.index.stage_version = delayed
    p.write_text("new concurrent evidence")
    with ThreadPoolExecutor(max_workers=2) as pool:
        writing = pool.submit(service.ingest, "txt", str(p))
        assert entered.wait(3)
        reading = pool.submit(service.index.search, "evidence")
        release.set()
        assert writing.result(timeout=3).ok
        assert {h["version"] for h in reading.result(timeout=3)} == {2}
