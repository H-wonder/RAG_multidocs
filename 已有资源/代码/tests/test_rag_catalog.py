import re
from types import SimpleNamespace

import pytest
from rag_catalog import Catalog


class Index:
    def __init__(self):
        self.collection = "test"
        self.uri = "local"
        self.db_name = "rag"
        self.embedder = SimpleNamespace(
            model_name="bge",
            dim=2,
            _load=lambda: SimpleNamespace(
                tokenizer=SimpleNamespace(encode=lambda t: list(t)), max_seq_length=512
            ),
        )
        self.rows = {}
        self.fail = False

    def ensure_ready(self):
        pass

    def client(self):
        return self

    def query(self, collection, filter, **kwargs):
        did = re.search(r'doc_id == "([^"]+)"', filter)[1]
        version = int(re.search(r"version == (\d+)", filter)[1])
        return [{"count(*)": len(self.rows.get((did, version), []))}]

    def ingest(self, did, chunks, version, **kwargs):
        if self.fail:
            raise RuntimeError("Milvus unavailable")
        self.rows[did, version] = chunks


def test_incremental_noop_update_and_failed_publish_preserves_old_version(tmp_path):
    root = tmp_path / "docs"
    root.mkdir()
    p = root / "a.txt"
    p.write_text("原始知识")
    index = Index()
    catalog = Catalog(tmp_path / "state/catalog.sqlite", index)
    first = catalog.ingest(p, root)
    assert catalog.ingest(p, root)["status"] == "UNCHANGED"
    p.write_text("更新后的知识")
    second = catalog.ingest(p, root)
    assert second["version"] == 2 and first["doc_id"] == second["doc_id"]
    assert index.rows[first["doc_id"], 1][0].id != index.rows[first["doc_id"], 2][0].id
    p.write_text("第三版")
    index.fail = True
    with pytest.raises(RuntimeError):
        catalog.ingest(p, root)
    assert "version == 2" in catalog.active_filter()
    assert "version == 3" not in catalog.active_filter()
    with catalog.connect() as c:
        assert (
            c.execute("SELECT status FROM versions WHERE version=3").fetchone()[0]
            == "FAILED"
        )


def test_same_filename_under_different_directories_has_distinct_identity(tmp_path):
    for folder in ["a", "b"]:
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "same.txt").write_text("知识")
    c = Catalog(tmp_path / "state/catalog.sqlite", Index())
    x = c.ingest(tmp_path / "a/same.txt", tmp_path)
    y = c.ingest(tmp_path / "b/same.txt", tmp_path)
    assert x["doc_id"] != y["doc_id"]
    assert x["doc_id"] in c.active_filter(["a/same.txt"])
    assert y["doc_id"] not in c.active_filter(["a/same.txt"])
    with pytest.raises(ValueError):
        c.active_filter(["missing.txt"])


def test_collection_or_model_change_rejected(tmp_path):
    index = Index()
    Catalog(tmp_path / "c.sqlite", index)
    index.embedder.model_name = "different"
    with pytest.raises(ValueError, match="绑定"):
        Catalog(tmp_path / "c.sqlite", index)


def test_rebuild_when_milvus_rows_missing(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("知识")
    index = Index()
    c = Catalog(tmp_path / "c.sqlite", index)
    first = c.ingest(p, tmp_path)
    index.rows.clear()
    assert c.ingest(p, tmp_path)["version"] == first["version"] + 1


def test_neighbors_never_load_superseded_version(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("旧知识")
    c = Catalog(tmp_path / "c.sqlite", Index())
    first = c.ingest(p, tmp_path)
    old_id = c.index.rows[first["doc_id"], 1][0].id
    p.write_text("新知识")
    second = c.ingest(p, tmp_path)
    new_id = c.index.rows[second["doc_id"], 2][0].id
    assert c.neighbors([old_id]) == []
    assert c.neighbors([new_id])[0]["text"] == "新知识"
