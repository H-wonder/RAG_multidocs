from manifest_store import ManifestStore, doc_id_for


def _store(tmp_path):
    return ManifestStore(tmp_path / "manifest.db")


def test_scan_detects_add_change_and_delete(tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    note = inbox / "note.md"
    note.write_text("hello", encoding="utf-8")
    store = _store(tmp_path)

    first = store.scan([note], inbox)
    assert len(first["added"]) == 1
    doc_id = first["added"][0]["doc_id"]
    assert doc_id == doc_id_for("note.md")

    second = store.scan([note], inbox)
    assert second["added"] == [] and second["changed"] == []

    note.write_text("hello world", encoding="utf-8")
    third = store.scan([note], inbox)
    assert len(third["changed"]) == 1
    assert store.get(doc_id)["version"] == 2

    fourth = store.scan([], inbox)
    assert any(item["doc_id"] == doc_id for item in fourth["removed"])
    assert store.get(doc_id)["status"] == "DELETED"


def test_mark_and_pending_lifecycle(tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    note = inbox / "a.md"
    note.write_text("content a", encoding="utf-8")
    store = _store(tmp_path)
    doc_id = store.scan([note], inbox)["added"][0]["doc_id"]

    assert any(row["doc_id"] == doc_id for row in store.pending())

    store.mark(doc_id, "PARSED", pages=3, simhash="deadbeef")
    row = store.get(doc_id)
    assert row["status"] == "PARSED" and row["pages"] == 3
    assert row["simhash"] == "deadbeef"

    store.mark(doc_id, "INDEXED")
    assert not any(row["doc_id"] == doc_id for row in store.pending())


def test_events_are_recorded(tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    note = inbox / "b.md"
    note.write_text("content b", encoding="utf-8")
    store = _store(tmp_path)
    doc_id = store.scan([note], inbox)["added"][0]["doc_id"]
    store.mark(doc_id, "FAILED", last_error="boom")

    events = store.events(doc_id)
    assert any(event["event"] == "discovered" for event in events)
    assert any(event["event"] == "status:FAILED" for event in events)


def test_indexed_fingerprints_excludes_non_indexed(tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    a = inbox / "a.md"
    b = inbox / "b.md"
    a.write_text("aaa", encoding="utf-8")
    b.write_text("bbb", encoding="utf-8")
    store = _store(tmp_path)
    result = store.scan([a, b], inbox)
    ids = [item["doc_id"] for item in result["added"]]

    store.mark(ids[0], "INDEXED", simhash="a" * 16)
    store.mark(ids[1], "DUPLICATE", simhash="b" * 16)

    fingerprints = store.indexed_fingerprints()
    assert fingerprints == [(ids[0], "a" * 16)]


def test_unknown_status_is_rejected(tmp_path):
    store = _store(tmp_path)
    try:
        store.mark("nope", "SOMETHING", pages=1)
    except ValueError as exc:
        assert "未知状态" in str(exc)
    else:
        raise AssertionError("应当拒绝未知状态")
