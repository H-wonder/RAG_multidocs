
def test_txt_ingest_and_search_returns_source(runtime, tmp_path):
    ingest, search = runtime.ingest, runtime.search
    path = tmp_path / "notes.txt"
    path.write_text("检索增强生成需要检索器和生成器。", encoding="utf-8")

    report = ingest.ingest("txt", str(path))

    assert report.ok
    assert report.chunks >= 1
    assert report.documents[0]["source"] == "notes.txt"
    hits = search.search("检索器")
    assert hits
    assert hits[0]["source"] == "notes.txt"
    assert "检索器" in hits[0]["text"]


def test_markdown_extension_is_loaded_and_searchable(runtime, tmp_path):
    ingest, search = runtime.ingest, runtime.search
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "guide.md").write_text(
        "# 切块\n\n## 表格\n\n表格必须整块保留。\n", encoding="utf-8"
    )

    report = ingest.ingest("connector.markdown", str(docs))

    assert report.ok
    assert report.documents[0]["source"] == "guide.md"
    hits = search.search("表格")
    assert hits
    assert hits[0]["source"] == "guide.md"
    assert "表格" in hits[0]["section_path"]


def test_reingest_replaces_document_chunks(runtime, tmp_path):
    index, ingest = runtime.index, runtime.ingest
    path = tmp_path / "note.txt"
    path.write_text("第一版内容，包含检索器。", encoding="utf-8")
    ingest.ingest("txt", str(path))
    first = index.stats()["chunks"]

    path.write_text("第二版内容，仍然包含检索器。", encoding="utf-8")
    ingest.ingest("txt", str(path))

    assert index.stats()["chunks"] == first
    assert index.list_docs()[0]["source"] == "note.txt"
