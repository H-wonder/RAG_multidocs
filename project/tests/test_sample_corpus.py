from rag_multidocs.cli import build_runtime


def test_campus_corpus_has_multiple_document_types(project_root):
    corpus = project_root / "samples" / "campus-corpus"
    expected = [
        "README.md",
        "txt/library-hours.txt",
        "txt/campus-faq.txt",
        "markdown/rag-basics.md",
        "markdown/lab-guide.md",
        "html/library-services.html",
        "pdf/course-project-guide.pdf",
        "pdf/scanned-library-notice.png",
        "office/course-faq.docx",
    ]
    for relative in expected:
        assert (corpus / relative).is_file(), relative

    assert (corpus / "pdf" / "course-project-guide.pdf").read_bytes().startswith(b"%PDF")
    assert (corpus / "pdf" / "scanned-library-notice.png").read_bytes().startswith(b"\x89PNG")
    assert (corpus / "office" / "course-faq.docx").read_bytes().startswith(b"PK")


def test_campus_corpus_txt_markdown_html_ingest_and_search(settings, project_root):
    settings.enabled_connectors = ["txt", "connector.markdown", "connector.html"]
    settings.index_backend = "keyword"
    runtime = build_runtime(settings)
    corpus = project_root / "samples" / "campus-corpus"

    assert runtime.ingest.ingest("txt", str(corpus / "txt")).ok
    assert runtime.ingest.ingest(
        "connector.markdown", str(corpus / "markdown")
    ).ok
    assert runtime.ingest.ingest("connector.html", str(corpus / "html")).ok

    sources = {doc["source"] for doc in runtime.index.list_docs()}
    assert {"library-hours.txt", "campus-faq.txt", "rag-basics.md", "library-services.html"} <= sources

    hits = runtime.search.search("馆际互借")
    assert any(hit["source"] in {"library-hours.txt", "library-services.html"} for hit in hits)

    hits = runtime.search.search("切块")
    assert any(hit["source"] == "rag-basics.md" for hit in hits)

    hits = runtime.search.search("借阅期限")
    assert any(hit["source"] == "campus-faq.txt" for hit in hits)
