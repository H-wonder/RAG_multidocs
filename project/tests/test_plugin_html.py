from rag_multidocs.cli import build_runtime


def _html_runtime(settings):
    settings.enabled_connectors = ["connector.html"]
    return build_runtime(settings)


def test_html_plugin_extracts_visible_text(settings, tmp_path):
    docs = tmp_path / "html"
    docs.mkdir()
    (docs / "guide.html").write_text(
        """
        <html>
          <head>
            <title>检索指南</title>
            <style>.hidden { color: red; }</style>
            <script>const secretToken = "SHOULD_NOT_INDEX";</script>
          </head>
          <body>
            <h1>第一章</h1>
            <p>可见正文，包含检索词。</p>
            <p>第二段。</p>
          </body>
        </html>
        """,
        encoding="utf-8",
    )
    runtime = _html_runtime(settings)

    report = runtime.ingest.ingest("connector.html", str(docs))

    assert report.ok
    assert report.documents[0]["source"] == "guide.html"
    assert report.documents[0]["title"] == "检索指南"
    hits = runtime.search.search("检索词")
    assert hits
    assert hits[0]["source"] == "guide.html"
    assert "检索词" in hits[0]["text"]
    assert not runtime.search.search("SHOULD_NOT_INDEX")


def test_html_plugin_errors(settings, tmp_path):
    runtime = _html_runtime(settings)

    empty = tmp_path / "empty"
    empty.mkdir()
    report = runtime.ingest.ingest("connector.html", str(empty))
    assert not report.ok
    assert report.issues[0].code == "EMPTY_SOURCE"

    txt = tmp_path / "note.txt"
    txt.write_text("not html", encoding="utf-8")
    report = runtime.ingest.ingest("connector.html", str(txt))
    assert not report.ok
    assert report.issues[0].code == "UNSUPPORTED_TYPE"


def test_html_plugin_can_be_disabled(settings, tmp_path):
    settings.enabled_connectors = ["txt"]
    runtime = build_runtime(settings)
    html = tmp_path / "a.html"
    html.write_text("<p>text</p>", encoding="utf-8")

    report = runtime.ingest.ingest("connector.html", str(html))

    assert not report.ok
    assert report.issues[0].code == "CONNECTOR_DISABLED"
