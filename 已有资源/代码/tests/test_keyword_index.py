from document_model import Document
from chunking import chunk_markdown, chunk_text
from keyword_index import KeywordIndex, tokenize


def _txt_chunks(text: str, source: str = "notes.txt"):
    doc = Document.from_text(text, source)
    return chunk_text(doc)


def _md_chunks(text: str, source: str = "guide.md"):
    doc = Document.from_text(text, source, media_type="text/markdown")
    return chunk_markdown(doc)


def test_tokenize_chinese_and_english():
    tokens = tokenize("RAG 检索器 retrieval")
    assert "rag" in tokens and "retrieval" in tokens
    assert "检" in tokens and "检索" in tokens


def test_search_returns_hit_with_source():
    index = KeywordIndex()
    index.add(_txt_chunks("检索增强生成需要检索器和生成器。", "notes/rag.txt"))
    index.add(_md_chunks("# 切块\n\n表格必须整块保留。", "docs/chunking.md"))

    hits = index.search("检索器")
    assert hits
    assert hits[0]["source"] == "notes/rag.txt"
    assert "检索器" in hits[0]["text"]
    assert hits[0]["chunk_id"] and hits[0]["doc_id"]


def test_search_filters_by_doc_id():
    index = KeywordIndex()
    a = _txt_chunks("检索器负责取回段落。", "a.txt")
    b = _txt_chunks("检索器也可以联合训练。", "b.txt")
    index.add(a)
    index.add(b)

    doc_a = a[0].doc_id
    hits = index.search("检索器", doc_ids=[doc_a])
    assert len(hits) == 1
    assert hits[0]["doc_id"] == doc_a


def test_empty_query_and_unknown_term_return_empty():
    index = KeywordIndex()
    index.add(_txt_chunks("检索器负责取回段落。"))
    assert index.search("") == []
    assert index.search("完全不存在的词xyz") == []


def test_remove_doc_and_clear():
    index = KeywordIndex()
    chunks = _txt_chunks("检索器负责取回段落。", "a.txt")
    index.add(chunks)
    assert index.stats()["docs"] == 1

    removed = index.remove_doc(chunks[0].doc_id)
    assert removed == len(chunks)
    assert index.search("检索器") == []

    index.add(chunks)
    index.clear()
    assert index.stats() == {"chunks": 0, "tokens": 0, "docs": 0}


def test_save_and_load_roundtrip(tmp_path):
    index = KeywordIndex()
    index.add(_txt_chunks("检索器负责取回段落。", "a.txt"))
    path = index.save(tmp_path / "index.json")

    loaded = KeywordIndex.load(path)
    hits = loaded.search("检索器")
    assert hits
    assert hits[0]["source"] == "a.txt"
    assert loaded.stats() == index.stats()


def test_same_chunk_id_is_replaced_not_duplicated():
    index = KeywordIndex()
    chunks = _txt_chunks("第一版内容。", "a.txt")
    index.add(chunks)
    chunks[0].text = "第二版内容，包含检索器。"
    index.add(chunks)

    assert index.stats()["chunks"] == 1
    hits = index.search("检索器")
    assert hits and "第二版" in hits[0]["text"]
