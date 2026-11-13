from rag_multidocs.cli import build_runtime


def test_milvus_unavailable_falls_back_to_keyword(settings):
    settings.index_backend = "milvus"
    settings.fallback_to_keyword = True
    settings.milvus_uri = "http://127.0.0.1:1"
    settings.embed_backend = "hashing"
    settings.embed_dim = 64

    runtime = build_runtime(settings)

    assert runtime.backend == "keyword"
    assert "回退" in runtime.warning
