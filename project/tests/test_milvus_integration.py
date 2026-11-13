import os

import pytest

from rag_multidocs.cli import build_runtime
from rag_multidocs.config import Settings

pytest.importorskip("pymilvus")


@pytest.mark.integration
def test_milvus_roundtrip(tmp_path, project_root):
    settings = Settings(
        data_dir=str(tmp_path / "data"),
        connectors_dir=str(project_root / "connectors"),
        index_backend="milvus",
        fallback_to_keyword=False,
        milvus_uri=os.environ.get("MILVUS_URI", "http://localhost:19530"),
        milvus_db=os.environ.get("MILVUS_DB", "ci_rag"),
        milvus_collection=os.environ.get("MILVUS_COLLECTION", "ci_chunks"),
        embed_backend="hashing",
        embed_dim=256,
    )

    if not os.environ.get("MILVUS_URI"):
        pytest.skip("MILVUS_URI 未设置；本地默认跳过 Milvus 集成测试")

    # CI 设置了 MILVUS_URI，连接失败必须让测试失败，不能伪装成 skip。
    runtime = build_runtime(settings)

    try:
        txt = tmp_path / "notes.txt"
        txt.write_text("检索增强生成需要检索器和生成器。", encoding="utf-8")
        report = runtime.ingest.ingest("txt", str(txt))
        assert report.ok, report.as_dict()

        hits = runtime.search.search("检索器", top_k=3)
        assert hits
        assert hits[0]["source"] == "notes.txt"
        assert "检索器" in hits[0]["text"]

        docs = runtime.index.list_docs()
        assert docs and docs[0]["source"] == "notes.txt"
    finally:
        runtime.index.drop()
        runtime.index.close()
