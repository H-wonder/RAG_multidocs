import json

from rag_multidocs.chunking import chunk_docparse_ir
from rag_multidocs.connectors.pdf import document_from_ir


def test_docparse_ir_maps_to_document_and_structured_chunks(project_root):
    ir = json.loads(
        (project_root / "samples" / "docparse_ir_sample.json").read_text(
            encoding="utf-8"
        )
    )
    document = document_from_ir(ir, "rag-intro.pdf")
    chunks = chunk_docparse_ir(ir, document)

    assert document.title == "1 Introduction"
    assert document.metadata["pages"] == 2
    assert chunks
    assert all(chunk.page >= 1 for chunk in chunks)
    assert any(chunk.content_type == "table" for chunk in chunks)
    assert any(chunk.section_path for chunk in chunks)
    assert any("Retriever" in chunk.text for chunk in chunks)
