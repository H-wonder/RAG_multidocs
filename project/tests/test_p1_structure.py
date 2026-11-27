import copy

import pytest

from rag_multidocs.model import Document, embedding_text
from rag_multidocs.ingestion.structured import StructuredOptions, structured_chunks
from rag_multidocs.ingestion.chunking import estimate_tokens
from rag_multidocs.ingestion.service import canonical_content
from rag_multidocs.qa.grounding import citations
from test_p0_publication import setup


def test_nested_metadata_and_chunk_quality_reach_public_report(tmp_path):
    from rag_multidocs.model import normalize_hit
    from rag_multidocs.connectors.registry import ConnectorRegistry
    from rag_multidocs.contracts import ConnectorResult
    from rag_multidocs.index import KeywordIndex
    from rag_multidocs.ingestion.service import IngestService
    doc=Document.from_text('', 'damaged.pdf')
    doc.segments=[{'type':'table','content':'Metric | Baseline | Weighted\nAccuracy | 96 80% . | 93 00% .','page':7}]
    class Input:
        name='test.input'
        def load(self,config):
            return ConnectorResult([copy.deepcopy(doc)])
    registry=ConnectorRegistry()
    registry.register(Input())
    service=IngestService(registry,KeywordIndex(path=tmp_path/'index.json'))
    report=service.ingest('test.input','opaque-source')
    assert report.documents[0]['needs_review']
    assert report.documents[0]['quality_status']=='partial_or_review'
    assert any(w['code']=='UNRESOLVED_NUMERIC_FORMAT' for w in report.documents[0]['warnings'])
    service.catalog.close()
    strict=IngestService(registry,KeywordIndex(path=tmp_path/'strict.json'),strict=True)
    assert not strict.ingest('test.input','opaque-source').ok
    assert strict.index.search('Accuracy')==[]
    strict.catalog.close()
    hit=normalize_hit({'text':'raw','metadata':{'location':{'worksheet':'Real'},'warnings':[{'code':'ORIGINAL','message':'source'}],'needs_review':True}})
    assert hit['location']=={'worksheet':'Real'} and hit['warnings'][0]['code']=='ORIGINAL' and hit['needs_review']


def table_document():
    doc = Document.from_text("", "book.xlsx")
    rows = [["Object", "Metrics", "Metrics"], ["Object", "Accuracy", "Recall"]] + [[f"M{i}", f"{i}%", f"{i+1}%"] for i in range(20)]
    doc.segments = [{"id": "table", "type": "table", "page": 0, "section_path": "Sheet: Metrics", "content": "\n".join(" | ".join(r) for r in rows), "meta": {"location": {"worksheet": "Metrics", "cell_range": "A1:C22", "row_numbers": list(range(1,23)), "column_start": 1}, "table_data": {"rows": rows, "header_rows": 2}}}]
    return doc


def test_grouped_table_repeats_two_headers_preserves_all_rows_and_exact_ranges():
    doc = table_document()
    chunks = structured_chunks(doc, StructuredOptions(max_tokens=140, min_tokens=0, overlap_tokens=0))
    assert len(chunks) > 1
    data_rows = []
    for chunk in chunks:
        assert chunk.text.startswith("Object | Metrics | Metrics\nObject | Accuracy | Recall\n")
        assert estimate_tokens(embedding_text(chunk.as_dict())) <= 140
        data_rows.extend(chunk.metadata['table_data']['rows'][2:])
        assert chunk.metadata['table_headers']==['Object','Metrics > Accuracy','Metrics > Recall']
        numbers=chunk.metadata['location']['row_numbers']
        assert numbers[:2]==[1,2]
        assert all(f'M{n-3}' in chunk.text for n in numbers[2:])
        if numbers[2]>3:
            assert chunk.metadata['location']['cell_range'].startswith('A1:C2,')
        for cell in chunk.metadata['semantic_cells']:
            assert cell['quote'] in chunk.text and cell['value'] in cell['quote']
    assert data_rows==doc.segments[0]['meta']['table_data']['rows'][2:]


def test_identical_tables_in_distinct_worksheets_not_deduplicated():
    doc=table_document()
    second=copy.deepcopy(doc.segments[0])
    second.update(id='second',section_path='Sheet: Other',reading_order=1)
    second['meta']['location']['worksheet']='Other'
    doc.segments.append(second)
    chunks=structured_chunks(doc)
    assert {c.metadata['location']['worksheet'] for c in chunks}=={'Metrics','Other'}


def test_semantic_embedding_view_is_explicit_and_original_quote_stays_same():
    doc=table_document()
    base=structured_chunks(doc,StructuredOptions(max_tokens=200,min_tokens=0,overlap_tokens=0))
    enhanced=structured_chunks(doc,StructuredOptions(max_tokens=200,min_tokens=0,overlap_tokens=0,semantic_retrieval=True))
    assert all(not c.metadata['retrieval_text'] for c in base)
    assert all(c.metadata['retrieval_text'] for c in enhanced)
    assert all(estimate_tokens(embedding_text(c.as_dict()))<=200 for c in enhanced)


def test_damaged_native_numeric_cells_flagged_without_inference():
    doc=Document.from_text('', 'native.pdf')
    doc.segments=[{'type':'table','content':'指标 | 原始 | 加权\n准确率 | 96 80% . | 93 00% .','page':7,'bbox':[1,2,3,4]}]
    chunks=structured_chunks(doc)
    assert chunks[0].metadata['needs_review']
    assert chunks[0].text.endswith('96 80% . | 93 00% .')
    assert any(w['code']=='UNRESOLVED_NUMERIC_FORMAT' for w in chunks[0].metadata['warnings'])


def test_only_consumed_metadata_affects_content_fingerprint():
    doc=table_document()
    original=canonical_content(doc)
    doc.segments[0]['meta']['unknown_runtime_counter']=42
    assert canonical_content(doc)==original
    doc.segments[0]['meta']['location']['worksheet']='Changed'
    assert canonical_content(doc)!=original


def test_catalog_metadata_uses_pinned_version_and_citations(tmp_path):
    service=setup(tmp_path)
    doc=table_document()
    chunks=structured_chunks(doc)
    service.catalog.publish(doc,chunks,'config1')
    with service.catalog.read_snapshot() as snapshot:
        rows=service.index.search('M1')
        enriched=service.catalog.attach_metadata(rows,snapshot)
        assert enriched and all(h['location']['worksheet']=='Metrics' for h in enriched)
        c=citations([{'citation_ids':[1],'supporting_quotes':['Object']}],enriched)[0]
        assert c['location']['worksheet']=='Metrics' and c['page']==0 and not c['bbox']
        damaged=copy.deepcopy(rows)
        damaged[0]['text']='tampered index payload'
        with pytest.raises(RuntimeError,match='INDEX_PAYLOAD_MISMATCH'):
            service.catalog.attach_metadata(damaged,snapshot)
