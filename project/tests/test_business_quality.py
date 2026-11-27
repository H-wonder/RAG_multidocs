"""Structural regressions; real source acceptance is recorded separately."""
from rag_multidocs.model import Document, normalize_hit
from rag_multidocs.ingestion.structured import structured_chunks, StructuredOptions
from rag_multidocs.qa.table_evidence import table_view
from rag_multidocs.qa.tasks import plan_tasks, synthesis_question


def test_page_ocr_tables_keep_cells_and_both_viewport_headers():
    doc = Document.from_text('', 'scan.pdf')
    rows = [['', 'A', 'B'], ['1', 'Item', 'Choice']]
    rows += [[str(i), 'Entry ' + str(i), 'Option ' + str(i)] for i in range(2, 22)]
    html = '<table>' + ''.join('<tr>' + ''.join('<td>' + c + '</td>' for c in r) + '</tr>' for r in rows) + '</table>'
    doc.segments = [{'type': 'page', 'content': 'Overview\n' + html + '\nEnd', 'page': 1}]
    chunks = structured_chunks(doc, StructuredOptions(max_tokens=100, min_tokens=0, overlap_tokens=0))
    tables = [c for c in chunks if c.content_type == 'table']
    assert len(tables) > 1
    assert all(c.metadata['table_data']['header_rows'] == 2 for c in tables)
    assert [r for c in tables for r in c.metadata['table_data']['rows'][2:]] == rows[2:]
    assert all('<td>' not in c.text for c in chunks)


def test_repeated_horizontal_schema_preserves_empty_cells_and_exact_quotes():
    doc = Document.from_text('', 'parallel.pdf')
    doc.segments = [{'type': 'table', 'content': '课程 | 学分 | 成绩 | 课程 | 学分 | 成绩\n甲课 | 2.5 | 87 | 乙课 | 1.0 | 92\n丙课 | 4.5 | 79 |  |  | ', 'page': 1}]
    chunks = structured_chunks(doc)
    hits = [normalize_hit({**c.as_dict(), 'chunk_id': c.id}) for c in chunks]
    view = table_view(hits[0])
    assert [(r['row_label'], r['cells']['学分'], r['cells']['成绩']) for r in view['rows']] == [('甲课', '2.5', '87'), ('乙课', '1.0', '92'), ('丙课', '4.5', '79')]
    assert all(r['supporting_quote'] in hits[0]['text'] for r in view['rows'])
    tasks = plan_tasks('甲课、乙课、丙课分别是多少分、多少学分？', hits)
    assert len(tasks) == 6
    assert {t['expected_value'] for t in tasks} == {'2.5', '87', '1.0', '92', '4.5', '79'}


def test_enumeration_is_not_forced_into_single_atomic_claim():
    assert synthesis_question('这份表格从哪些方面评价申请者？有哪些志愿选项？')


def test_short_native_lines_merge_without_crossing_section_or_page():
    doc = Document.from_text('', 'history.pdf')
    doc.segments = [{'type': 'text', 'content': t, 'page': 1, 'reading_order': i}
                    for i, t in enumerate(['一座建筑最初称为甲。', '后来改为乙。', '最终改称丙。'])]
    doc.segments.append({'type': 'text', 'content': '第二页。', 'page': 2})
    chunks = structured_chunks(doc, StructuredOptions(min_tokens=100))
    assert len(chunks) == 2
    assert '甲' in chunks[0].text and '乙' in chunks[0].text and '丙' in chunks[0].text
    assert chunks[1].page == 2


def test_schema_question_keeps_header_when_filename_matches_a_column():
    import json
    from rag_multidocs.qa.grounding import GroundedQA
    text = '姓名 | 沟通能力 | 志愿 | 出勤\n小甲 | 4.5 | 宣传 | 1'
    hit = {'chunk_id': 'row', 'doc_id': 'form', 'source': '出勤.pdf',
           'text': text, 'content_type': 'table',
           'table_headers': ['姓名', '沟通能力', '志愿', '出勤'],
           'table_data': {'header_rows': 1}}
    class Chat:
        def complete(self, messages):
            blocks = json.loads(messages[1]['content'])['evidence']
            fields = {key: value['field'] for b in blocks for key, value in b['lines'].items()}
            keys = [key for key, value in fields.items() if value == '沟通能力']
            assert keys, 'The model must actually receive the assessment field'
            return json.dumps({key: 'assessment' if key in keys else 'identity' for key in fields})
    result = GroundedQA(Chat()).generate('根据出勤.pdf，这份表格从哪些方面评价候选人？', [hit])
    assert result['status'] == 'answered'
    assert result['citations'][0]['excerpts'] == ['姓名 | 沟通能力 | 志愿 | 出勤']


def test_semantic_rewriter_accepts_only_a_complete_json_fence():
    import pytest
    from rag_multidocs.retrieval.rewrite import SemanticRewriter
    class Chat:
        def complete(self, messages):
            return '```json\n["SVM在2026年的准确率"]\n```'
    assert SemanticRewriter(Chat())('SVM在2026年的准确率？') == ['SVM在2026年的准确率']
    class Malformed(Chat):
        def complete(self, messages):
            return '额外说明\n```json\n["SVM在2026年的准确率"]\n```'
    with pytest.raises(ValueError):
        SemanticRewriter(Malformed())('SVM在2026年的准确率？')


def test_semantic_rewriter_unwraps_query_objects_without_relaxing_constraints():
    import pytest
    from rag_multidocs.retrieval.rewrite import SemanticRewriter
    class Chat:
        def complete(self, messages):
            return '```json\n[{"query":"SVM在2026年的准确率"}]\n```'
    assert SemanticRewriter(Chat())('SVM在2026年的准确率？') == ['SVM在2026年的准确率']
    class ChangedYear(Chat):
        def complete(self, messages):
            return '[{"query":"SVM在2025年的准确率"}]'
    with pytest.raises(ValueError):
        SemanticRewriter(ChangedYear())('SVM在2026年的准确率？')


def test_planned_section_does_not_support_completed_claim():
    import pytest
    from rag_multidocs.qa.grounding import validate_claim
    evidence = [{'text': '随后进行独立验证。', 'section_path': '后续计划'}]
    c = {'text': '已经完成独立验证。', 'citation_ids': [1], 'supporting_quotes': ['随后进行独立验证。']}
    with pytest.raises(ValueError, match='L3_PLANNED_AS_COMPLETED'):
        validate_claim(c, evidence)
    c['text'] = '后续安排进行独立验证。'
    assert validate_claim(c, evidence)
    evidence[0]['text'] = '已完成第一轮验证，后续扩展覆盖范围。'
    c.update(text='已完成第一轮验证。', supporting_quotes=['已完成第一轮验证，后续扩展覆盖范围。'])
    assert validate_claim(c, evidence)
