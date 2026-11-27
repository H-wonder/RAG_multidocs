import json
from dataclasses import replace

import pytest

from rag_multidocs.connectors.registry import ConnectorRegistry
from rag_multidocs.connectors.txt import TxtConnector
from rag_multidocs.core import RAGCore
from rag_multidocs.index import KeywordIndex
from rag_multidocs.qa.grounding import GroundedQA, _numbers, validate_claim
from rag_multidocs.retrieval.controller import RetrievalPolicy, select_context
from rag_multidocs.retrieval.sources import resolve_sources
from rag_multidocs.qa.tasks import plan_tasks


def claim(text, quote, index=1):
    return {'text': text, 'citation_ids': [index], 'supporting_quotes': [quote]}


def hit(did, source, text):
    return {'chunk_id': did + '-chunk', 'doc_id': did, 'source': source, 'text': text}


@pytest.mark.parametrize('query,expected,code', [
    ('a.pdf金额？', ['a'], None),
    ('multi_a.pdf金额？', ['b'], None),
    ('a.pdf与multi_a.pdf比较金额', ['a', 'b'], None),
    ('《指南甲》保修期？', ['a'], None),
    ('文件missing.xyz金额？', None, 'SOURCE_NOT_FOUND'),
    ('不存在.pdf金额？', None, 'SOURCE_NOT_FOUND'),
    ('在a.pdf中金额？', ['a'], None),
    ('《不存在》金额？', None, 'SOURCE_NOT_FOUND'),
    ('金额是多少？', None, None),
])
def test_exact_source_identity_and_unknown_fail_closed(query, expected, code):
    docs = [{'doc_id': 'a', 'source': 'a.pdf', 'title': '指南甲'}, {'doc_id': 'b', 'source': 'multi_a.pdf'}]
    result = resolve_sources(query, docs)
    assert result['doc_ids'] == expected
    assert result['blocked'] == bool(code)
    if code:
        assert result['warnings'][0]['code'] == code


def test_duplicate_basename_different_identities_is_not_silently_selected():
    docs = [{'doc_id': 'a', 'source': 'a/同名.pdf'}, {'doc_id': 'b', 'source': 'b/同名.pdf'}]
    result = resolve_sources('同名.pdf中的金额', docs)
    assert result['blocked'] and result['warnings'][0]['code'] == 'SOURCE_AMBIGUOUS'
    assert resolve_sources('a/同名.pdf中的金额', docs)['doc_ids'] == ['a']


def test_explicit_api_scope_is_intersected_never_overridden():
    result = resolve_sources('a.pdf金额？', [{'doc_id': 'a', 'source': 'a.pdf'}], ['b'])
    assert result['blocked'] and result['doc_ids'] == []


def test_known_ascii_name_cannot_be_taken_from_another_chinese_filename():
    result = resolve_sources('不存在a.pdf金额？', [{'doc_id': 'a', 'source': 'a.pdf'}])
    assert result['blocked'] and result['warnings'][0]['code'] == 'SOURCE_NAME_BOUNDARY_UNCERTAIN'


@pytest.mark.parametrize('query', ['model.fit方法如何使用？', 'client.search(query)是什么意思？', 'np.array怎么用？'])
def test_ordinary_dotted_api_names_are_not_unknown_files(query):
    result = resolve_sources(query, [{'doc_id': 'a', 'source': 'notes.md'}])
    assert not result['explicit'] and not result['blocked'] and result['doc_ids'] is None


def test_two_named_sources_reserve_context_even_when_nearly_identical():
    rows = [dict(hit('a', 'a.pdf', 'Warranty: 36 months.'), selection_relevance=1, support_score=1),
            dict(hit('a', 'a.pdf', 'Warranty: 36 months.'), chunk_id='a-other', selection_relevance=.9, support_score=1),
            dict(hit('b', 'b.pdf', 'Warranty: 36 months.'), selection_relevance=.1, support_score=1)]
    chosen, _, _ = select_context(rows, [], {'context_k': 2, 'context_tokens': 1000}, RetrievalPolicy(), ['a', 'b'])
    assert {h['doc_id'] for h in chosen} == {'a', 'b'}


def test_scope_survives_recall_neighbors_and_active_version(tmp_path):
    registry = ConnectorRegistry()
    registry.register(TxtConnector())
    core = RAGCore(registry, KeywordIndex(path=tmp_path/'index.json'),
                   policy=replace(RetrievalPolicy(), reranker='none', min_relevance=0))
    try:
        a, b = tmp_path/'one.txt', tmp_path/'two.txt'
        a.write_text('Amount: 17 CNY. Warranty period: 12 months.')
        b.write_text('Amount: 31 CNY. Warranty period: 48 months.')
        da = core.ingest('txt', str(a)).documents[0]['doc_id']
        core.ingest('txt', str(b))
        before = core.search('one.txt的Amount和缺失指标？')
        assert before['source_constraints']['doc_ids'] == [da]
        assert all(h['doc_id'] == da for t in before['retrieval_trace'] for h in t['candidates'])
        assert all(t['source_doc_ids'] == [da] for t in before['retrieval_trace'])
        a.write_text('Amount: 29 CNY. Warranty period: 18 months.')
        core.ingest('txt', str(a))
        after = core.search('one.txt的Amount？')
        assert after['hits'] and all(h['version'] == 2 and h['doc_id'] == da for h in after['hits'])
        blocked = core.search('absent.txt金额？')
        assert not blocked['hits'] and blocked['index_calls'] == 0
        assert core.search('Amount')['source_constraints']['explicit'] is False
    finally:
        core.close()


@pytest.mark.parametrize('second', ['Amount', 'Amount: 17 CNY', 'What is the amount?'])
def test_second_task_cannot_reuse_other_source_or_omit_value(second):
    rows = [hit('a', 'one.pdf', 'Amount: 17 CNY'), hit('b', 'two.pdf', 'Amount: 31 CNY')]
    class Chat:
        def complete(self, messages):
            payload = json.loads(messages[1]['content'])
            q = payload['question']
            if 'one.pdf' in q:
                return json.dumps({'status': 'answered', 'claims': [claim('Amount: 17 CNY', 'Amount: 17 CNY')]})
            # Even a true quote from source A cannot answer source B.
            return json.dumps({'status': 'answered', 'claims': [claim(second, 'Amount: 17 CNY')]})
    result = GroundedQA(Chat(), max_calls=2).generate('one.pdf金额是多少？two.pdf金额是多少？', rows)
    assert result['status'] == 'partial'
    assert '17' in result['answer'] and '31' not in result['answer']
    assert len({t['task_id'] for t in result['task_requirements']}) == 2
    assert result['unanswered_subquestions']


def test_two_sources_correct_values_and_actual_prompt_binding():
    rows = [hit('a', 'one.pdf', 'Amount: 17 CNY'), hit('b', 'two.pdf', 'Amount: 31 CNY')]
    class Chat:
        def complete(self, messages):
            payload = json.loads(messages[1]['content'])
            assert len(payload['evidence']) == 1
            e = payload['evidence'][0]
            return json.dumps({'status': 'answered', 'claims': [claim(e['text'], e['text'], e['index'])]})
    result = GroundedQA(Chat(), max_calls=2).generate('one.pdf金额是多少？two.pdf金额是多少？', rows)
    assert result['status'] == 'answered' and len(result['claims']) == 2
    assert all(validate_claim(c, result['answer_evidence']) for c in result['claims'])
    assert all(len(json.loads(a['messages'][1]['content'])['evidence']) == 1 for a in result['attempts'])


def test_comparison_without_question_marks_still_has_per_source_tasks():
    rows = [hit('a', 'a.pdf', 'Warranty period: 12 months'), hit('b', 'b.pdf', 'Warranty period: 48 months')]
    scope = resolve_sources('比较a.pdf与b.pdf的保修期', [{'doc_id': h['doc_id'], 'source': h['source']} for h in rows])
    tasks = plan_tasks('比较a.pdf与b.pdf的保修期', rows, scope)
    assert len(tasks) == 2 and {tuple(t['required_doc_ids']) for t in tasks} == {('a',), ('b',)}
    assert {t['expected_value'] for t in tasks} == {'12 months', '48 months'}


@pytest.mark.parametrize('value', ['80.00%', '125.50 CNY', '7', '36 months', '5 percentage points'])
def test_explicit_structured_field_extraction_preserves_literal_units(value):
    rows = [dict(hit('t', 'values.xlsx', f'Object | Metric\nDevice | {value}'), content_type='table',
                 location={'worksheet': 'S', 'cell_range': 'A1:B2'})]
    class NoModel:
        def complete(self, _):
            raise AssertionError('Deterministic extraction must not masquerade as model generation')
    result = GroundedQA(NoModel(), structured_table_answers=True).generate('Device的Metric是多少？', rows)
    assert result['status'] == 'answered' and value in result['answer']
    assert result['answer_mode'] == 'structured_extraction' and not result['used_llm']
    assert result['model_calls'] == 0 and result['generation_evidence'] == []
    assert result['extractions'][0]['supporting_quote'] == f'Device | {value}'
    assert validate_claim(result['claims'][0], result['answer_evidence'])


def test_percentage_points_are_not_percent():
    assert _numbers('5 percentage points') == _numbers('5百分点')
    assert _numbers('5%') != _numbers('5百分点')


def test_same_numeric_value_different_units_still_rejected():
    data = [hit('a', 'a.txt', 'Amount: 30 CNY')]
    with pytest.raises(ValueError, match='UNIT'):
        validate_claim(claim('Amount: 30%', 'Amount: 30 CNY'), data)


@pytest.mark.parametrize('raw,formatted,formula,cached', [
    (0.8, '80.00%', None, None), (80, '80', None, None),
    (30, '30 CNY', None, None), ('=A2*2', '60', '=A2*2', 60),
])
def test_original_cell_format_cache_and_address_are_exposed(raw, formatted, formula, cached):
    row = dict(hit('x', 'x.xlsx', f'Object | Metric\nDevice | {formatted}'), content_type='table',
               table_headers=['Object', 'Metric'],
               location={'worksheet': 'S', 'cell_range': 'A1:B2', 'column_start': 1},
               semantic_cells=[{'row': 'Device', 'column': 'Metric', 'value': formatted,
                                'quote': f'Device | {formatted}', 'source_row': 2}],
               table_data={'header_rows': 1, 'cells': [{'address': 'B2', 'value': raw,
                   'display': formatted, 'number_format': '0.00%' if raw == .8 else 'General',
                   'formula': formula, 'cached_value': cached}]})
    task = plan_tasks('Device的Metric？', [row])[0]
    assert task['target_cell']['source_cell']['address'] == 'B2'
    assert task['target_cell']['source_cell']['value'] == raw
    result = GroundedQA(None, structured_table_answers=True).generate('Device的Metric？', [row])
    assert result['status'] == 'answered' and formatted in result['answer']
    assert result['extractions'][0]['source_cell']['cached_value'] == cached


def test_uncached_formula_is_never_executed_by_structured_answer():
    row = dict(hit('x', 'x.xlsx', 'Object | Metric\nDevice | =A2*2'), content_type='table')
    result = GroundedQA(None, structured_table_answers=True).generate('Device的Metric？', [row])
    assert result.get('answer_mode') != 'structured_extraction'
    assert result['status'] != 'answered'


def test_two_objects_repeating_first_value_is_partial_even_without_sources():
    rows = [hit('a', 'all.txt', 'Device A Amount: 17 CNY\nDevice B Amount: 31 CNY')]
    class Chat:
        def complete(self, _):
            return json.dumps({'status': 'answered', 'claims': [claim('Device A Amount: 17 CNY', 'Device A Amount: 17 CNY')]})
    result = GroundedQA(Chat(), max_calls=2).generate('Device A的Amount是多少？Device B的Amount是多少？', rows)
    assert result['status'] == 'partial' and len(result['claims']) == 1


def test_valid_first_answer_survives_later_model_and_repair_failure():
    rows = [hit('a', 'one.pdf', 'Amount: 17 CNY'), hit('b', 'two.pdf', 'Amount: 31 CNY')]
    class Chat:
        calls = 0
        def complete(self, _):
            self.calls += 1
            if self.calls > 1:
                raise RuntimeError('provider failed')
            return json.dumps({'status': 'answered', 'claims': [claim('17 CNY', 'Amount: 17 CNY')]})
    result = GroundedQA(Chat(), max_calls=3).generate('one.pdf金额？two.pdf金额？', rows)
    assert result['status'] == 'partial' and '17 CNY' in result['answer']
    assert result['model_calls'] == 3 and len(result['claims']) == 1


def test_known_source_does_not_make_a_missing_metric_available():
    rows = [hit('a', 'one.pdf', 'Accuracy: 93.5%')]
    result = GroundedQA(None).generate('one.pdf中的税率是多少？', rows)
    assert result['status'] == 'insufficient_evidence' and result['model_calls'] == 0


def test_source_task_citations_reindex_against_actual_answer_evidence():
    rows = [hit('unused', 'unused.pdf', 'Unused evidence.'), hit('a', 'one.pdf', 'Amount: 17 CNY')]
    class Chat:
        def complete(self, messages):
            e = json.loads(messages[1]['content'])['evidence'][0]
            assert e['index'] == 2
            return json.dumps({'status': 'answered', 'claims': [claim('17 CNY', e['text'], 2)]})
    result = GroundedQA(Chat()).generate('one.pdf金额？', rows)
    assert result['status'] == 'answered' and result['claims'][0]['citation_ids'] == [1]
    assert result['generation_evidence'] == result['answer_evidence'] == [rows[1]]
    assert validate_claim(result['claims'][0], result['answer_evidence'])


def test_reported_operands_are_not_a_completed_sum_when_units_conflict():
    rows = [dict(hit('x', 'x.xlsx', 'Object | Amount\nDevice A | 30 CNY\nDevice B | 30 USD'), content_type='table')]
    result = GroundedQA(None, structured_table_answers=True, allow_calculation=True).generate('Device A和Device B的Amount合计', rows)
    assert result['status'] == 'partial'
    assert len(result['claims']) == 2 and not result.get('calculations')
    assert any(w['code'] == 'CALCULATION_UNAVAILABLE' for w in result['warnings'])


def test_identical_numbers_and_identical_quotes_do_not_erase_source_identity():
    rows = [hit('a', 'one.pdf', 'Amount: 17 CNY'), hit('b', 'two.pdf', 'Amount: 17 CNY')]
    class Chat:
        def complete(self, messages):
            # Text and value are true in both documents. A citation to source A
            # still cannot complete the task bound to source B.
            return json.dumps({'status': 'answered', 'claims': [claim('17 CNY', 'Amount: 17 CNY', 1)]})
    result = GroundedQA(Chat(), max_calls=2).generate('one.pdf金额？two.pdf金额？', rows)
    assert result['status'] == 'partial' and len(result['claims']) == 1
    assert result['subquestion_results'][1]['status'] != 'answered'
    assert any(w['code'] == 'L1_TASK_EVIDENCE_MISMATCH' for w in result['warnings'])


@pytest.mark.parametrize('amount,fee,hours', [('173.25', '6', '18'), ('812.40', '0', '12')])
def test_fee_is_not_total_amount_and_daily_duration_keeps_its_unit(amount, fee, hours):
    rows = [hit('invoice', 'invoice.pdf', f'金额 Amount: {amount} CNY\n费用 Fee: {fee} CNY\n设备每天{hours}小时可用。')]
    class Chat:
        def complete(self, messages):
            q = json.loads(messages[1]['content'])['question']
            if '每天' in q:
                c = claim(f'{hours}小时', f'设备每天{hours}小时可用。')
            else:
                c = claim(f'{fee} CNY', f'费用 Fee: {fee} CNY')
            return json.dumps({'status': 'answered', 'claims': [c]})
    result = GroundedQA(Chat(), max_calls=2).generate('invoice.pdf设备每天开放多久？费用是多少？', rows)
    assert result['status'] == 'answered'
    assert {hours+'hour', fee+'cny'} <= _numbers(result['answer'])
    assert [t['expected_value'] for t in result['task_requirements']] == [hours+'小时', fee+' CNY']
    assert result['model_calls'] == 2


def test_correctly_quoted_total_amount_cannot_complete_the_fee_task():
    rows = [hit('invoice', 'invoice.pdf', 'Amount: 812.40 CNY\nFee: 3 CNY')]
    class Chat:
        def complete(self, _):
            return json.dumps({'status': 'answered', 'claims': [claim('812.40 CNY', 'Amount: 812.40 CNY')]})
    result = GroundedQA(Chat()).generate('invoice.pdf的费用是多少？', rows)
    assert result['status'] != 'answered' and not result.get('claims')


def test_global_total_is_not_a_named_items_unmapped_value():
    from rag_multidocs.qa.tasks import finite_value
    assert finite_value('Screw的Amount是多少？', [hit('x', 'x.pdf', 'Amount: 812.40 CNY\nScrew 5 20.50 CNY')]) is None
