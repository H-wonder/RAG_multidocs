from rag_multidocs.retrieval.controller import comparison_facets, select_context, RetrievalPolicy, RetrievalController
import pytest

from rag_multidocs.qa.grounding import GroundedQA, normalize_line_refs
from rag_multidocs.qa.tasks import enumeration_task, plan_tasks
from rag_multidocs.retrieval.sources import resolve_sources
from rag_multidocs.retrieval.rewrite import question_parts


def test_comparison_facets_reuse_source_alias_and_preserve_numeric_constraints():
    rows = [{'section_path': '一、和光台/礼仪台/旧台 > 典礼', 'text': '礼仪台举行仪式。礼仪台作为起点。'}]
    assert comparison_facets('分析和光台与内院在明代时期迎宾中的不同作用，并说明文化意义。', rows) == ['和光台 礼仪台 迎宾', '内院 迎宾']
    assert comparison_facets('比较甲组和乙组在2026年救援中的各自作用？', []) == ['甲组 2026年救援 2026年', '乙组 2026年救援 2026年']
    assert comparison_facets('根据历史说明如何迎宾。', rows) == []


def test_descriptive_file_suffix_does_not_hard_filter_normal_retrieval():
    docs = [{'doc_id': 'a', 'source': '编号-机构-维修单.pdf'}, {'doc_id': 'b', 'source': '课程说明.pdf'}]
    result = resolve_sources('根据完整维修单文件，合计多少钱？', docs)
    assert not result['explicit'] and result['doc_ids'] is None and not result['blocked']
    exact = resolve_sources('根据《编号-机构-维修单》，合计多少钱？', docs)
    assert exact['explicit'] and exact['doc_ids'] == ['a']
    assert question_parts('根据     ，保修期是多少年，费用是多少元？') == ['保修期是多少年', '费用是多少元']


def test_factual_tasks_keep_covered_source_and_ignore_an_unrelated_table():
    rows = [{'chunk_id': 'facts', 'doc_id': 'a', 'text': '保修期：两年。费用：70元。',
             'context_coverage_groups': ['a:0:0', 'a:0:1']},
            {'chunk_id': 'other', 'doc_id': 'a', 'text': '型号 | 费用\n乙 | 20', 'content_type': 'table'}]
    tasks = plan_tasks('保修期是多少年，费用是多少元？', rows)
    assert len(tasks) == 1 and tasks[0]['factual_question']
    assert tasks[0]['source_ids'] == ['facts']


def test_source_facet_evidence_precedes_generic_high_ranked_rows_with_same_budget():
    def hit(key, doc, rel, groups=()):
        return {'chunk_id': key, 'doc_id': doc, 'source': doc, 'text': key, 'support_score': 1,
                'selection_relevance': rel, 'context_coverage_groups': list(groups)}
    rows = [hit('general_a', 'a', 1), hit('general_b', 'b', .99),
            hit('a_first', 'a', .8, ['a:0']), hit('a_second', 'a', .7, ['a:1']),
            hit('b_first', 'b', .4, ['b:0']), hit('b_second', 'b', .3, ['b:1'])]
    picked, _, used = select_context(rows, [], {'context_k': 4, 'context_tokens': 1000}, RetrievalPolicy(), ['a','b'])
    assert {h['chunk_id'] for h in picked} == {'a_first', 'a_second', 'b_first', 'b_second'}
    assert used <= 1000
    rows[-1]['text'] = '过大证据' * 2000
    picked, _, used = select_context(rows, [], {'context_k': 4, 'context_tokens': 1000}, RetrievalPolicy(), ['a','b'])
    assert 'b_second' not in {h['chunk_id'] for h in picked} and used <= 1000


def test_reference_normalization_retains_source_identity_and_rejects_ambiguous_ids():
    evidence = [{'chunk_id': 'alpha'}, {'chunk_id': 'beta'}]
    lines = {'E1': (1, '原文甲。'), 'E2': (2, '原文乙。')}
    assert normalize_line_refs(['alpha.E1', 'E2:E2'], lines, evidence) == ['E1', 'E2']
    assert normalize_line_refs(['alpha', 'E1'], lines, evidence) == ['E1']
    assert normalize_line_refs('E1', lines, evidence) == ['E1']
    for refs in [['alpha'], ['beta.E1'], ['E1:E2'], ['E9'], [1], ['beta', 'E1'], 'E1,E2']:
        with pytest.raises(ValueError, match='L1_INVALID_EVIDENCE_LINE'):
            normalize_line_refs(refs, lines, evidence)


def test_category_enumeration_merges_actual_cells_without_turning_comments_into_options():
    rows = [{'chunk_id': 'a', 'content_type': 'table', 'table_headers': ['姓名', '志愿'],
             'text': '姓名 | 志愿\n小甲 | 宣传/设计，我希望进一步了解\n小乙 | 摄影；宣传'}]
    result = enumeration_task('有哪些志愿选项？', rows)
    assert list(result['items']) == ['宣传', '设计', '摄影']
    assert all(v['quote'] in rows[0]['text'] for v in result['items'].values())
    assert enumeration_task('小甲的志愿是什么？', rows) is None


def test_recommendation_name_selection_is_bound_to_one_original_row_and_entity():
    import json
    row = {'chunk_id': 'a', 'doc_id': 'doc', 'source': '资源清单.txt', 'content_type': 'table',
           'table_headers': ['平台名称', '主管机构', '认证级别'],
           'text': '平台名称 | 主管机构 | 认证级别\n接口实验台 | 工程组 | 实验级\n运行平台 | 服务组 | 生产级'}
    class Chat:
        def complete(self, messages):
            records = json.loads(messages[1]['content'])['records']
            assert set(records) == {'接口实验台'}
            return json.dumps({'接口实验台': '可用于练习接口验证。'}, ensure_ascii=False)
    result = GroundedQA(Chat(), max_calls=1).generate('希望选择实验级的平台练习接口验证，请推荐并说明主管机构。', [row])
    assert result['status'] == 'answered'
    assert '接口实验台' in result['answer'] and '工程组' in result['answer']
    assert result['citations'][0]['excerpts'] == ['接口实验台 | 工程组 | 实验级']
    class WrongEntity(Chat):
        def complete(self, messages):
            return json.dumps({'接口实验台': '推荐运行平台，适合练习。'}, ensure_ascii=False)
    result = GroundedQA(WrongEntity(), max_calls=1).generate('希望选择实验级的平台练习接口验证，请推荐。', [row])
    assert result['status'] == 'generation_failed'
    assert any(w['code'] == 'L3_RECOMMENDATION_ENTITY_MISMATCH' for w in result['warnings'])
    class EmbeddedWrongEntity(Chat):
        def complete(self, messages):
            return json.dumps({'接口实验台': '其优势来自运行平台的接口能力。'}, ensure_ascii=False)
    result = GroundedQA(EmbeddedWrongEntity(), max_calls=1).generate('希望选择实验级的平台练习接口验证，请推荐。', [row])
    assert result['status'] == 'generation_failed'


def test_focused_source_routes_survive_fusion_and_precede_optional_rewrites():
    from test_a_plus_retrieval import h, NoRerank
    class Index:
        def __init__(self):
            self.calls = []
        def search(self, query, **kwargs):
            self.calls.append((query, kwargs.get('doc_ids')))
            scope = kwargs.get('doc_ids')
            if len(scope) == 1:
                did = scope[0]
                return [h(f'{did}-focused-{i}', '和光台用于迎宾。', source=did) for i in range(20)]
            return [h(f'general-{i}', '和光台与内院的总体介绍。', source='a' if i < 30 else 'b') for i in range(48)]
    index = Index()
    policy = RetrievalPolicy(candidate_k=48, reranker='none', use_neighbors=False)
    result = RetrievalController(index, NoRerank(), policy, rewriter=lambda _: ['和光台迎宾', '内院迎宾']).retrieve(
        '比较和光台与内院在迎宾中的不同作用，并说明文化意义。', doc_ids=['a', 'b'])
    assert index.calls[1][1] in [['a'], ['b']] and index.calls[2][1] in [['a'], ['b']]
    assert index.calls[1][1] != index.calls[2][1]
    base_ids = {h['chunk_id'] for h in result['retrieval_trace'][0]['base_candidates']}
    assert {'a-focused-10', 'b-focused-10'} <= base_ids
    assert result['index_calls'] <= 4 and len(base_ids) <= 48


def test_parallel_summary_delimiters_and_source_scope_preserve_all_aspects():
    from rag_multidocs.retrieval.rewrite import summary_parts
    from rag_multidocs.retrieval.sources import without_source_names
    scope=resolve_sources('请总结《编号项目报告》的主要内容和观点。',[{'doc_id':'a','source':'编号项目报告.docx'}])
    assert scope['doc_ids']==['a']
    assert summary_parts(without_source_names(scope['original_query'],scope))==('的主要内容','观点')
    assert summary_parts('总结项目已完成的工作、遇到的问题及下一步计划。')==('项目已完成的工作','遇到的问题','下一步计划')
    assert RetrievalPolicy.baseline().resources('总结项目进展及遇到的问题。')['context_k']==6
    assert RetrievalPolicy(context_tokens=900,context_k=2).resources('总结项目进展及遇到的问题。')['context_tokens']==900


def test_literal_bullet_enumeration_keeps_continuations_limits_and_other_questions():
    rows=[{'chunk_id':'list','doc_id':'d','source':'check.md','text':'## 器材借用\n- 不超过4人\n- 要求：(1)登记\n(2) 归还检查\n- 时间：周二闭馆前归还'}]
    value=enumeration_task('器材借用有哪些具体要求？',rows)
    assert len(value['items'])==3 and '(2) 归还检查' in list(value['items'])[1]
    assert all(v['quote'] in rows[0]['text'] for v in value['items'].values())
    assert enumeration_task('场馆参观有哪些要求？',rows) is None
    assert enumeration_task('器材借用有哪些具体要求？',rows+[dict(rows[0],chunk_id='ambiguous')]) is None
    class Chat:
        def complete(self,m):raise AssertionError('literal source list needs no generation')
    r=GroundedQA(Chat(),structured_table_answers=True).generate('器材借用有哪些具体要求？',rows)
    assert r['status']=='answered' and '不超过4人' in r['answer'] and '归还检查' in r['answer'] and '周二闭馆前归还' in r['answer']
