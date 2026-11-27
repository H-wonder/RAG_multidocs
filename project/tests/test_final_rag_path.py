"""Confirmed shared-path regressions; independent fictional source material."""
from rag_multidocs.qa.llm import answer_with_evidence
from rag_multidocs.execution import request_scope, record
from rag_multidocs.retrieval.controller import RetrievalController, RetrievalPolicy
from rag_multidocs.retrieval.sources import resolve_sources

class Chat:
    def __init__(self, raw): self.raw, self.calls = raw, 0
    def complete(self, messages):
        self.calls += 1
        record('transport_attempts')
        return self.raw

def table():
    return {'chunk_id':'t', 'doc_id':'d', 'source':'维护.csv', 'content_type':'table',
            'text':'站点 | 维护时长（天）\n东闸 | 23.50\n西闸 | 6.25',
            'table_headers':['站点', '维护时长（天）']}

def test_field_dictionary_cannot_convert_a_question_to_an_explicit_source():
    rows=[{'chunk_id':'a','doc_id':'a','source':'观测.txt','text':'宁渡站由叶岚负责。周五开放。'},
          {'chunk_id':'b','doc_id':'b','source':'词典.txt','text':'负责人：岗位名称\n开放时间：字段说明'}]
    class Index:
        def search(self, q, **kw):
            return [h for h in rows if kw.get('doc_ids') is None or h['doc_id'] in kw['doc_ids']]
    r=RetrievalController(Index(),policy=RetrievalPolicy.baseline()).retrieve('宁渡站的负责人和开放时间是什么？')
    assert not r['source_constraints']['explicit']
    assert {h['doc_id'] for h in r['hits']} == {'a','b'}
    scope=resolve_sources('根据词典.txt，负责人是什么？', rows, doc_ids=['a'])
    assert scope['blocked'] and scope['warnings'][0]['code']=='SOURCE_SCOPE_CONFLICT'

def test_mapped_operands_reach_decimal_calculation_instead_of_early_extraction():
    chat=Chat('不得调用')
    with request_scope() as budget:
        r=answer_with_evidence('东闸和西闸的维护时长合计是多少？',[table()],chat,
                               allow_calculation=True,structured_table_answers=True)
    assert r['answer_mode']=='structured_calculation' and r['calculations'][0]['result']=='29.75'
    assert '29.75 天' in r['answer'] and r['citations'][0]['index']==1
    assert chat.calls==budget.counters['generation_calls']==budget.counters['transport_attempts']==0

def test_existing_explicit_binary_operation_remains_available():
    h={'chunk_id':'a','source':'计量.txt','text':'观测值：31\n校准值：4'}
    r=answer_with_evidence('请计算31除以4的结果。',[h],Chat('不得调用'),allow_calculation=True)
    assert r['answer_mode']=='structured_calculation' and r['calculations'][0]['result']=='7.75'

def test_unimplemented_table_operation_does_not_report_operands_as_completed_math():
    r=answer_with_evidence('东闸和西闸的维护时长乘以后的结果是多少？',[table()],Chat('不得调用'),
                           allow_calculation=True,structured_table_answers=True)
    assert r['status']=='partial' and not r.get('calculations')
    assert r['warnings'][0]['code']=='CALCULATION_NOT_PERFORMED'

def test_missing_field_falls_back_and_negative_fact_is_not_uncertainty():
    chat=Chat('维护时长6.25天 [1]，负责人根据现有资料无法确定。')
    r=answer_with_evidence('西闸的维护时长和负责人是什么？',[table()],chat,structured_table_answers=True)
    assert chat.calls==1 and r['status']=='partial'
    r=answer_with_evidence('提供在线申请吗？',[{'chunk_id':'a','source':'申请.txt','text':'该站未提供在线申请服务。'}],Chat('该站未提供在线申请服务 [1]。'))
    assert r['status']=='answered'

def test_natural_generation_is_counted_once_and_raw_numbers_are_not_deleted():
    h={'chunk_id':'a','source':'存储.txt','text':'可用空间96.50GB，日用量范围12至19GB。'}
    raw='可用空间96.50GB，日用量范围12至19GB [1]。'
    with request_scope() as budget:
        r=answer_with_evidence('空间与日用量范围？',[h],Chat(raw))
    assert r['answer']==raw and budget.counters['generation_calls']==budget.counters['transport_attempts']==1


def test_oversized_early_chunk_does_not_prevent_a_later_complete_evidence_block():
    rows=[{'chunk_id':'a','source':'旁注.txt','text':'无关说明。'*600},
          {'chunk_id':'b','source':'开放.txt','text':'宁渡站周五开放。'}]
    r=answer_with_evidence('宁渡站何时开放？',rows,Chat('周五开放 [1]。'),max_evidence_chars=500)
    assert [h['chunk_id'] for h in r['generation_evidence']]==['b']
    assert r['status']=='answered' and r['citations'][0]['source']=='开放.txt'


def test_uncalibrated_support_score_does_not_veto_an_experimental_candidate():
    from rag_multidocs.retrieval.controller import select_context
    hit={'chunk_id':'a','doc_id':'a','source':'测量.csv','text':'甲 | 3\n乙 | 8',
         'support_score':0,'selection_relevance':1}
    rows,_,tokens=select_context([hit],[],{'context_k':4,'context_tokens':2000},RetrievalPolicy())
    assert rows==[hit] and tokens<=2000


def test_source_prefix_is_not_counted_as_an_additional_requested_table_field():
    chat=Chat('不得调用')
    r=answer_with_evidence('根据维护.csv，西闸维护时长是多少？',[table()],chat,structured_table_answers=True)
    assert r['answer_mode']=='structured_extraction' and chat.calls==0 and '6.25' in r['answer']


def test_multiple_rows_cannot_mask_a_missing_requested_column():
    chat=Chat('维护时长见表 [1]，负责人根据现有资料无法确定。')
    r=answer_with_evidence('东闸和西闸的维护时长和负责人分别是什么？',[table()],chat,structured_table_answers=True)
    assert r['answer_mode']=='natural_language' and r['status']=='partial' and chat.calls==1
