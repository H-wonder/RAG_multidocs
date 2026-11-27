from rag_multidocs.qa.llm import answer_with_evidence
from rag_multidocs.retrieval.controller import RetrievalController
from rag_multidocs.retrieval.profiles import recommended_policies

class Chat:
    def __init__(self, text):
        self.text=text
        self.calls=0
    def complete(self, messages):
        self.calls+=1
        assert '证据' in messages[0]['content'] and '问题：' in messages[1]['content']
        return self.text

def hit(key,text,doc='a',index=0,kind='text',**extra):
    return dict(chunk_id=key,doc_id=doc,source=doc+'.txt',text=text,chunk_index=index,content_type=kind,**extra)

def test_plain_multi_fact_uses_source_text_without_generated_quote_protocol():
    h=hit('a','成立年份：2019年\n开放时间：周二09:30-12:30')
    chat=Chat('2019年成立，周二09:30至12:30开放 [1]。')
    r=answer_with_evidence('哪年成立，何时开放？',[h],chat)
    assert r['status']=='answered' and chat.calls==1
    assert r['citations'][0]['excerpts']==[h['text']]
    assert r['generation_evidence']==[h]

def test_natural_numbers_are_not_deleted_by_a_token_whitelist():
    h=hit('a','存储桶 | 可用空间（GB）\n档案桶 | 96.50',kind='table',table_headers=['存储桶','可用空间（GB）'])
    raw='档案桶可用空间为96.50GB [1]。'
    r=answer_with_evidence('档案桶可用空间？',[h],Chat(raw))
    assert r['answer']==raw and r['attempts'][0]['raw_response']==raw
    assert not r['warnings'] and not r['fact_verified']

def test_table_header_units_and_target_entity_are_preserved():
    h=hit('a','项目 | 费用（元）\n桥梁 | 13.750\n路灯 | 4.250',kind='table',table_headers=['项目','费用（元）'])
    r=answer_with_evidence('路灯费用？',[h],Chat('不得调用'),structured_table_answers=True)
    assert r['status']=='answered' and '4.250' in r['answer'] and '13.750' not in r['answer']
    assert r['model_calls']==0

def test_literal_value_does_not_invent_an_additional_citation():
    rows=[hit('a','预约须提前3天。'),hit('b','负责人：乔舟\n地点：海岚站',doc='b')]
    raw='须找乔舟并提前3天预约 [1]。'
    r=answer_with_evidence('找谁预约？',rows,Chat(raw))
    assert r['answer']==raw and [c['index'] for c in r['citations']]==[1]
    assert not r['fact_verified']  # Valid source IDs are not semantic verification.

def test_unquoted_or_unknown_reference_does_not_silently_become_verified():
    h=hit('a','数量：15台')
    r=answer_with_evidence('数量？',[h],Chat('15台'))
    assert r['status']=='partial' and '15台' in r['answer']
    r=answer_with_evidence('数量？',[h],Chat('15台 [9]。'))
    assert r['status']=='partial' and not r['citations'] and '[9]' not in r['answer']

def test_complete_long_source_block_is_not_cut_at_1200_characters():
    text='材料说明。'*310+'截止日期为2027年3月8日。'
    h=hit('a',text)
    r=answer_with_evidence('截止日期？',[h],Chat('截止日期为2027年3月8日 [1]。'),context_tokens=16000,output_tokens=1024)
    assert r['status']=='answered' and r['generation_evidence'][0]['text']==text

def test_neighbor_context_restores_a_relevant_paragraph_under_the_same_budget():
    rows=[hit('h','月度报告',index=0),hit('done','已完成登记。',index=1),hit('plan','下月更换电源。',index=3),hit('other','其他站开放说明。',doc='b')]
    missing=hit('problem','读码器仍偶发掉线。',index=2)
    class Index:
        def search(self,*a,**kw):return rows
        def neighbors(self,*a,**kw):return rows[:3]+[missing]
    r=RetrievalController(Index(),policy=recommended_policies()['stable']).retrieve('概括本月工作、当前问题及下月计划。')
    assert 'problem' in {h['chunk_id'] for h in r['hits']}
    assert len(r['hits'])<=6 and r['context_tokens']<=3000


def test_no_evidence_reply_needs_no_synthetic_zero_reference():
    r=answer_with_evidence('未知容量？',[hit('a','地点：海岚站')],Chat('根据现有资料无法确定。 [0]'))
    assert r['status']=='insufficient_evidence' and r['answer']=='根据现有资料无法确定。'
    assert not r['citations'] and not r['warnings']


def test_existing_cell_reader_handles_empty_fields_without_model_guessing():
    h=hit('a','项目 | 费用（元） | 备注\n路灯 | 4.250 | ',kind='table',table_headers=['项目','费用（元）','备注'])
    class NoChat:
        def complete(self, messages):raise AssertionError('exact cells use the existing reader')
    r=answer_with_evidence('路灯的费用和备注分别是什么？',[h],NoChat(),structured_table_answers=True)
    assert r['status']=='answered' and '4.250' in r['answer'] and '空白' in r['answer']
    r=answer_with_evidence('路灯的费用和负责人分别是什么？',[h],Chat('费用为4.250元 [1]，负责人根据现有资料无法确定。'),structured_table_answers=True)
    assert r['status']=='partial' and r['model_calls']==1


def test_basic_retrieval_does_not_discard_a_ranked_hit_for_zero_lexical_overlap():
    from rag_multidocs.retrieval.controller import select_context
    h=hit('a','甲 | 3\n乙 | 8',support_score=0,selection_relevance=1)
    selected,_,_=select_context([h],[],{'context_k':4,'context_tokens':2000},recommended_policies()['stable'])
    assert selected==[h]
