import copy
import json

import pytest

from rag_multidocs.model import Document, normalize_hit
from rag_multidocs.ingestion.structured import structured_chunks
from rag_multidocs.qa.grounding import GroundedQA, validate_claim, _numbers
from rag_multidocs.qa.calculations import calculate
from rag_multidocs.qa.table_evidence import table_view
from test_a_plus_grounding import claim


def table_evidence():
    doc=Document.from_text('', 'measurements.xlsx')
    rows=[['Model','Accuracy','Recall'],['LogReg','90%','80%'],['SVM','95%','85%']]
    doc.segments=[{'id':'table','type':'table','content':'\n'.join(' | '.join(row) for row in rows),'section_path':'Sheet: Models','meta':{'location':{'worksheet':'Models','cell_range':'A1:C3','row_numbers':[1,2,3],'column_start':1},'table_data':{'rows':rows,'header_rows':1}}}]
    chunks=structured_chunks(doc)
    result=[]
    for c in chunks:
        hit=c.as_dict()
        hit['chunk_id']=hit['id']
        result.append(normalize_hit(hit))
    return result


def test_four_atomic_row_metric_answers_and_real_location_in_prompt():
    values={('LogReg','Accuracy'):'90%',('LogReg','Recall'):'80%',('SVM','Accuracy'):'95%',('SVM','Recall'):'85%'}
    class Chat:
        calls=0
        def complete(self,messages):
            self.calls+=1
            payload=json.loads(messages[1]['content'])
            assert payload['evidence'][0]['location']['worksheet']=='Models'
            suffix=payload['question'].rsplit('（',1)[-1]
            for (obj,metric),value in values.items():
                if f'“{obj}”行' in suffix and f'“{metric}”列' in suffix:
                    quote='LogReg | 90% | 80%' if obj=='LogReg' else 'SVM | 95% | 85%'
                    target = payload['mapped_table_target']
                    assert (target['row'], target['column'], target['value']) == (obj, metric, value)
                    assert target['supporting_quote'] == quote
                    assert target['source_chunk_id'] == payload['evidence'][0]['chunk_id']
                    return json.dumps({'status':'answered','claims':[claim(f'{obj} {metric} {value}',quote)]})
            raise AssertionError(payload['question'])
    chat=Chat()
    result=GroundedQA(chat,max_calls=4,atomic_single=True).generate('比较LogReg和SVM的Accuracy和Recall。',table_evidence())
    assert chat.calls==4 and result['status']=='answered' and not result['unanswered_subquestions']
    assert len(result['claims'])==4
    assert result['citations'][0]['location']['cell_range']=='A1:C3'
    assert all(validate_claim(c,result['generation_evidence']) for c in result['claims'])


@pytest.mark.parametrize('wrong', ['LogReg Recall', 'LogReg Accuracy 90%'])
def test_mapped_cell_cannot_be_completed_by_label_or_other_column(wrong):
    class Chat:
        def complete(self, messages):
            return json.dumps({'status': 'answered', 'claims': [claim(wrong, 'LogReg | 90% | 80%')]})
    result = GroundedQA(Chat(), atomic_single=True).generate('LogReg的Recall是多少？', table_evidence())
    assert result['status'] != 'answered'
    assert not result.get('claims')
    assert any(w['code'] == 'TABLE_TASK_VALUE_MISSING' for w in result['warnings'])


def test_mapped_nonnumeric_cell_requires_the_value_and_keeps_exact_quote():
    evidence = [{'chunk_id': 't', 'content_type': 'table', 'text': 'Device | Status\nDevice A | PAID'}]
    class Chat:
        def complete(self, messages):
            return json.dumps({'status': 'answered', 'claims': [claim('Device A Status PAID', 'Device A | PAID')]})
    result = GroundedQA(Chat(), atomic_single=True).generate('Device A的Status？', evidence)
    assert result['status'] == 'answered'
    assert result['claims'][0]['supporting_quotes'] == ['Device A | PAID']


def test_multilevel_header_view_and_wrong_column_rejected():
    evidence=table_evidence()
    h=evidence[0]
    h['text']='Object | Metrics | Metrics\nObject | Accuracy | Recall\nModel C | 91% | 87%'
    h['metadata']['table_data']['header_rows']=2
    h['metadata']['table_headers']=['Object','Metrics > Accuracy','Metrics > Recall']
    h['table_headers']=h['metadata']['table_headers']
    h['table_data']=h['metadata']['table_data']
    view=table_view(h,'Model C Accuracy')
    assert view['rows'][0]['cells']['Metrics > Accuracy']=='91%'
    assert validate_claim(claim('Model C Accuracy 91%','Model C | 91% | 87%'),[h])
    with pytest.raises(ValueError,match='TABLE_RELATION'):
        validate_claim(claim('Model C Accuracy 87%','Model C | 91% | 87%'),[h])


def test_decimal_arithmetic_is_separate_from_source_quotes_and_units():
    evidence=table_evidence()
    result=calculate('比较LogReg与SVM的Accuracy差值',evidence)
    assert result['result']=='5' and result['unit']=='percentage_points'
    assert [o['value'] for o in result['operands']]==['90','95']
    assert all(o['quote'] in evidence[o['citation_id']-1]['text'] for o in result['operands'])
    class Chat:
        def complete(self,messages):return json.dumps({'status':'insufficient_evidence','claims':[]})
    answer=GroundedQA(Chat(),allow_calculation=True).generate('比较LogReg与SVM的Accuracy差值',evidence)
    assert answer['calculations'][0]['result']=='5'
    assert '计算结果（非原文摘录）' in answer['answer'] and answer['status']=='partial'
    assert answer['calculation_citations'] and answer['fact_verified'] is False


def test_conflicting_source_values_do_not_get_silently_picked_for_arithmetic():
    evidence=table_evidence()
    other=copy.deepcopy(evidence[0])
    other['chunk_id']='other'
    for c in other['metadata']['semantic_cells']:
        if c['row']=='SVM' and c['column']=='Accuracy':
            c['value']='94%'
    other['semantic_cells']=other['metadata']['semantic_cells']
    assert calculate('比较LogReg与SVM的Accuracy差值',evidence+[other]) is None


def test_unsupported_absence_is_rejected_and_f1_is_not_a_numeric_operand():
    evidence=[{'chunk_id':'c','text':'After the comparison table.'}]
    with pytest.raises(ValueError,match='UNSUPPORTED_ABSENCE'):
        validate_claim(claim('The comparison information is not provided.','After the comparison table.'),evidence)
    assert _numbers('F1 score 0.85')=={'0.85'}
    assert _numbers('Accuracy 95%')=={'95%'}
    assert _numbers('Accuracy 95.00%')=={'95%'}


def test_one_shared_repair_retains_valid_partial_when_repair_fails():
    class Chat:
        calls=0
        def complete(self,messages):
            self.calls+=1
            if self.calls==3:
                raise OSError('repair failed')
            return json.dumps({'status':'partial' if self.calls==1 else 'answered','claims':[claim('借阅期限30天。','借阅期限30天。')]})
    chat=Chat()
    result=GroundedQA(chat,max_calls=4).generate('借阅期限？开放时间？',[{'chunk_id':'c','text':'借阅期限30天。'}])
    assert chat.calls==3 and result['generation_repair_count']==1
    assert result['status']=='partial' and '30天' in result['answer'] and result['citations']


def test_missing_explicit_year_does_not_borrow_other_year_score():
    class Chat:
        def complete(self,messages):raise AssertionError('no model call for absent explicit year')
    result=GroundedQA(Chat(),atomic_single=True).generate('2029年SVM的Accuracy是多少？',table_evidence())
    assert result['status']=='insufficient_evidence' and result['model_calls']==0


def test_english_units_keep_dimensions_and_allow_only_literal_equivalence():
    evidence=[{'chunk_id':'c','text':'Warranty period is 24 months.'}]
    assert validate_claim(claim('Warranty period is 24 month.','Warranty period is 24 months.'),evidence)
    with pytest.raises(ValueError,match='NUMERIC_UNIT'):
        validate_claim(claim('Warranty period is 24 days.','Warranty period is 24 months.'),evidence)
def test_quote_grammar_retains_cross_language_fact_despite_heading_overlap():
    from rag_multidocs.qa.grounding import quote_choices
    evidence=[{'text':'设备使用说明 Equipment Notice\nWarranty period: 36 months\nFee: 0 CNY','content_type':'text'}]
    choices=quote_choices(evidence,'设备保修期是多少个月？')
    assert 'Warranty period: 36 months' in choices
    assert all(choice in evidence[0]['text'] for choice in choices)
    assert len(choices)<=64

