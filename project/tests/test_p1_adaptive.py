from dataclasses import replace

from rag_multidocs.retrieval.needs import build_needs, support
from rag_multidocs.retrieval.controller import RetrievalController, RetrievalPolicy, select_context
from test_a_plus_retrieval import h, NoRerank


def test_cartesian_object_metric_requirements_and_no_cooccurrence_proof():
    needs=build_needs('比较逻辑回归与SVM的准确率和召回率')
    assert [(n.object,n.metric) for n in needs]==[('逻辑回归','accuracy'),('逻辑回归','recall'),('SVM','accuracy'),('SVM','recall')]
    hit=h('a','逻辑回归准确率80%。SVM召回率90%。')
    assert [support(hit,n)['supported'] for n in needs]==[True,False,False,True]
    assert not support(h('b','逻辑回归和SVM准确率90%。'),needs[0])['supported']
    assert not support(h('b','SVM2准确率90%。'),needs[2])['supported']
    assert not support(h('b','SVM准确率90元。'),needs[2])['supported']


def test_mapped_table_relations_do_not_mix_rows_columns_or_damaged_values():
    need=build_needs('比较逻辑回归与SVM的准确率和召回率')[2]
    hit=h('table','Model | Accuracy | Recall\nSVM | 95% | 85%')
    hit['semantic_cells']=[{'row':'SVM','column':'Recall','value':'85%','quote':'SVM | 95% | 85%'}]
    assert not support(hit,need)['supported']
    hit['semantic_cells'].append({'row':'SVM','column':'Accuracy','value':'95%','quote':'SVM | 95% | 85%'})
    assert support(hit,need)['value']=='95%'
    hit['semantic_cells'][-1]['value']='95 00% .'
    assert not support(hit,need)['supported']


def test_coverage_utility_uses_new_relations_not_source_count():
    needs=build_needs('比较逻辑回归与SVM的准确率')
    candidates=[h('lr','逻辑回归准确率80%。',.9),h('duplicate','逻辑回归准确率80%。',.8,source='different'),h('svm','SVM准确率90%。',.75)]
    selected,missing,cost=select_context(candidates,needs,{'context_k':2,'context_tokens':300},RetrievalPolicy())
    assert [x['chunk_id'] for x in selected]==['lr','svm']
    assert not missing and cost<=300


def test_hand_computable_coverage_gain_changes_second_choice():
    needs=build_needs('比较逻辑回归与SVM的准确率')
    candidates=[h('first','逻辑回归准确率80%。',.9),h('duplicate','逻辑回归准确率80%。',.55),h('new','SVM准确率90%。',.45)]
    policy=RetrievalPolicy(use_mmr=False)
    selected,missing,_=select_context(candidates,needs,{'context_k':2,'context_tokens':500},policy)
    # After first: duplicate=.65*.55=.3575; new=.65*.45+.15/2=.3675.
    assert [h['chunk_id'] for h in selected]==['first','new'] and not missing


def test_directed_queries_rescue_only_missing_object_metrics_and_preserve_year():
    class Index:
        def __init__(self):self.calls=[]
        def search(self,q,**kwargs):
            self.calls.append((q,kwargs))
            if q.startswith('SVM 结果'):
                return [h('svm','SVM准确率90%。SVM召回率85%。',source='svm.txt')]
            return [h('lr','逻辑回归准确率80%。逻辑回归召回率70%。',source='lr.txt')]
    index=Index()
    policy=RetrievalPolicy(reranker='none',use_rewrite=False)
    result=RetrievalController(index,NoRerank(),policy).retrieve('比较逻辑回归与SVM的准确率和召回率，限定2026年')
    assert len(result['retrieval_trace'])==2
    assert result['retrieval_trace'][0]['finite_relation_coverage']['context']==[0,1]
    assert result['retrieval_trace'][1]['finite_relation_coverage']['context']==[0,1,2,3]
    assert index.calls[1][0].startswith('SVM 结果') and '准确率' in index.calls[1][0] and '召回率' in index.calls[1][0] and '2026年' in index.calls[1][0]
    assert len(result['auxiliary_queries'])<=3 and result['index_calls']<=4


def test_simple_hard_query_escalates_and_explicit_override_stays_fixed():
    class Index:
        def __init__(self):self.limits=[]
        def search(self,q,**kwargs):
            self.limits.append(kwargs['top_k'])
            return []
    index=Index()
    controller=RetrievalController(index,NoRerank(),RetrievalPolicy(reranker='none',use_rewrite=False))
    result=controller.retrieve('稀有术语查询')
    assert index.limits==[12,24]
    assert result['retrieval_trace'][1]['resources']['escalated']
    index=Index()
    RetrievalController(index,NoRerank(),replace(controller.policy,candidate_k=10,context_tokens=900)).retrieve('稀有术语查询')
    assert index.limits==[10,10]


def test_existing_candidate_relation_gap_is_not_recalled_as_if_missing():
    class Index:
        calls=0
        def search(self,q,**kwargs):
            self.calls+=1
            return [h('lr','逻辑回归准确率80%。'),h('svm','SVM准确率90%。')]
    index=Index()
    policy=RetrievalPolicy(reranker='none',use_rewrite=False,context_k=1)
    result=RetrievalController(index,NoRerank(),policy).retrieve('比较逻辑回归与SVM的准确率')
    assert index.calls==1 and any(w['code']=='CONTEXT_SELECTION_GAP' for w in result['warnings'])
    assert result['missing_evidence']
