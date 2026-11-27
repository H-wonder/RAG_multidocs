from rag_multidocs.evaluation.metrics import run_suite, ablation_policies


def test_scoped_cases_keep_fixed_corpus_and_real_timing_and_null_reviews():
    class Core:
        def list_documents(self):
            return [{'doc_id':'included','source':'a'},{'doc_id':'excluded','source':'b'}]
        def ask(self,q,**kwargs):
            assert kwargs=={'doc_ids':['included']}
            h={'chunk_id':'x','text':'Accuracy 95%','source':'a','page':0}
            return {'status':'answered','answer':'Accuracy 95%','citations':[{}],'generation_evidence':[h],'generation_context_tokens_estimate':100,'retrieval_trace':[{'base_candidates':[h],'reranked_candidates':[h],'candidates':[h],'context_ids':['x'],'rerank_seconds':.2}],'retrieval_seconds':.4,'generation_seconds':.3}
    case={'id':'c','question':'q','answerable':True,'category':'table','source_allowlist':['a'],'expected_evidence':[{'source':'a','contains':'95%'}],'automatic_answer_contains':['95%']}
    report=run_suite(Core(),[case],generate=True)
    summary=report['summary']
    assert summary['hit_at_3']==1 and summary['rerank_seconds_mean']==.2
    assert summary['literal_answer_check_rate']==1
    assert summary['answer_accuracy'] is None and summary['citation_precision'] is None and summary['table_numeric_error_rate'] is None
    assert summary['total_seconds_p95']>=0 and report['by_type']['table']['questions']==1


def test_required_nine_policies_and_module_off_variants_are_distinct():
    policies=ablation_policies()
    required=['baseline_rrf','lightweight','crossencoder','mmr','neighbors','coverage','adaptive','directed_recall','full']
    assert all(k in policies for k in required)
    assert policies['baseline_rrf'].reranker=='none' and not policies['baseline_rrf'].use_coverage
    assert policies['full'].use_recall and policies['full'].use_rewrite and policies['full'].adaptive
    assert not policies['full_no_mmr'].use_mmr and policies['full_no_coverage'].use_coverage is False
    assert policies['full'].resources('简单事实')['candidate_k']==12
    assert policies['full'].resources('比较A与B')['candidate_k']==48


def test_frozen_scope_enumerated_once_for_multiple_requests():
    class Core:
        enumerations=0
        def list_documents(self):
            self.enumerations+=1
            return [{'doc_id':'real','source':'a'}]
        def search(self,q,**kwargs):
            assert kwargs=={'doc_ids':['real']}
            return {'hits':[]}
    core=Core()
    cases=({'id':str(i),'question':'q','answerable':True,'source_allowlist':['a']} for i in range(3))
    assert run_suite(core,cases)['summary']['questions']==3
    assert core.enumerations==1
