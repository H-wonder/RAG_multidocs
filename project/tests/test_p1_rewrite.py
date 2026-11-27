import pytest

from rag_multidocs.retrieval.rewrite import level1_queries, validate_rewrites, query_keywords
from rag_multidocs.retrieval.controller import RetrievalController, RetrievalPolicy
from test_a_plus_retrieval import NoRerank


def test_level1_preserves_numbers_aliases_and_no_single_character_spam():
    q='比较SVM在2026年和逻辑回归的准确率，限制24小时'
    queries=level1_queries(q)
    assert any('支持向量机' in s and '2026年' in s and '24小时' in s for s in queries)
    assert all('支 持 向 量' not in s for s in queries)
    assert 'SVM' in query_keywords(q)


@pytest.mark.parametrize('value',[['SVM准确率99%'],['SVM在2025年准确率'],['XGBoost在2026年准确率']])
def test_semantic_rewrite_rejects_invented_numbers_and_identifiers(value):
    with pytest.raises(ValueError):
        validate_rewrites('比较SVM在2026年准确率',value)


def test_original_query_retained_and_invalid_rewriter_falls_back():
    class Index:
        def __init__(self): self.calls=[]
        def search(self,q,**kwargs):
            self.calls.append(q)
            return []
    index=Index()
    policy=RetrievalPolicy(reranker='none',use_recall=False)
    result=RetrievalController(index,NoRerank(),policy,lambda q:['SVM准确率99%']).retrieve('比较SVM在2026年准确率')
    assert index.calls[0]=='比较SVM在2026年准确率'
    assert all('99%' not in q for q in index.calls)
    assert any(w['code']=='QUERY_REWRITE_FAILED' for w in result['warnings'])
    assert len(index.calls)<=3
