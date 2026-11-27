
import pytest

from rag_multidocs.retrieval.controller import RetrievalController, RetrievalPolicy, select_context


def h(key, text, rel=0.8, source="a", page=1):
    return {
        "chunk_id": key,
        "doc_id": source,
        "source": source,
        "page": page,
        "text": text,
        "score": rel,
        "normalized_relevance": rel,
        "rerank_score": rel,
    }


class NoRerank:
    def rerank(self, name, query, hits):
        return hits, []


def test_adaptive_resources_and_fixed_override():
    p = RetrievalPolicy()
    assert p.resources("开放时间是多少？")["candidate_k"] == 12
    assert p.resources("为什么需要重排？")["context_k"] == 4
    assert p.resources("比较逻辑回归与SVM")["context_k"] == 6
    assert (
        RetrievalPolicy(adaptive=False, context_tokens=900).resources("比较A和B")["context_tokens"]
        == 900
    )
    with pytest.raises(ValueError):
        RetrievalPolicy(candidate_k=49).resources("x")


def test_directed_recall_targets_missing_object_merges_and_limits():
    class Index:
        def __init__(self):
            self.calls = []

        def search(self, q, **kw):
            self.calls.append((q, kw))
            if q.startswith("SVM 结果"):
                return [h("svm", "SVM 的实验结果：准确率90%。", source="b")]
            return [h("lr", "逻辑回归的实验结果：准确率80%。")]

    index = Index()
    p = RetrievalPolicy(reranker="none", use_rewrite=False)
    result = RetrievalController(index, NoRerank(), p).retrieve("比较逻辑回归与SVM的实验结果")
    assert len(result["retrieval_trace"]) == 2
    assert any(q.startswith("SVM 结果") for q, kw in index.calls)
    assert len(result["auxiliary_queries"]) <= 3
    assert len({x["doc_id"] for x in result["hits"]}) == 2
    assert not result["missing_evidence"]
    assert all(kw["top_k"] <= 48 for q, kw in index.calls)


def test_query_rewrite_failure_keeps_original_and_no_character_spam():
    class Index:
        def __init__(self):
            self.calls = []

        def search(self, q, **kw):
            self.calls.append(q)
            return []

    index = Index()

    def bad(q):
        raise RuntimeError()

    r = RetrievalController(index, NoRerank(), rewriter=bad).retrieve("比较图书馆开放时间和借阅期限")
    assert index.calls[0] == "比较图书馆开放时间和借阅期限"
    assert any(w["code"] == "QUERY_REWRITE_FAILED" for w in r["warnings"])
    assert len(r["auxiliary_queries"]) <= 3 and len(r["retrieval_trace"]) <= 2
    assert all("图 书 馆" not in q for q in index.calls)


def test_coverage_gain_not_document_count_and_all_neighbors_budgeted():
    p = RetrievalPolicy()
    resources = {"context_k": 2, "context_tokens": 200}
    candidates = [
        h("a", "逻辑回归 的结果", 0.9),
        h("b", "逻辑回归 的结果重复", 0.8, source="b"),
        h("s", "SVM 结果", 0.75, source="a"),
    ]
    hits, missing, used = select_context(candidates, ["逻辑回归", "SVM"], resources, p)
    assert {x["chunk_id"] for x in hits} == {"a", "s"} and not missing and used <= 200
    huge = h("huge", "超长" * 4000, 1)
    assert not select_context([huge], ["超长"], resources, p)[0]


def test_low_relevance_candidates_not_context():
    hits, missing, tokens = select_context(
        [h("a", "低相关", 0.01)],
        ["目标"],
        {"context_k": 2, "context_tokens": 1000},
        RetrievalPolicy(),
    )
    assert hits == [] and tokens == 0
