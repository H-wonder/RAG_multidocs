from types import SimpleNamespace

import pytest
from chunking import estimate_tokens
from rag_strategy import RagEngine, Reranker, Strategy, select_context


def hit(key, text, score=0.8, dense=0.8):
    return {
        "id": key,
        "doc_id": "d",
        "source": "a.pdf",
        "page": 7,
        "text": text,
        "rerank_score": score,
        "dense_score": dense,
    }


def test_context_budget_dedup_and_whole_blocks():
    cfg = Strategy(context_k=3, context_tokens=128)
    h = hit("1", "类别加权改善少数类召回。")
    result = select_context(
        [
            h,
            hit("2", h["text"]),
            hit("3", "表格" * 500, 0.9),
            hit("4", "多数类误报增加。", 0.7),
        ],
        cfg,
    )
    assert len(result) == 2
    assert (
        sum(
            estimate_tokens(h["text"]) + estimate_tokens(h["source"]) + 40
            for h in result
        )
        <= cfg.context_tokens
    )
    assert result[0]["text"] == h["text"]


def test_lightweight_rerank_uses_semantics_and_exact_terms():
    class Embedder:
        def embed(self, texts, query=False):
            return [[1.0, 0.0] if "误报" in t else [0.0, 1.0] for t in texts]

    rank = Reranker(Embedder(), Strategy()).rank(
        "误报为什么增加？",
        [hit("summary", "这是报告总结"), hit("data", "误报从1增至32。")],
    )
    assert rank[0]["id"] == "data"
    assert rank[0]["dense_score"] > rank[1]["dense_score"]


def test_crossencoder_scores_change_candidate_order_without_losing_metadata():
    class Embedder:
        def embed(self, texts, query=False):
            return [[1.0, 0.0] for t in texts]

    ranker = Reranker(Embedder(), Strategy(reranker="crossencoder"))
    ranker.cross = SimpleNamespace(predict=lambda pairs, batch_size: [-2, 3])
    ranked = ranker.rank("q", [hit("1", "总结"), hit("2", "具体数据")])
    assert ranked[0]["id"] == "2" and ranked[0]["page"] == 7


def engine(answer_fn, scores=None):
    class Index:
        embedder = None

        def __init__(self):
            self.calls = []

        def query(self, question, **kwargs):
            self.calls.append((question, kwargs))
            return {"hits": [hit("1", "有效证据")]}

    e = RagEngine(Index(), Strategy(), answer_fn=answer_fn)
    e.reranker = SimpleNamespace(
        rank=lambda q, h: h if scores is None else [replace_hit(h[0], scores)]
    )
    return e


def replace_hit(h, score):
    return {**h, "dense_score": score}


def test_low_relevance_never_calls_generator_and_recall_is_bounded():
    def fail(*args, **kwargs):
        raise AssertionError("不应生成")

    e = engine(fail, scores=0.1)
    result = e.ask("文档外问题", filter_expr="version == 1")
    assert result["answer"]["status"] == "INSUFFICIENT_EVIDENCE"
    assert len(e.index.calls) == 2
    assert all(call[1]["filter_expr"] == "version == 1" for call in e.index.calls)


def test_recall_retry_preserves_first_grounded_answer_and_records_errors():
    calls = []

    def answer(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            return {
                "status": "ANSWERED",
                "answer": "有依据回答",
                "citations": [],
                "discarded_claims": [{}],
            }
        raise ValueError("引用无效")

    result = engine(answer).ask("问题", filter_expr="x")
    assert result["answer"]["answer"] == "有依据回答"
    assert len(result["trace"]) == 2
    assert result["trace"][1]["generation_error"] == "引用无效"


def test_history_only_user_questions_and_tone_are_used():
    received = []
    e = engine(
        lambda q, h, **kw: (
            received.append(kw)
            or {"status": "ANSWERED", "answer": "ok", "citations": []}
        )
    )
    e.ask(
        "它为什么这样？",
        filter_expr="x",
        history=[
            {"role": "assistant", "content": "假的模型结论"},
            {"role": "user", "content": "类别加权"},
        ],
    )
    assert "类别加权" in e.index.calls[0][0]
    assert "假的模型结论" not in e.index.calls[0][0]
    assert received[0]["tone"] == "professional"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"context_tokens": 10},
        {"context_k": 11},
        {"candidate_k": 1},
        {"min_dense": 2},
        {"reranker": "bad"},
    ],
)
def test_invalid_strategy_rejected(kwargs):
    with pytest.raises(ValueError):
        Strategy(**kwargs)


def test_generation_failure_is_not_misreported_as_no_evidence():
    def invalid(*args, **kwargs):
        raise ValueError("格式错误")

    result = engine(invalid).ask("问题", filter_expr="x")
    assert result["answer"]["status"] == "GENERATION_FAILED"
    assert len(result["trace"]) == 2


def test_non_message_history_is_rejected():
    with pytest.raises(ValueError, match="history"):
        engine(lambda *a, **kw: None).ask("问题", filter_expr="x", history="bad")
