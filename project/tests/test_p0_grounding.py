import json

import pytest

from rag_multidocs.qa.grounding import GroundedQA, validate_claim
from test_a_plus_grounding import claim, evidence


def test_fee_waiver_counterexample_rejected_without_claiming_general_entailment():
    e = [{"chunk_id": "fee", "source": "rules", "text": "逾期费用每天每本0.2元。"}]
    with pytest.raises(ValueError, match="L3_FEE_WAIVER_CONTRADICTION"):
        validate_claim(claim("图书馆免除所有逾期费用。", e[0]["text"]), e)
    # Limited rules do not establish general semantics; this must never be exposed as fact_verified.
    assert validate_claim(claim("图书馆支持航天研究。", e[0]["text"]), e)


def test_model_partial_not_promoted_and_repair_failure_keeps_valid_part():
    class Chat:
        def __init__(self, repair=False):
            self.calls, self.repair = 0, repair

        def complete(self, messages):
            self.calls += 1
            if self.calls == 2:
                raise OSError("repair offline")
            claims = [claim("借阅期限30天。", "借阅期限30天。")]
            if self.repair:
                claims.append(claim("100元。", "借阅期限30天。"))
            return json.dumps(
                {"status": "answered" if self.repair else "partial", "claims": claims}
            )

    direct = GroundedQA(Chat()).generate("借阅期限", evidence())
    assert direct["status"] == "partial" and direct["fact_verified"] is False
    c = Chat(repair=True)
    result = GroundedQA(c).generate("借阅期限", evidence())
    assert c.calls == 2 and result["status"] == "partial"
    assert result["citations"] and "30天" in result["answer"] and "100元" not in result["answer"]
    assert any(w["code"] == "LLM_REPAIR_ERROR" for w in result["warnings"])


def test_full_prompt_and_repair_history_are_budgeted():
    class Chat:
        calls = 0
        max_tokens = 100

        def complete(self, messages):
            self.calls += 1
            return "invalid " * 1000

    c = Chat()
    qa = GroundedQA(c, context_tokens=1000, output_tokens=100)
    result = qa.generate("q", evidence())
    assert c.calls <= 1
    assert result["model_calls"] == c.calls
    assert result["status"] in {"generation_failed", "insufficient_evidence"}
    c = Chat()
    result = GroundedQA(c, context_tokens=300, output_tokens=100).generate(
        "长问题" * 100, evidence()
    )
    assert c.calls == 0 and result["model_calls"] == 0


def test_outer_json_fence_is_only_transport_normalization():
    from rag_multidocs.qa.grounding import parse_response

    raw = json.dumps({"status": "partial", "claims": []})
    assert parse_response("```json\n" + raw + "\n```")["status"] == "partial"
    with pytest.raises(ValueError):
        parse_response("Here is my answer: " + raw)
    with pytest.raises(ValueError):
        parse_response("```json\n{broken}\n```")


def test_table_column_relation_and_quote_continuity():
    text = "指标 | 原始逻辑回归 | 类别加权逻辑回归\n少数类漏判数 | 15 | 3"
    e = [{"chunk_id": "t", "source": "table.pdf", "text": text, "content_type": "table"}]
    assert validate_claim(
        claim("原始逻辑回归少数类漏判数15，类别加权逻辑回归少数类漏判数3。", text), e
    )
    with pytest.raises(ValueError, match="TABLE_RELATION"):
        validate_claim(
            claim("原始逻辑回归少数类漏判数3，类别加权逻辑回归少数类漏判数15。", text), e
        )


def test_prompt_trimming_returns_only_effective_generation_evidence():
    class Chat:
        max_tokens = 100
        seen = None

        def complete(self, messages):
            self.seen = json.loads(messages[1]["content"])["evidence"]
            return json.dumps({"status": "insufficient_evidence", "claims": []})

    c = Chat()
    qa = GroundedQA(c, context_tokens=1500, output_tokens=100)
    data = [
        {"chunk_id": "short", "text": "short fact"},
        {"chunk_id": "huge", "text": "long " * 2000},
    ]
    result = qa.generate("q", data)
    assert [h["chunk_id"] for h in result["generation_evidence"]] == ["short"]
    assert [h["chunk_id"] for h in c.seen] == ["short"]
    assert result["generation_context_tokens_estimate"] < 1500
    assert all(a["prompt_tokens_estimate"] + 100 <= 1500 for a in result["attempts"])
    assert len(data) == 2  # caller-owned evidence remains intact
