import json

import pytest

from rag_multidocs.qa.grounding import GroundedQA, validate_claim
from rag_multidocs.retrieval.controller import RetrievalPolicy, select_context
from rag_multidocs.retrieval.sources import fact_years, resolve_sources


def hit(did, source, text):
    return {"chunk_id": f"c-{did}", "doc_id": did, "source": source, "text": text}


class Chat:
    def __init__(self, response):
        self.response = response
        self.prompts = []

    def complete(self, messages):
        self.prompts.append(messages)
        return json.dumps(self.response, ensure_ascii=False)


def test_dates_in_source_names_do_not_consume_real_time_constraints():
    docs = [{"doc_id": "a", "source": "资料/附件.云溪大学中央仪器开放运行报告（2026年）.pdf"},
            {"doc_id": "b", "source": "校准手册.md"}]
    q = "请结合《附件.云溪大学中央仪器开放运行报告（2026年）》以及校准手册.md，比较2023和2024年的仪器故障率。"
    scope = resolve_sources(q, docs)
    assert scope["doc_ids"] == ["a", "b"] and not scope["blocked"]
    assert fact_years(q, scope) == ["2023", "2024"]
    assert resolve_sources("不存在校准手册.md里的温度", docs)["blocked"]


def test_unquoted_topic_does_not_become_a_source_constraint():
    docs = [{"doc_id": "a", "source": "南区/春季实验设备巡检记录（2024年）.pdf"},
            {"doc_id": "b", "source": "北区/秋季实验设备巡检记录（2024年）.pdf"}]
    result = resolve_sources("根据2024年实验设备巡检记录，哪些设备停机？", docs)
    assert not result["explicit"] and not result["blocked"] and result["doc_ids"] is None


def test_finite_metric_miss_does_not_preempt_a_valid_model_answer():
    rows = [hit("a", "公共测试室.txt", "测试室每周开放六天。")]
    quote = rows[0]["text"]
    chat = Chat({"status": "answered", "claims": [
        {"text": quote, "citation_ids": [1], "supporting_quotes": [quote]},
    ]})
    result = GroundedQA(chat, atomic_single=True).generate("公共测试室.txt的开放时间如何安排？", rows)
    assert result["status"] == "answered" and result["model_calls"] == 1
    assert result["citations"][0]["excerpts"] == [quote]


def test_partial_comparison_cannot_borrow_an_unquoted_year():
    rows = [hit("a", "counts.txt", "2023年更换了7台仪器。"), hit("b", "prior.txt", "更换了8台仪器。")]
    chat = Chat({"status": "answered", "claims": [
        {"text": "更换了8台仪器。", "citation_ids": [2], "supporting_quotes": ["更换了8台仪器。"]},
        {"text": rows[0]["text"], "citation_ids": [1], "supporting_quotes": [rows[0]["text"]]},
    ]})
    result = GroundedQA(chat).generate("2023年和2024年分别更换了几台仪器？", rows)
    assert result["status"] == "partial" and "7台" in result["answer"]
    assert "8台" not in result["answer"]
    assert {w["code"] for w in result["warnings"]} >= {"L3_TEMPORAL_SUPPORT", "TEMPORAL_SCOPE_INCOMPLETE"}


def test_synthesis_uses_all_sources_and_binds_each_reference_to_original():
    rows = [hit("a", "目标说明.txt", "实训重点是接口测试。还需要了解故障恢复。"),
            hit("b", "练习说明.md", "模拟训练要求完成接口验证。")]
    chat = Chat({"status": "answered", "claims": [
        {"text": "目标包括接口测试和故障恢复。", "kind": "fact", "refs": ["E1", "E2"]},
        {"text": "可用模拟训练练习接口测试。", "kind": "analysis", "refs": ["E1", "E3"]},
        {"text": "虚构内容。", "kind": "fact", "refs": ["E42"]},
    ]})
    question = "综合目标说明.txt以及练习说明.md，推荐如何训练？"
    result = GroundedQA(chat, atomic_single=True, max_calls=1).generate(question, rows)
    assert result["status"] == "partial" and result["answer_mode"] == "synthesis"
    assert "分析与建议" in result["answer"] and "虚构" not in result["answer"]
    assert {c["doc_id"] for c in result["citations"]} == {"a", "b"}
    assert len(result["citations"][0]["excerpts"]) >= 2
    assert all(validate_claim(c, result["answer_evidence"]) for c in result["claims"])


def test_duplicate_reference_pair_and_numeric_hallucination_still_fail():
    rows = [hit("a", "targets.txt", "共有4个练习。")]
    claim = {"text": "共有7个练习。", "citation_ids": [1], "supporting_quotes": [rows[0]["text"]]}
    with pytest.raises(ValueError, match="L3_NUMERIC_UNIT_MISMATCH"):
        validate_claim(claim, rows)
    claim["citation_ids"] *= 2
    claim["supporting_quotes"] *= 2
    with pytest.raises(ValueError, match="L1_DUPLICATE_CITATION"):
        validate_claim(claim, rows)


def test_explicit_table_category_limits_recommendations_to_aligned_original_rows():
    eligible = " | 试验平台甲 | 工程小组 | 实验级 | "
    other = " | 云平台乙 | 运行小组 | 生产级 | "
    row = dict(hit("a", "资源清单.txt", "类别组 | 平台名称 | 主管机构 | 认证级别 | 说明\n"
                   + eligible + "\n" + other),
               content_type="table",
               table_headers=["类别组", "平台名称", "主管机构", "认证级别", "说明"])
    class ChoiceChat(Chat):
        def complete(self, messages):
            self.prompts.append(messages)
            records = json.loads(messages[1]["content"])["records"]
            assert set(records) == {"试验平台甲"}
            return json.dumps({"试验平台甲": "可以先在试验环境验证流程。"})
    chat = ChoiceChat(None)
    result = GroundedQA(chat, atomic_single=True).generate("希望选择实验级的平台，请推荐并写出主管机构。", [row])
    assert result["status"] == "answered"
    assert "试验平台甲" in result["answer"] and "工程小组" in result["answer"]
    assert other not in chat.prompts[0][1]["content"]
    assert result["citations"][0]["excerpts"] == [eligible]


def test_extra_synthesis_items_do_not_discard_grounded_items():
    rows = [hit("a", "活动.txt", "参加者应先登记。")]
    chat = Chat({"status": "answered", "claims": [
        {"text": "参加者应先登记。", "kind": "fact", "refs": ["E1"]},
    ] * 9})
    result = GroundedQA(chat, max_calls=1).generate("总结活动安排。", rows)
    assert result["status"] == "partial" and result["citations"]
    assert len(result["claims"]) == 8
    assert any(w["code"] == "L1_CLAIM_LIMIT" for w in result["warnings"])


def test_context_budget_reserves_tokens_for_explicit_sources_before_extra_chunks():
    rows = [dict(hit("a", "notes.txt", "普通说明" * 30), chunk_id=f"a{i}", selection_relevance=1)
            for i in range(3)]
    rows.append(dict(hit("b", "terms.md", "适用条件" * 30), selection_relevance=0.1))
    result, _, tokens = select_context(
        rows, [], {"context_k": 4, "context_tokens": 520}, RetrievalPolicy.baseline(), ["a", "b"],
    )
    assert [row["doc_id"] for row in result[:2]] == ["a", "b"] and tokens <= 520
