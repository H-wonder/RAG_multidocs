import copy
import json

import pytest

from rag_multidocs.qa.grounding import GroundedQA, validate_claim, quote_choices
from rag_multidocs.qa.llm import OpenAICompatibleChat
from rag_multidocs.qa.table_evidence import table_view


@pytest.mark.parametrize("text", [
    "指标 | 模型甲 | 模型乙\n漏判数 | 17 | 4\n误报数 | 2 | 8",
    "| 指标 | 模型甲 | 模型乙 |\n| :--- | ---: | --- |\n| 漏判数 | 17 | 4 |\n| 误报数 | 2 | 8 |",
    "指标 | 模型甲 | 模型乙\r\n漏判数 | 17 | 4\r\n误报数 | 2 | 8",
])
def test_flat_table_semantics_preserve_exact_source_spans(text):
    hit = {"chunk_id": "table", "text": text, "bbox": "[1,2,3,4]", "page": 8}
    original = copy.deepcopy(hit)
    view = table_view(hit, "漏判数是多少？")
    assert view["status"] == "mapped_flat_table"
    row = view["rows"][0]
    assert row["cells"] == {"模型甲": "17", "模型乙": "4"}
    assert row["statements"] == ["模型甲的漏判数：17", "模型乙的漏判数：4"]
    for item in [view["header"], *view["rows"]]:
        start, end = item["raw_span"]
        assert text[start:end] == item.get("text", item.get("supporting_quote"))
    assert hit == original


@pytest.mark.parametrize("text", [
    "指标 | 模型 | 模型\n漏判 | 1 | 2",
    "指标 | 模型甲 | 模型乙\n漏判 | 1",
    "指标 | 模型甲 | 模型乙\n漏判 | 1 | 2\n漏判 | 3 | 4",
    "指标 | 模型甲 | 模型乙\n漏判 | 1 |",
])
def test_ambiguous_tables_are_not_inferred(text):
    assert table_view({"chunk_id": "t", "text": text}) == {"status": "ambiguous", "rows": []}


def test_numeric_format_is_reported_never_repaired():
    text = "指标 | 甲 | 乙\n准确率 | 96 80% . | 93 00% ."
    row = table_view({"chunk_id": "t", "text": text})["rows"][0]
    assert row["cells"]["甲"] == "96 80% ."
    assert row["warnings"] == ["UNRESOLVED_NUMERIC_FORMAT"]
    assert "96.80" not in json.dumps(row)


def test_semantic_table_view_is_not_an_l2_quote():
    text = "指标 | 甲 | 乙\n漏判数 | 17 | 4"
    hit = {"chunk_id": "t", "text": text, "content_type": "table"}
    c = {"text": "甲的漏判数17。", "citation_ids": [1], "supporting_quotes": ["甲的漏判数：17"]}
    with pytest.raises(ValueError, match="L2_FALSE_QUOTE"):
        validate_claim(c, [hit])
    c["supporting_quotes"] = ["漏判数 | 17 | 4"]
    assert validate_claim(c, [hit])


def test_distinct_column_claims_share_original_row_not_invented_cells():
    text = "指标 | 实验组 | 对照组\n耗时 | 12ms | 36ms"
    e = [{"chunk_id": "t", "text": text, "content_type": "table"}]
    for label, value in [("实验组", "12ms"), ("对照组", "36ms")]:
        assert validate_claim({"text": f"{label}耗时{value}", "citation_ids": [1], "supporting_quotes": ["耗时 | 12ms | 36ms"]}, e)
    with pytest.raises(ValueError, match="L3_TABLE_RELATION"):
        validate_claim({"text": "实验组耗时36ms", "citation_ids": [1], "supporting_quotes": ["耗时 | 12ms | 36ms"]}, e)


class TwoQuestionChat:
    def __init__(self, second="answered"):
        self.calls = []
        self.second = second

    def complete(self, messages):
        question = json.loads(messages[1]["content"])["question"]
        self.calls.append(question)
        index = len(self.calls)
        if index == 2 and self.second == "error":
            raise OSError("offline")
        status = self.second if index == 2 else "answered"
        claims = [] if status == "insufficient_evidence" else [{
            "text": "门诊周二开放。" if index == 1 else "退款须凭原始收据。",
            "citation_ids": [index],
            "supporting_quotes": ["门诊周二开放。" if index == 1 else "退款须凭原始收据。"],
        }]
        return json.dumps({"status": status, "claims": claims})


def multi_evidence():
    return [{"chunk_id": "a", "source": "schedule.txt", "text": "门诊周二开放。"},
            {"chunk_id": "b", "source": "refund.md", "text": "退款须凭原始收据。"}]


def test_explicit_generation_tasks_each_answered_without_recall():
    chat = TwoQuestionChat()
    original = multi_evidence()
    result = GroundedQA(chat).generate("门诊何时开放？退款需要什么？", original, missing=["词面假缺口"])
    assert chat.calls == ["门诊何时开放？", "退款需要什么？"]
    assert result["status"] == "answered" and result["model_calls"] == 2
    assert len(result["citations"]) == 2 and "原始收据" in result["answer"]
    assert result["unanswered_subquestions"] == [] and result["fact_verified"] is False
    assert all(a["messages"] and a["raw_response"] for a in result["attempts"])
    assert len(original) == 2


@pytest.mark.parametrize("second", ["partial", "insufficient_evidence", "error"])
def test_missing_second_answer_stays_partial_and_preserves_first(second):
    chat = TwoQuestionChat(second)
    result = GroundedQA(chat).generate("门诊何时开放？退款需要什么？", multi_evidence())
    assert result["status"] == "partial"
    assert result["unanswered_subquestions"] == ["退款需要什么"]
    assert "门诊周二" in result["answer"] and result["citations"]
    assert result["model_calls"] == 2


def test_three_tasks_respect_total_call_limit():
    chat = TwoQuestionChat()
    result = GroundedQA(chat).generate("开放时间？退款要求？预约方式？", multi_evidence())
    assert len(chat.calls) == result["model_calls"] == 2
    assert result["status"] == "partial"
    assert result["subquestion_results"][2]["status"] == "not_attempted"


def test_single_question_does_not_trigger_segmentation_and_empty_skips_model():
    chat = TwoQuestionChat()
    result = GroundedQA(chat).generate("门诊何时开放？", multi_evidence())
    assert result["status"] == "answered" and len(chat.calls) == 1
    result = GroundedQA(chat).generate("开放时间？退款要求？", [])
    assert result["status"] == "insufficient_evidence" and result["model_calls"] == 0


def test_explicit_table_columns_get_distinct_generation_tasks():
    data = [{"chunk_id": "t", "source": "new-table.pdf", "content_type": "table",
             "text": "指标 | 模型甲 | 模型乙\n漏判数 | 17 | 4"}]
    class Chat:
        calls = []

        def complete(self, messages):
            q = json.loads(messages[1]["content"])["question"]
            self.calls.append(q)
            value = "模型甲漏判数17" if "仅回答“模型甲”" in q else "模型乙漏判数4"
            return json.dumps({"status": "answered", "claims": [{"text": value, "citation_ids": [1], "supporting_quotes": ["漏判数 | 17 | 4"]}]})
    chat = Chat()
    result = GroundedQA(chat).generate("模型甲和模型乙的漏判数分别多少？", data)
    assert result["status"] == "answered" and len(result["claims"]) == 2
    assert "模型甲漏判数17" in result["answer"] and "模型乙漏判数4" in result["answer"]
    assert len(chat.calls) == 2 and len(result["citations"]) == 1
    assert all(q == "漏判数 | 17 | 4" for q in result["citations"][0]["excerpts"])


def test_exact_quote_choices_do_not_repair_punctuation_or_cell_text():
    data = [{"chunk_id": "a", "text": "门诊周二开放。退款须凭收据。"},
            {"chunk_id": "t", "content_type": "table", "text": "指标 | 甲\n率 | 96 80% ."}]
    choices = quote_choices(data)
    assert "门诊周二开放。" in choices and "门诊周二开放." not in choices
    assert "率 | 96 80% ." in choices and all("96.80" not in q for q in choices)
    assert all(any(q in h["text"] for h in data) for q in choices)
    # Enum membership alone cannot transfer a quote to a different citation.
    with pytest.raises(ValueError, match="L2_FALSE_QUOTE"):
        validate_claim({"text": "退款须凭收据。", "citation_ids": [2], "supporting_quotes": ["退款须凭收据。"]}, data)


def test_subquestion_quote_candidates_are_only_a_bounded_relevance_aid():
    evidence = [{"chunk_id": "s", "text": "图书馆开放时间。\n自助还书机24小时可用。\n闭馆公告另行发布。"}]
    assert quote_choices(evidence, "自助还书机可用时间是多少") == ["自助还书机24小时可用。"]
    # No lexical match does not manufacture supporting evidence or assert coverage.
    assert quote_choices(evidence, "unmatched") == quote_choices(evidence)
    assert len(quote_choices([{"chunk_id": "long", "text": "\n".join(f"原文{i}。" for i in range(100))}, *evidence])) <= 64


def test_explicit_english_coordinated_fields_preserve_both_tasks():
    evidence = [{"chunk_id": "record", "source": "project.txt", "text": "Owner: Alice\nDeadline: Friday"}]
    class Chat:
        calls = []

        def complete(self, messages):
            q = json.loads(messages[1]["content"])["question"]
            self.calls.append(q)
            text = "Owner: Alice" if "owner" in q else "Deadline: Friday"
            return json.dumps({"status": "answered", "claims": [{"text": text, "citation_ids": [1], "supporting_quotes": [text]}]})
    chat = Chat()
    result = GroundedQA(chat).generate("What are the owner and deadline in the project record?", evidence)
    assert chat.calls == ["What is the owner in the project record？", "What is deadline in the project record？"]
    assert result["status"] == "answered" and result["model_calls"] == 2
    assert "Alice" in result["answer"] and "Friday" in result["answer"]
    assert quote_choices(evidence, "What is deadline in the project record?") == ["Deadline: Friday"]


def test_opted_in_schema_quote_enum_is_per_call_and_preserves_shared_config(monkeypatch):
    schema = {"type": "json_schema", "json_schema": {"schema": {"properties": {"claims": {"items": {"properties": {"supporting_quotes": {"items": {"type": "string"}}}}}}}}}
    original = copy.deepcopy(schema)
    chat = OpenAICompatibleChat("http://local", "test", response_format=schema)
    sent = []
    def post(messages, response_format_override=None):
        sent.append(response_format_override)
        return "ok"
    monkeypatch.setattr(chat, "_post", post)
    assert chat.complete_grounded([], ["原文甲。"] ) == "ok"
    assert chat.complete_grounded([], ["原文乙。"] ) == "ok"
    def items(value):
        return value["json_schema"]["schema"]["properties"]["claims"]["items"]["properties"]["supporting_quotes"]["items"]
    assert items(sent[0])["enum"] == ["原文甲。"]
    assert items(sent[1])["enum"] == ["原文乙。"]
    assert schema == original and chat.response_format == original
    chat.response_format = None
    assert chat.complete_grounded([], ["原文丙。"] ) == "ok" and sent[-1] is None
