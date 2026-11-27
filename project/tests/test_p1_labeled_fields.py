import json

from rag_multidocs.qa.grounding import GroundedQA, validate_claim


class FieldChat:
    def __init__(self, responder):
        self.responder = responder

    def complete_fields(self, messages, task_ids, line_ids):
        return json.dumps({"answers": self.responder(json.loads(messages[1]["content"]))})


def hit(source, text):
    return {"source": source, "doc_id": source, "chunk_id": source, "text": text}


def use_fields(values):
    def respond(prompt):
        return [{"task_id": t["task_id"], "line_id": t["line_id"], "value": value}
                for t, value in zip(prompt["tasks"], values)]
    return FieldChat(respond)


def test_chinese_coordinated_fields_use_literal_lines_with_units():
    evidence = [
        hit("lab.pdf", "1 借阅须知\n3 服务点名称：东区设备站\n4 开放时间：07:15 - 19:45\n5 设备站容量：63人\n6 状态：例行维护"),
        hit("other.png", "1 另一站点\n2 服务点名称：西区设备站\n3 开放时间：07:15 - 19:45\n4 设备站容量：63人"),
    ]
    qa = GroundedQA(use_fields(["07:15 - 19:45", "63人", "例行维护"]))
    result = qa.generate("东区设备站的开放时间、容量及状态分别是什么？", evidence)
    assert result["status"] == "answered" and result["model_calls"] == 1
    assert len(result["task_requirements"]) == 3
    assert [c["source"] for c in result["citations"]] == ["lab.pdf"]
    assert result["citations"][0]["excerpts"] == evidence[0]["text"].splitlines()[2:]
    assert all(validate_claim(c, result["answer_evidence"]) for c in result["claims"])


def test_wrong_line_with_same_value_and_missing_unit_keep_partial():
    evidence = [hit("orders.txt", "设备名称：水压计\n采购金额：85.50元\n维护费用：85.50元")]
    def reply(prompt):
        if len(prompt["tasks"]) == 1:
            return []
        first, second = prompt["tasks"]
        return [{"task_id": first["task_id"], "line_id": first["line_id"], "value": "85.50元"},
                {"task_id": second["task_id"], "line_id": first["line_id"], "value": "85.50"},
                ]
    result = GroundedQA(FieldChat(reply)).generate('"orders.txt"中的采购金额和维护费用分别是多少？', evidence)
    assert result["status"] == "partial" and len(result["claims"]) == 1
    assert result["unanswered_subquestions"] == ["orders.txt 维护费用"]
    assert "L1_TASK_EVIDENCE_MISMATCH" in str(result["attempts"])
    def wrong_unit(prompt):
        return [{"task_id": t["task_id"], "line_id": t["line_id"],
                 "value": "85.50" if t["field"] == "采购金额" else "85.50元"} for t in prompt["tasks"]]
    result = GroundedQA(FieldChat(wrong_unit)).generate(
        '"orders.txt"中的采购金额和维护费用分别是多少？', evidence)
    assert result["status"] == "partial"
    assert "FIELD_VALUE_MISMATCH" in str(result["attempts"])


def test_ambiguous_field_cannot_select_first_record():
    evidence = [hit("log.txt", "设备名称：甲机器\n能耗：120 kWh\n设备名称：乙机器\n能耗：130 kWh")]
    result = GroundedQA(use_fields([])).generate("甲机器的能耗是多少？", evidence)
    assert result["status"] == "insufficient_evidence"
    assert not result["claims"] and not result["model_calls"]


def test_complete_literal_line_is_a_valid_value_wrapper():
    evidence = [hit("log.txt", "设备名称：水泵\n7 保修期限：22个月")]
    result = GroundedQA(use_fields(["7 保修期限：22个月"])).generate('"log.txt"中的保修期限是多少？', evidence)
    assert result["status"] == "answered"
    assert result["answer"] == "保修期限：22个月 [1]"
    assert result["citations"][0]["excerpts"] == ["7 保修期限：22个月"]
    assert json.loads(result["attempts"][0]["raw_response"])["answers"][0]["value"] == "7 保修期限：22个月"


def test_retry_sees_only_unfinished_task_and_its_literal_line():
    evidence = [hit("log.txt", "设备名称：水泵\n保修期：22个月\n维护费：90元")]
    prompts = []

    def read_first(prompt):
        prompts.append(prompt)
        task = prompt["tasks"][0]
        value = next(row["text"].split("：")[1] for row in prompt["evidence_lines"]
                     if row["id"] == task["line_id"])
        return [{"task_id": task["task_id"], "line_id": task["line_id"], "value": value}]

    result = GroundedQA(FieldChat(read_first)).generate('"log.txt"中的保修期和维护费分别是多少？', evidence)
    assert result["status"] == "answered" and result["model_calls"] == 2
    assert [t["field"] for t in prompts[1]["tasks"]] == ["维护费"]
    assert [line["text"] for line in prompts[1]["evidence_lines"]] == ["维护费：90元"]
