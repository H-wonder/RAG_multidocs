from rag_multidocs.qa.grounding import normalize_line_refs, GroundedQA
from rag_multidocs.qa.calculations import calculate
from rag_multidocs.retrieval.sources import (
    labeled_record_source,
    resolve_sources,
    without_source_names,
)
from rag_multidocs.retrieval.rewrite import comparison_parts
from rag_multidocs.connectors.txt import TxtConnector
from rag_multidocs.contracts import ConnectorConfig
import pytest


def test_source_wrappers_are_removed_without_polluting_compared_entities():
    q = "结合《历史.pdf》和《说明.docx》，分析甲厅和乙院在迎宾中的不同作用，以及礼制意义？"
    scope = resolve_sources(
        q, [{"doc_id": "a", "source": "历史.pdf"}, {"doc_id": "b", "source": "说明.docx"}]
    )
    assert comparison_parts(without_source_names(q, scope)) == ("甲厅", "乙院", "迎宾")


def test_factual_source_uses_literal_record_not_a_related_catalog():
    hits = [
        {
            "doc_id": "a",
            "content_type": "table",
            "text": "项目 | 绩效\n甲 | 3",
            "table_headers": ["项目", "绩效"],
        },
        {"doc_id": "b", "text": "平均绩效（API）：4.2\n负责人：甲"},
    ]
    assert labeled_record_source("我的API是多少？", hits)["doc_id"] == "b"
    assert labeled_record_source("请推荐绩效较高的项目。", hits) is None
    hits.append({"doc_id": "c", "text": "平均绩效（API）：4.5"})
    assert labeled_record_source("我的API是多少？", hits) is None


def test_wire_aliases_bind_to_the_same_actual_source_and_line():
    evidence = [{"chunk_id": "a", "source": "notes.md"}, {"chunk_id": "b", "source": "rules.pdf"}]
    lines = {"E1": (1, "第一句。"), "E2": (2, "第二句。")}
    assert normalize_line_refs(["notes.md", "E1"], lines, evidence) == ["E1"]
    assert normalize_line_refs(["evidence[0].lines[E1]"], lines, evidence) == ["E1"]
    assert normalize_line_refs(["evidence.lines.E2"], lines, evidence) == ["E2"]
    for refs in [
        ["rules.pdf", "E1"],
        ["notes.md"],
        ["evidence[0].lines[E2]"],
        ["notes.md", "E2", "E1"],
    ]:
        with pytest.raises(ValueError):
            normalize_line_refs(refs, lines, evidence)


@pytest.mark.parametrize("encoding", ["utf-8-sig", "utf-16", "utf-32", "gb18030"])
def test_text_encoding_is_lossless_for_supported_encodings(tmp_path, encoding):
    text = "维修站：东岭\n保管员：陈工\n"
    p = tmp_path / "登记.txt"
    p.write_bytes(text.encode(encoding))
    r = TxtConnector().load(ConnectorConfig("txt", str(p), {}))
    assert not r.issues and r.documents[0].content == text
    assert "\ufffd" not in r.documents[0].content


def test_checked_product_is_a_derived_result_not_a_fabricated_source_number():
    q = "如果每个仓库都能装7箱货物，那么所有仓库一共可以装多少箱货物？"
    evidence = [{"chunk_id": "a", "doc_id": "d", "source": "count.txt", "text": "中心有9个仓库。"}]
    value = calculate(q, evidence)
    assert value["result"] == "63" and value["operands"][1]["origin"] == "question_assumption"

    class NoChat:
        def complete(self, messages):
            raise AssertionError("finite calculation needs no generation")

    r = GroundedQA(NoChat(), allow_calculation=True).generate(q, evidence)
    assert r["status"] == "answered" and "63" in r["answer"]
    assert r["claims"][0]["supporting_quotes"] == ["中心有9个仓库。"]
    assert calculate(q, evidence + [{"text": "另有8个仓库。"}]) is None


def test_observed_path_qualified_line_keys_keep_exact_source_ownership():
    evidence = [
        {"chunk_id": "a", "source": "../../records/甲.pdf"},
        {"chunk_id": "b", "source": "说明.docx"},
    ]
    lines = {"E1": (1, "第一句。"), "E2": (2, "第二句。")}
    assert normalize_line_refs(["../../records/甲.pdf:E1", "evidence.E2"], lines, evidence) == [
        "E1",
        "E2",
    ]
    with pytest.raises(ValueError):
        normalize_line_refs(["../../records/甲.pdf:E2"], lines, evidence)
    with pytest.raises(ValueError):
        normalize_line_refs(["different.pdf:E1"], lines, evidence)
