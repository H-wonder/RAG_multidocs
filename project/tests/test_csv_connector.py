import csv
import io
from pathlib import Path
import pytest
from rag_multidocs.connectors.registry import ConnectorRegistry
from rag_multidocs.contracts import ConnectorConfig
from rag_multidocs.ingestion.structured import structured_chunks
from rag_multidocs.qa.table_evidence import table_view


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig", "gbk", "gb18030"])
def test_csv_uses_existing_table_contract_without_losing_quoted_or_empty_values(tmp_path, encoding):
    values = [
        ["标识", "单价", "数量", "备注"],
        ["K-41", "3.500", "2", '待检,需复查\n保留"双引号"|分隔'],
        ["K-42", "", "5", ""],
    ]
    stream = io.StringIO(newline="")
    csv.writer(stream).writerows(values)
    p = tmp_path / "arbitrary.csv"
    p.write_bytes(stream.getvalue().encode(encoding))
    registry = ConnectorRegistry(Path(__file__).resolve().parents[1] / "connectors")
    result = registry.load("connector.excel", ConnectorConfig("connector.excel", str(p), {}))
    assert not result.issues and len(result.documents) == 1
    document = result.documents[0]
    assert document.segments[0]["meta"]["table_data"]["rows"] == values
    chunks = structured_chunks(document)
    view = table_view({**chunks[0].as_dict(), "chunk_id": chunks[0].id})
    assert view["rows"][0]["cells"]["单价"] == "3.500"
    assert view["rows"][0]["cells"]["备注"] == values[1][3]
    assert view["rows"][1]["row_label"] == "K-42" and "单价" not in view["rows"][1]["cells"]
    assert all(row["supporting_quote"] in chunks[0].text for row in view["rows"])


def test_malformed_csv_reports_error_instead_of_shifting_columns(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("项目,费用\n甲,1,2\n", encoding="utf8")
    registry = ConnectorRegistry(Path(__file__).resolve().parents[1] / "connectors")
    result = registry.load("connector.excel", ConnectorConfig("connector.excel", str(p), {}))
    assert not result.documents and result.issues[0].code == "CSV_COLUMN_COUNT_MISMATCH"


def test_reordered_csv_calculation_uses_explicit_header_unit(tmp_path):
    from rag_multidocs.qa.calculations import calculate
    from rag_multidocs.qa.grounding import validate_claim

    p = tmp_path / "cost.csv"
    p.write_text("费用（元）,项目\n18.750,风扇\n6.250,台灯\n", encoding="utf8")
    registry = ConnectorRegistry(Path(__file__).resolve().parents[1] / "connectors")
    document = registry.load(
        "connector.excel", ConnectorConfig("connector.excel", str(p), {})
    ).documents[0]
    chunk = structured_chunks(document)[0]
    hit = {**chunk.as_dict(), "chunk_id": chunk.id}
    calculation = calculate("风扇和台灯的费用合计是多少？", [hit])
    assert calculation["result"] == "25.000" and calculation["unit"] == "元"
    assert [v["value"] for v in calculation["operands"]] == ["18.750", "6.250"]
    claim = {
        "text": "风扇的费用为18.750元。",
        "citation_ids": [1],
        "supporting_quotes": ["18.750 | 风扇"],
    }
    assert validate_claim(claim, [hit])
    for wrong in ["台灯的费用为18.750元。", "风扇的费用为18.750%。"]:
        with pytest.raises(ValueError):
            validate_claim({**claim, "text": wrong}, [hit])


def test_blank_csv_cell_is_not_zero_or_a_neighboring_field(tmp_path):
    from rag_multidocs.qa.tasks import plan_tasks
    from rag_multidocs.qa.grounding import GroundedQA
    from rag_multidocs.retrieval.sources import resolve_sources

    p = tmp_path / "设备费用.csv"
    p.write_text("费用（元）,项目,数量\n,线缆,1\n", encoding="utf8")
    registry = ConnectorRegistry(Path(__file__).resolve().parents[1] / "connectors")
    doc = registry.load(
        "connector.excel", ConnectorConfig("connector.excel", str(p), {})
    ).documents[0]
    c = structured_chunks(doc)[0]
    h = {**c.as_dict(), "chunk_id": c.id}
    q = "根据设备费用.csv，线缆的数量是多少？线缆的费用是否填写？"
    scope = resolve_sources(q, [{"doc_id": doc.doc_id, "source": doc.source}])
    tasks = plan_tasks(q, [h], scope)
    assert [t["expected_value"] for t in tasks] == ["1", ""]

    class NoChat:
        def complete(self, m):
            raise AssertionError("literal cells need no generation")

    r = GroundedQA(NoChat(), structured_table_answers=True).generate(
        q, [h], source_constraints=scope
    )
    assert r["status"] == "answered" and "未填写" in r["answer"] and "数量：1" in r["answer"]
