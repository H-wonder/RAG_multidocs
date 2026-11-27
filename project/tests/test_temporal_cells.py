import pytest
from rag_multidocs.qa.grounding import validate_claim
from rag_multidocs.qa.table_evidence import temporal_equal, table_view
from rag_multidocs.qa.tasks import plan_tasks


def table():
    text = "时段 | 年级 | 课程\n2027年3月6日 下午14:20-16:10 | 2026级 | 算法原理\n2027年3月8日 上午8:40-10:20 | 2026级 | 概率方法"
    return {
        "chunk_id": "t",
        "doc_id": "d",
        "source": "arrangements.pdf",
        "content_type": "table",
        "table_headers": ["时段", "年级", "课程"],
        "text": text,
    }


def test_temporal_equivalence_keeps_calendar_and_range_order():
    assert temporal_equal("2027年3月6日 下午14:20-16:10", "2027/03/06 下午2:20至4:10")
    assert not temporal_equal("2027年3月6日 下午14:20-16:10", "2027/03/07 14:20至16:10")
    assert not temporal_equal("14:20-16:10", "16:10至14:20")
    assert not temporal_equal("14:20-16:10", "14:20")


def test_reordered_table_columns_keep_entities_and_complete_time_ranges():
    hit = table()
    view = table_view(hit, "算法原理和概率方法分别在什么时间考试？")
    assert [r["row_label"] for r in view["rows"]] == ["算法原理", "概率方法"]
    tasks = plan_tasks("算法原理和概率方法分别在什么时间考试？", [hit])
    assert len(tasks) == 2 and {t["target_cell"]["column"] for t in tasks} == {"时段"}
    assert all("16:10" in t["expected_value"] or "10:20" in t["expected_value"] for t in tasks)


def test_equivalent_date_expression_does_not_weaken_wrong_date_rejection():
    hit = table()
    quote = hit["text"].splitlines()[1]
    good = {
        "text": "算法原理的时段为2027-03-06 14:20至16:10。",
        "citation_ids": [1],
        "supporting_quotes": [quote],
    }
    assert validate_claim(good, [hit])
    for text in [
        "算法原理的时段为2027-03-08 14:20至16:10。",
        "算法原理的时段为2027-03-06 16:10至14:20。",
    ]:
        with pytest.raises(ValueError):
            validate_claim({**good, "text": text}, [hit])
