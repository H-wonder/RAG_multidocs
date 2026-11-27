import io
import json

import pytest
from grounded_answer import generate_answer, validate_answer


def evidence():
    return [
        {
            "id": 1,
            "chunk_id": "c1",
            "source": "report.pdf",
            "page": 7,
            "text": "多数类误报增加。",
        }
    ]


def test_citations_use_retrieved_source_and_page():
    result = validate_answer(
        {
            "insufficient_evidence": False,
            "claims": [
                {
                    "text": "多数类误报增加。",
                    "citation_ids": [1, 1],
                    "supporting_quotes": ["多数类误报增加。", "多数类误报增加。"],
                },
            ],
        },
        evidence(),
    )
    assert result["answer"] == "多数类误报增加。 [1，p7]"
    assert result["citations"] == evidence()


@pytest.mark.parametrize("refs", [[], [2], [0], [True], ["1"], None])
def test_reject_missing_or_invented_citations(refs):
    with pytest.raises(ValueError, match="引用编号"):
        validate_answer(
            {
                "insufficient_evidence": False,
                "claims": [
                    {"text": "回答", "citation_ids": refs},
                ],
            },
            evidence(),
        )


def test_insufficient_evidence_has_no_fabricated_answer():
    result = validate_answer({"insufficient_evidence": True, "claims": []}, evidence())
    assert result["status"] == "INSUFFICIENT_EVIDENCE"
    assert result["citations"] == []
    with pytest.raises(ValueError, match="证据不足"):
        validate_answer(
            {
                "insufficient_evidence": True,
                "claims": [{"text": "猜测", "citation_ids": [1]}],
            },
            evidence(),
        )


def test_no_evidence_does_not_call_ollama(monkeypatch):
    def unexpected_call(*args, **kwargs):
        raise AssertionError("不应调用 Ollama")

    monkeypatch.setattr("urllib.request.build_opener", unexpected_call)
    assert generate_answer("问题", [])["status"] == "INSUFFICIENT_EVIDENCE"


def test_local_demo_rejects_remote_endpoint():
    with pytest.raises(ValueError, match="loopback"):
        generate_answer("问题", [], base_url="https://example.com")


@pytest.mark.parametrize("quotes", [[], ["并未增加误报。"], [""]])
def test_reject_fabricated_or_missing_supporting_quote(quotes):
    with pytest.raises(ValueError, match="摘录"):
        validate_answer(
            {
                "insufficient_evidence": False,
                "claims": [
                    {"text": "回答", "citation_ids": [1], "supporting_quotes": quotes},
                ],
            },
            evidence(),
        )


def test_supporting_quote_tolerates_pdf_line_breaks():
    result = validate_answer(
        {
            "insufficient_evidence": False,
            "claims": [
                {
                    "text": "误报增加。",
                    "citation_ids": [1],
                    "supporting_quotes": ["多数类\n误报增加。"],
                },
            ],
        },
        evidence(),
    )
    assert result["status"] == "ANSWERED"


def test_generation_keeps_valid_claim_and_records_rejected_one(monkeypatch):
    payload = {
        "insufficient_evidence": False,
        "claims": [
            {
                "text": "误报增加。",
                "citation_ids": [1],
                "supporting_quotes": ["多数类误报增加。"],
            },
            {
                "text": "准确率达到100%。",
                "citation_ids": [1],
                "supporting_quotes": ["准确率达到100%。"],
            },
        ],
    }

    class FakeOpener:
        def open(self, request, timeout):
            return io.StringIO(
                json.dumps({"done": True, "message": {"content": json.dumps(payload)}})
            )

    monkeypatch.setattr("urllib.request.build_opener", lambda *args: FakeOpener())
    result = generate_answer(
        "结果如何？",
        [{"id": "c1", "source": "report.pdf", "page": 7, "text": "多数类误报增加。"}],
    )
    assert result["answer"] == "误报增加。 [1，p7]"
    assert len(result["discarded_claims"]) == 1
    assert "100%" not in result["answer"]


def test_invalid_model_response_retries_once_then_fails(monkeypatch):
    calls = []

    class FakeOpener:
        def open(self, request, timeout):
            calls.append(request)
            return io.StringIO(json.dumps({"done": True, "message": {"content": "{}"}}))

    monkeypatch.setattr("urllib.request.build_opener", lambda *args: FakeOpener())
    with pytest.raises(ValueError):
        generate_answer(
            "结果如何？",
            [
                {
                    "id": "c1",
                    "source": "report.pdf",
                    "page": 7,
                    "text": "多数类误报增加。",
                }
            ],
        )
    assert len(calls) == 2
