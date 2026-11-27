import io
import json
from urllib.error import HTTPError

import pytest

from rag_multidocs.execution import request_scope
from rag_multidocs.qa.llm import (
    ChatCompletionError,
    OpenAICompatibleChat,
    answer_with_evidence,
    build_evidence,
    build_messages,
    reasoning_output_budget,
    resolve_chat_budget,
)


class FakeChat:
    def __init__(self):
        self.messages = None

    def complete(self, messages):
        self.messages = messages
        return "检索器负责取回段落 [1]。"


def _hits():
    return [
        {
            "chunk_id": "c1",
            "doc_id": "d1",
            "source": "notes.txt",
            "page": 2,
            "section_path": "检索 > 训练",
            "text": "检索器负责从外部知识库取回相关段落。",
            "score": 0.9,
        }
    ]


def test_evidence_contains_source_and_section():
    evidence = build_evidence(_hits())
    assert "[1] notes.txt p2 §检索 > 训练" in evidence
    assert "检索器" in evidence


def test_answer_is_grounded_and_has_citations():
    chat = FakeChat()
    result = answer_with_evidence("检索器做什么", _hits(), chat)

    assert result["used_llm"] is True
    assert "[1]" in result["answer"]
    assert result["citations"][0]["source"] == "notes.txt"
    assert result["citations"][0]["page"] == 2
    assert chat.messages is not None
    assert "证据" in chat.messages[1]["content"]
    assert "notes.txt" in chat.messages[1]["content"]


def test_no_hits_does_not_call_model():
    class ExplodingChat:
        def complete(self, messages):
            raise AssertionError("没有证据时不应调用模型")

    result = answer_with_evidence("不存在的问题", [], ExplodingChat())
    assert result["used_llm"] is False
    assert result["citations"] == []
    assert "没有检索到" in result["answer"]


def test_messages_include_numbered_evidence():
    messages = build_messages("问题", _hits())
    assert messages[0]["role"] == "system"
    assert "[1]" in messages[1]["content"]


def test_answer_streams_when_model_supports_it():
    class StreamChat:
        def stream(self, messages, on_token):
            on_token("检索器")
            on_token("负责取回段落 [1]。")
            return "检索器负责取回段落 [1]。"

        def complete(self, messages):
            raise AssertionError("有 stream 时不应调用 complete")

    tokens = []
    result = answer_with_evidence("检索器做什么", _hits(), StreamChat(), on_token=tokens.append)

    assert "".join(tokens) == result["answer"]
    assert result["used_llm"] is True
    assert result["citations"][0]["source"] == "notes.txt"


def test_answer_falls_back_to_complete_with_on_token():
    tokens = []
    result = answer_with_evidence(
        "检索器做什么", _hits(), FakeChat(), on_token=tokens.append
    )

    assert tokens == [result["answer"]]
    assert "[1]" in result["answer"]


def _completion(content, finish="stop", reasoning=None, **extra):
    return {"choices": [{"message": {"content": content, "reasoning_content": reasoning},
                         "finish_reason": finish}], **extra}


def _responses(monkeypatch, *items):
    pending = iter(items)
    bodies = []

    def respond(request, **kwargs):
        bodies.append(json.loads(request.data))
        item = next(pending)
        if isinstance(item, Exception):
            raise item
        return io.BytesIO(item if isinstance(item, bytes) else json.dumps(item).encode())

    monkeypatch.setattr("rag_multidocs.qa.llm.urllib.request.urlopen", respond)
    return bodies


@pytest.mark.parametrize(("message", "code"), [
    (_completion(None), "LLM_EMPTY_CONTENT"),
    (_completion("   "), "LLM_EMPTY_CONTENT"),
    (_completion(None, "stop", "hidden"), "LLM_REASONING_ONLY"),
    (_completion(None, "length"), "LLM_OUTPUT_TRUNCATED"),
    (_completion("unfinished answer", "length", "hidden"), "LLM_OUTPUT_TRUNCATED"),
    (_completion(None, "content_filter"), "LLM_CONTENT_FILTERED"),
    (_completion(None, "tool_calls"), "LLM_UNSUPPORTED_FINISH"),
    ({"error": {"message": "synthetic secret diagnostic"}}, "LLM_API_ERROR"),
    ({"choices": []}, "LLM_PROTOCOL_ERROR"),
])
def test_completion_reports_protocol_state_without_leaking_details(monkeypatch, message, code):
    bodies = _responses(monkeypatch, message)
    client = OpenAICompatibleChat("https://fake.invalid/v1", "test", reasoning_retry_tokens=2048)
    with pytest.raises(ChatCompletionError) as exc:
        client.complete([])
    assert exc.value.code == code
    assert "secret" not in str(exc.value) and "hidden" not in str(exc.value)
    assert len(bodies) == 1


def test_reasoning_exhaustion_retries_once_using_bounded_tokens(monkeypatch):
    budget = resolve_chat_budget("dashscope", "deepseek-v4.1-flash-1017", 6144)
    only_reasoning = _completion(None, "length", usage={
        "prompt_tokens": 87, "completion_tokens": 6144,
        "completion_tokens_details": {"reasoning_tokens": 6144},
    })
    calls = _responses(monkeypatch, only_reasoning, _completion("Final answer", reasoning="hidden"))
    client = OpenAICompatibleChat("https://fake.invalid/v1", "test", budget=budget)
    with request_scope() as scope:
        assert client.complete([]) == "Final answer"
    assert [body["max_tokens"] for body in calls] == [6144, 10240]
    assert calls[0]["reasoning_effort"] == "low" and "thinking_budget" not in calls[0]
    assert calls[1]["enable_thinking"] is False and "reasoning_effort" not in calls[1]
    assert scope.counters["reasoning_budget_retries"] == 1
    assert scope.counters["reported_reasoning_tokens"] == 6144
    assert reasoning_output_budget("openai", "ordinary", 1024) == 1024
    assert reasoning_output_budget("dashscope", "deepseek-v4.1-flash", 4096) == 8192
    calls = _responses(monkeypatch, only_reasoning, only_reasoning)
    with pytest.raises(ChatCompletionError) as exc:
        client.complete([])
    assert exc.value.code == "LLM_REASONING_TRUNCATED" and len(calls) == 2


def test_thinking_only_retry_can_free_answer_space_at_same_limit(monkeypatch):
    budget = resolve_chat_budget("dashscope", "deepseek-v4.1-flash-1017", 4096, 8192)
    assert budget.retry_tokens == budget.initial_tokens == 4096
    calls = _responses(monkeypatch, _completion(None, "length", "thinking"), _completion("Final"))
    client = OpenAICompatibleChat("https://fake.invalid/v1", "test", budget=budget)
    assert client.complete([]) == "Final"
    assert [b["max_tokens"] for b in calls] == [4096, 4096]
    assert calls[1]["enable_thinking"] is False


def test_truncated_visible_answer_retries_with_more_space_without_changing_effort(monkeypatch):
    budget = resolve_chat_budget("dashscope", "deepseek-v4.1-flash-1017")
    calls = _responses(monkeypatch, _completion("partial", "length"), _completion("Final"))
    with request_scope() as scope:
        assert OpenAICompatibleChat("https://fake.invalid/v1", "test", budget=budget).complete([]) == "Final"
    assert [b["max_tokens"] for b in calls] == [8192, 12288]
    assert all(b["reasoning_effort"] == "low" and "enable_thinking" not in b for b in calls)
    assert scope.counters["completion_budget_retries"] == 1


def test_ollama_auto_window_uses_allocated_context_without_reasoning_parameters(monkeypatch):
    def running(url, **kwargs):
        assert url.endswith("/api/ps")
        return io.BytesIO(b'{"models":[{"model":"gemma3:4b","context_length":16384}]}')

    monkeypatch.setattr("rag_multidocs.qa.llm.urllib.request.urlopen", running)
    budget = resolve_chat_budget("ollama", "gemma3:4b")
    assert budget.context_window == 16384 and budget.initial_tokens == budget.retry_tokens == 1024
    client = OpenAICompatibleChat("http://localhost:11434/v1", "gemma3:4b", budget=budget)
    body = json.loads(client._payload([], False))
    assert not {"thinking_budget", "enable_thinking", "reasoning_effort"} & body.keys()
    assert resolve_chat_budget("ollama", "another:4b").context_window == 4096


def _stream_events(*events, done=True):
    text = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    return (text + ("data: [DONE]\n\n" if done else "")).encode()


def _delta(**delta):
    return {"choices": [{"delta": delta, "finish_reason": None}]}


def test_stream_preserves_answer_only_and_usage_event(monkeypatch):
    bodies = _responses(monkeypatch, b": heartbeat\n\ndata:\n\n" + _stream_events(
        _delta(reasoning_content="hidden"), _delta(content="Final"), _delta(content=" answer"),
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"completion_tokens": 13}},
    ))
    client = OpenAICompatibleChat("https://fake.invalid/v1", "test")
    tokens = []
    with request_scope() as scope:
        assert client.stream([], tokens.append) == "Final answer"
    assert tokens == ["Final", " answer"] and bodies[0]["stream"]
    assert scope.counters["reported_completion_tokens"] == 13


def test_stream_interruption_does_not_restart_after_answer_tokens(monkeypatch):
    bodies = _responses(monkeypatch, _stream_events(_delta(content="partial"), done=False))
    tokens = []
    client = OpenAICompatibleChat("https://fake.invalid/v1", "test")
    with pytest.raises(ChatCompletionError) as exc:
        client.stream([], tokens.append)
    assert exc.value.code == "LLM_STREAM_INCOMPLETE"
    assert tokens == ["partial"] and len(bodies) == 1


def test_permission_http_error_is_distinct_and_does_not_retry(monkeypatch):
    error = HTTPError("https://fake.invalid/v1", 401, "sensitive", {}, io.BytesIO(b"sensitive"))
    calls = _responses(monkeypatch, error)
    with pytest.raises(ChatCompletionError) as exc:
        OpenAICompatibleChat("https://fake.invalid/v1", "test").complete([])
    assert exc.value.code == "LLM_HTTP_ERROR" and "401" in str(exc.value)
    assert "sensitive" not in str(exc.value) and len(calls) == 1
