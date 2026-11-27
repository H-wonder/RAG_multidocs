import io
import json

from rag_multidocs.cli import app, repl
from rag_multidocs.config import Settings
from rag_multidocs.core import RAGCore
from rag_multidocs.qa.llm import OpenAICompatibleChat


class PublicFactsChat:
    def __init__(self, model, requests):
        self.model = model
        self.requests = requests

    def stream(self, *_):
        raise AssertionError("The user entrypoint must display only a validated answer")

    def complete(self, messages):
        self.requests.append(self.model)
        assert '信标项目采用陶瓷外壳。' in messages[-1]['content']
        return '信标项目采用陶瓷外壳 [1]。'


def _fixture(tmp_path, monkeypatch):
    settings = Settings(
        data_dir=str(tmp_path / "state"), credentials_file=str(tmp_path / "creds.json"),
        index_backend="keyword", chat_provider="ollama", chat_model="sample-one",
    )
    config = settings.save(tmp_path / "config.json")
    source = tmp_path / "public.txt"
    source.write_text("信标项目采用陶瓷外壳。", encoding="utf-8")
    runtime = app.build_runtime(Settings.load(config))
    runtime.ingest.ingest("txt", str(source))
    requests = []
    monkeypatch.setattr(app, "configured_chat", lambda s, **kw: PublicFactsChat(s.chat_model, requests))
    return config, runtime, requests


def test_cli_and_print_shortcut_call_core_and_safe_trace(tmp_path, monkeypatch, capsys):
    config, runtime, requests = _fixture(tmp_path, monkeypatch)
    build_runtime = app.build_runtime
    counts = []
    original = RAGCore.ask

    def ask(core, question, **kwargs):
        counts.append(core)
        return original(core, question, **kwargs)

    def legacy_search(*args, **kwargs):
        raise AssertionError("A generated answer must not use legacy SearchService")

    monkeypatch.setattr(RAGCore, "ask", ask)
    monkeypatch.setattr(runtime.search, "search", legacy_search)
    monkeypatch.setattr(app, "build_runtime", lambda s: runtime)
    result = app.main(["--config", str(config), "ask", "信标项目使用哪类外壳？",
                       "--json", "--stream", "--trace", "--policy", "lightweight", "--top-k", "3"])
    out = json.loads(capsys.readouterr().out)
    assert result == 0 and counts == [runtime.core] and requests == ["sample-one"]
    assert out["status"] == "answered"
    assert out["citations"][0]["excerpts"] == ["信标项目采用陶瓷外壳。"]
    assert out["trace"]["policy"] == "lightweight"
    assert out["trace"]["reranker"][0]["name"] == "lightweight"
    assert "messages" not in out and "attempts" not in out
    assert "陶瓷" not in json.dumps(out["trace"], ensure_ascii=False)
    shortcut_runtime = build_runtime(Settings.load(config))
    monkeypatch.setattr(app, "build_runtime", lambda s: shortcut_runtime)
    assert app.main(["--config", str(config), "--policy", "stable",
                     "-p", "信标项目使用哪类外壳？", "--trace"]) == 0
    assert counts == [runtime.core, shortcut_runtime.core]


def test_repl_switches_policy_model_and_closes_single_runtime(tmp_path, monkeypatch, capsys):
    config, runtime, requests = _fixture(tmp_path, monkeypatch)
    assert runtime.registry is runtime.core.ingestion.registry
    assert runtime.index is runtime.core.index
    assert runtime.ingest is runtime.core.ingestion
    calls, closes = [], []
    original_close = runtime.core.controller.rerankers.close

    def build(settings):
        calls.append(settings.index_file)
        return runtime

    def close():
        closes.append(True)
        original_close()

    monkeypatch.setattr(app, "build_runtime", build)
    monkeypatch.setattr(runtime.core.controller.rerankers, "close", close)
    lines = iter([
        "信标项目使用哪类外壳？", "/policy lightweight", "/model chat ollama sample-two",
        "/ask 信标项目使用哪类外壳？", "/trace", "/model chat none", "/ask 信标项目使用哪类外壳？",
    ])

    def read(_):
        try:
            return next(lines)
        except StopIteration:
            raise EOFError from None

    monkeypatch.setattr("builtins.input", read)
    assert repl.run_repl(str(config)) == 0
    assert len(calls) == 1 and len(closes) == 1
    assert requests == ["sample-one", "sample-two"]
    out = capsys.readouterr().out
    assert out.count("状态 answered") == 2
    assert '"policy": "lightweight"' in out
    assert "状态 retrieval_only" in out


def test_cli_exposes_provider_failure_without_reasoning_or_unvalidated_content(
    tmp_path, monkeypatch, capsys,
):
    config, runtime, _ = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(app, "build_runtime", lambda s: runtime)
    monkeypatch.setattr(app, "configured_chat", lambda s, **kw: OpenAICompatibleChat(
        "https://fake.invalid/v1", "fake-reasoning",
    ))
    body = {"choices": [{"finish_reason": "stop",
                         "message": {"content": None, "reasoning_content": "private-reasoning"}}]}
    monkeypatch.setattr("rag_multidocs.qa.llm.urllib.request.urlopen",
                        lambda *a, **k: io.BytesIO(json.dumps(body).encode()))
    status = app.main(["--config", str(config), "ask", "信标项目使用哪类外壳？",
                       "--json", "--trace", "--stream"])
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert status == 2 and result["status"] == "generation_failed"
    assert result["citations"] == []
    assert "LLM_REASONING_ONLY" in result["trace"]["warnings"]
    assert "private-reasoning" not in captured.out + captured.err


def test_budget_switch_preserves_the_same_index_and_applies_provider_window(tmp_path, monkeypatch):
    config, runtime, _ = _fixture(tmp_path, monkeypatch)
    settings = Settings.load(config)
    monkeypatch.setattr("rag_multidocs.qa.llm._ollama_context", lambda *args: 16384)
    index = runtime.index
    local = runtime.ask(settings, "信标项目使用哪类外壳？")
    settings.chat_provider = "dashscope"
    settings.chat_model = "deepseek-v4.1-flash-1017"
    remote = runtime.ask(settings, "信标项目使用哪类外壳？")
    assert runtime.index is index and local["status"] == remote["status"] == "answered"
    assert local["token_budget"]["context_window"] == 16384
    assert local["token_budget"]["initial_completion_limit"] == 1024
    assert remote["token_budget"]["context_window"] == 1_000_000
    assert remote["token_budget"]["reserved_completion_limit"] == 12288
    assert remote["retrieval_trace"][0]["resources"]["context_tokens"] == 2000
    runtime.close()


def test_configured_core_reserves_full_evidence_budget_with_6144_completion(tmp_path, monkeypatch):
    from rag_multidocs.services import create_core

    settings = Settings(index_backend="keyword", data_dir=str(tmp_path/"state"),
                        chat_provider="dashscope", chat_model="deepseek-v4.1-flash-1017",
                        chat_max_tokens=6144)
    monkeypatch.setattr("rag_multidocs.services.configured_chat",
                        lambda s: PublicFactsChat(s.chat_model, []))
    core = create_core(settings, base_dir=tmp_path)
    try:
        paths = [tmp_path/"baseline.txt", tmp_path/"reference.txt"]
        for path in paths:
            path.write_text("信标项目采用陶瓷外壳。" + (path.stem[0] + "运行记录。") * 28,
                            encoding="utf-8")
            core.ingest("txt", str(path))
        result = core.ask("根据baseline.txt和reference.txt，信标项目使用哪类外壳？")
        assert len({h["doc_id"] for h in result["generation_evidence"]}) == 2
        assert result["token_budget"]["initial_completion_limit"] == 6144
        assert result["token_budget"]["reserved_completion_limit"] == 10240
        assert result["retrieval_trace"][0]["resources"]["context_tokens"] == 2000
        assert core.controller.policy.model_context_tokens == 8192
    finally:
        core.close()
