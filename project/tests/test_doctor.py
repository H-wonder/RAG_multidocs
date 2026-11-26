"""``rag-cli doctor`` 的单元测试：三态判定、建议文本、渲染与退出码。"""

import json
import sys
import types

from rag_multidocs.cli import main
from rag_multidocs.cli.doctor import CheckResult, render, run_checks
from rag_multidocs.config import Settings


def _settings(tmp_path, project_root, **overrides):
    values = dict(
        data_dir=str(tmp_path / "data"),
        connectors_dir=str(project_root / "connectors"),
        credentials_file=str(tmp_path / ".rag" / "credentials.json"),
        index_backend="keyword",
    )
    values.update(overrides)
    return Settings(**values)


def _config_file(tmp_path, project_root, **overrides):
    settings = _settings(tmp_path, project_root, **overrides)
    path = tmp_path / "config.json"
    settings.save(path)
    return path


def _by_key(results):
    return {r.key: r for r in results}


def test_doctor_healthy_config_passes(tmp_path, project_root, monkeypatch):
    py = tmp_path / "docparse_python"
    py.write_text("")
    settings = _settings(
        tmp_path, project_root,
        docparse_repo=str(project_root),
        docparse_python=str(py),
    )
    monkeypatch.setattr(
        "rag_multidocs.cli.doctor._tcp_reachable",
        lambda host, port, timeout=2.0: True,
    )
    results = run_checks(settings, "config.json")
    by_key = _by_key(results)
    assert by_key["milvus"].status == "pass"
    assert by_key["ollama"].status == "pass"
    assert by_key["docparse"].status == "pass"
    assert by_key["embedding"].status == "pass"  # local 无需 key
    assert by_key["chat"].status == "info"  # 未配置
    assert by_key["connectors"].status == "pass"


def test_milvus_unreachable_fails_when_backend_is_milvus(
    tmp_path, project_root, monkeypatch
):
    settings = _settings(tmp_path, project_root, index_backend="milvus")
    monkeypatch.setattr(
        "rag_multidocs.cli.doctor._tcp_reachable",
        lambda host, port, timeout=2.0: False,
    )
    results = run_checks(settings, "config.json")
    milvus = _by_key(results)["milvus"]
    assert milvus.status == "fail"
    assert milvus.suggestion


def test_milvus_unreachable_warns_when_keyword(tmp_path, project_root, monkeypatch):
    settings = _settings(tmp_path, project_root, index_backend="keyword")
    monkeypatch.setattr(
        "rag_multidocs.cli.doctor._tcp_reachable",
        lambda host, port, timeout=2.0: False,
    )
    results = run_checks(settings, "config.json")
    assert _by_key(results)["milvus"].status == "warn"


def test_unknown_embedding_provider_fails(tmp_path, project_root):
    settings = _settings(tmp_path, project_root, embed_backend="nope")
    results = run_checks(settings, "config.json")
    assert _by_key(results)["embedding"].status == "fail"


def test_docparse_missing_paths_fail(tmp_path, project_root):
    settings = _settings(
        tmp_path, project_root,
        docparse_repo=str(tmp_path / "missing-repo"),
        docparse_python=str(tmp_path / "missing-python"),
    )
    results = run_checks(settings, "config.json")
    assert _by_key(results)["docparse"].status == "fail"


def test_chat_missing_key_warns(tmp_path, project_root, monkeypatch):
    settings = _settings(tmp_path, project_root, chat_provider="deepseek")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    results = run_checks(settings, "config.json")
    chat = _by_key(results)["chat"]
    assert chat.status == "warn"
    assert "DEEPSEEK_API_KEY" in chat.message


def test_credentials_permission_warning_downgraded_on_windows_mount(
    tmp_path, project_root, monkeypatch
):
    cred = tmp_path / ".rag" / "credentials.json"
    cred.parent.mkdir(parents=True)
    cred.write_text('{"keys": {"DEEPSEEK_API_KEY": "x"}}')
    settings = _settings(tmp_path, project_root, credentials_file=str(cred))
    monkeypatch.setattr(
        "rag_multidocs.cli.doctor.permission_warning", lambda path: "权限过宽"
    )
    monkeypatch.setattr(
        "rag_multidocs.cli.doctor._on_windows_mount", lambda path: True
    )
    results = run_checks(settings, "config.json")
    cred_result = _by_key(results)["credentials"]
    assert cred_result.status == "info"
    assert cred_result.suggestion == ""  # 不给出无效的 chmod 建议


def test_render_text_format():
    results = [
        CheckResult("env", "环境", "pass", "ok"),
        CheckResult("milvus", "Milvus", "warn", "down", "start it"),
        CheckResult("chat", "chat", "fail", "bad", "fix it"),
        CheckResult("note", "note", "info", "hint"),
    ]
    out = render(results)
    assert "[PASS] 环境 ok" in out
    assert "[WARN] Milvus down" in out
    assert "建议: start it" in out
    assert "[FAIL] chat bad" in out
    assert "[INFO] note hint" in out


def test_render_json_format():
    results = [CheckResult("env", "环境", "pass", "ok")]
    data = json.loads(render(results, json_output=True))
    assert data[0]["key"] == "env"
    assert data[0]["status"] == "pass"
    assert data[0]["message"] == "ok"


def test_doctor_cli_returns_nonzero_on_fail(
    tmp_path, project_root, monkeypatch, capsys
):
    config = _config_file(tmp_path, project_root, index_backend="milvus")
    monkeypatch.setattr(
        "rag_multidocs.cli.doctor._tcp_reachable",
        lambda host, port, timeout=2.0: False,
    )
    code = main(["doctor", "--config", str(config)])
    out = capsys.readouterr().out
    assert code == 1
    assert "[FAIL]" in out


def test_doctor_cli_json_output(tmp_path, project_root, monkeypatch, capsys):
    config = _config_file(tmp_path, project_root)
    monkeypatch.setattr(
        "rag_multidocs.cli.doctor._tcp_reachable",
        lambda host, port, timeout=2.0: False,
    )
    code = main(["doctor", "--json", "--config", str(config)])
    out = capsys.readouterr().out
    assert code == 0  # keyword 模式，Milvus 不可达只是 warn
    data = json.loads(out)
    assert isinstance(data, list)
    assert any(item["key"] == "milvus" for item in data)


def test_full_skips_milvus_dim_when_unreachable(
    tmp_path, project_root, monkeypatch
):
    settings = _settings(tmp_path, project_root)
    monkeypatch.setattr(
        "rag_multidocs.cli.doctor._tcp_reachable",
        lambda host, port, timeout=2.0: False,
    )
    results = run_checks(settings, "config.json", full=True)
    assert all(r.key != "milvus_dim" for r in results)


def test_full_reports_dim_mismatch(tmp_path, project_root, monkeypatch):
    settings = _settings(tmp_path, project_root, embed_dim=512)

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def has_collection(self, name):
            return True

        def describe_collection(self, name):
            return {"fields": [{"name": "dense", "params": {"dim": 1024}}]}

        def close(self):
            pass

    fake_pymilvus = types.ModuleType("pymilvus")
    fake_pymilvus.MilvusClient = FakeClient
    monkeypatch.setitem(sys.modules, "pymilvus", fake_pymilvus)
    monkeypatch.setattr(
        "rag_multidocs.cli.doctor._tcp_reachable",
        lambda host, port, timeout=2.0: True,
    )

    results = run_checks(settings, "config.json", full=True)
    dim = _by_key(results)["milvus_dim"]
    assert dim.status == "warn"
    assert "1024" in dim.message and "512" in dim.message
