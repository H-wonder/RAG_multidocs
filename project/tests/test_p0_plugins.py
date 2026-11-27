import hashlib
import json
import multiprocessing
from pathlib import Path

from rag_multidocs.connectors.registry import ConnectorRegistry
from rag_multidocs.contracts import ConnectorConfig
from rag_multidocs.rerankers.registry import RerankerRegistry
from test_a_plus_plugins import candidates, write_plugin


def test_discovery_diagnostics_duplicate_dependency_and_version(tmp_path):
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "manifest.json").write_text("{")
    write_plugin(tmp_path, "duplicate", "def rank(r): return []")
    second = tmp_path / "second"
    second.mkdir()
    (second / "manifest.json").write_text((tmp_path / "duplicate/manifest.json").read_text())
    registry = RerankerRegistry(tmp_path)
    result, warnings = registry.rerank("duplicate", "q", candidates())
    assert result == candidates() and warnings[0]["code"] == "RERANKER_DUPLICATE"
    assert {w["code"] for w in registry.discovery_issues} == {
        "MANIFEST_INVALID",
        "PLUGIN_DUPLICATE",
    }
    plugin = write_plugin(tmp_path, "version", "def rank(r): raise AssertionError()")
    manifest = json.loads((plugin / "manifest.json").read_text())
    manifest["contract_version"] = "99"
    (plugin / "manifest.json").write_text(json.dumps(manifest))
    assert registry.rerank("version", "q", candidates())[1][0]["code"] == "RERANKER_VERSION_ERROR"
    manifest.update(contract_version="1", requires=["nonexistent_p0_dependency>=1.0"])
    (plugin / "manifest.json").write_text(json.dumps(manifest))
    assert (
        registry.rerank("version", "q", candidates())[1][0]["code"] == "RERANKER_DEPENDENCY_MISSING"
    )


def test_connector_legacy_allowlist_and_relative_helper_other_cwd(tmp_path, monkeypatch):
    plugin = write_plugin(
        tmp_path / "plugins",
        "new.source",
        """from .helper import result

def load(payload, context): return result()
""",
        kind="connector",
        enabled=False,
    )
    (plugin / "helper.py").write_text("""from rag_multidocs.contracts import ConnectorResult
from rag_multidocs.model import Document

def result(): return ConnectorResult([Document.from_text("independent knowledge", "external")])
""")
    root = Path(__file__).resolve().parents[1] / "src"
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*.py")}
    registry = ConnectorRegistry(plugin.parent, enabled={"new.source"})
    monkeypatch.chdir(tmp_path)
    assert (
        registry.load("new.source", ConnectorConfig("new.source", "unused")).documents[0].source
        == "external"
    )
    assert before == {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*.py")
    }
    registry.enabled = set()
    assert (
        registry.load("new.source", ConnectorConfig("new.source", "x")).issues[0].code
        == "CONNECTOR_DISABLED"
    )


def test_worker_start_failure_and_timeout_reaped(tmp_path, monkeypatch):
    write_plugin(tmp_path, "slow", "import time\ndef rank(r): time.sleep(3)")
    registry = RerankerRegistry(tmp_path, timeout_s=0.15)
    previous = {p.pid for p in multiprocessing.active_children()}
    assert registry.rerank("slow", "q", candidates())[1][0]["code"] == "RERANKER_TIMEOUT"
    assert not registry.last_execution["worker_alive"]
    assert {p.pid for p in multiprocessing.active_children()} <= previous
    monkeypatch.setattr(
        multiprocessing.process.BaseProcess,
        "start",
        lambda self: (_ for _ in ()).throw(RuntimeError("start failed")),
    )
    assert registry.rerank("slow", "q", candidates())[1][0]["code"] == "RERANKER_START_ERROR"


def test_untrusted_manifest_cannot_choose_inprocess(tmp_path):
    plugin = write_plugin(
        tmp_path,
        "external",
        "from rag_multidocs.contracts import RankedCandidate\ndef rank(r): return [RankedCandidate(h.candidate_id,i+1,1-i*.1) for i,h in enumerate(r.candidates)]",
    )
    manifest = json.loads((plugin / "manifest.json").read_text())
    manifest["execution"] = "inprocess"
    (plugin / "manifest.json").write_text(json.dumps(manifest))
    registry = RerankerRegistry(tmp_path)
    assert not registry.rerank("external", "q", candidates())[1]
    assert registry.last_execution["mode"] == "process"
    registry = RerankerRegistry(tmp_path, trusted_inprocess=[plugin])
    assert not registry.rerank("external", "q", candidates())[1]
    assert registry.last_execution["mode"] == "trusted_inprocess_no_hard_timeout"
