"""Observe the unchanged CLI and same public Core components in isolated data."""

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
import subprocess
import sys
import uuid

from rag_local import PROJECT, settings, chat
from rag_multidocs.cli.app import build_runtime
from rag_multidocs.core import RAGCore
from rag_multidocs.retrieval.controller import RetrievalPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=["milvus", "keyword"], default="milvus")
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use a new directory")
    out.mkdir(parents=True, exist_ok=True)
    config = settings(out, "p1_joint_" + uuid.uuid4().hex[:10], args.backend)
    config.chat_provider = "ollama"
    config.chat_model = "gemma3:4b"
    config.chat_base_url = "http://127.0.0.1:11434/v1"
    config.chat_temperature = 0
    config.chat_max_tokens = 700
    config.credentials_file = str(out / "unused_credentials.json")
    path = out / "config.json"
    path.write_text(json.dumps(asdict(config), indent=2))
    file = out / "joint_warranty.txt"
    file.write_text("设备保修期为12个月。")
    before = {
        str(p.relative_to(PROJECT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (PROJECT / "src/rag_multidocs/cli").rglob("*.py")
    }
    result = {
        "backend": args.backend,
        "collection": config.milvus_collection,
        "checks": {},
        "scope": "Legacy CLI shared ingestion/index is tested; legacy generation is unvalidated compatibility, not enhanced Core integration.",
    }
    env = dict(
        os.environ,
        PYTHONPATH=str(PROJECT / "src"),
        PYTHONDONTWRITEBYTECODE="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )

    def command(key, args):
        value = subprocess.run(
            [sys.executable, "-m", "rag_multidocs", "--config", str(path), *args],
            cwd=PROJECT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=120,
        )
        (out / (key + ".stdout")).write_text(value.stdout)
        (out / (key + ".stderr")).write_text(value.stderr)
        if value.returncode:
            raise RuntimeError("CLI " + key + " failed " + str(value.returncode))
        return json.loads(value.stdout)

    runtime = None
    service = None
    try:
        result["old_ingest"] = command(
            "old_ingest", ["ingest", "--source", "txt", "--path", str(file), "--json"]
        )
        result["old_search"] = command("old_search", ["search", "设备保修期", "--json"])
        result["old_answer"] = command("old_answer", ["ask", "设备保修期是多少？", "--json"])
        # Existing public constructor needs no UI modification or second index.
        runtime = build_runtime(config)
        service = RAGCore(
            runtime.registry,
            runtime.index,
            policy=RetrievalPolicy.baseline(atomic_generation=True),
            chat=chat(),
        )
        result["checks"]["same_index_object"] = service.index is runtime.index
        result["checks"]["same_catalog_object"] = (
            service.ingestion.catalog is runtime.ingest.catalog
        )
        result["new_answer"] = service.ask("设备保修期是多少？", style="concise")
        result["checks"]["new_core_fact"] = "12" in result["new_answer"]["answer"] and bool(
            result["new_answer"]["citations"]
        )
        file.write_text("设备保修期为24个月。")
        result["core_update"] = service.ingest("txt", str(file)).as_dict()
        service.close()
        service = None
        result["old_after_core_update"] = command(
            "old_after_core_update", ["search", "设备保修期", "--json"]
        )
        # CLI search JSON currently returns a list, preserve that old contract.
        rows = result["old_after_core_update"]
        if isinstance(rows, dict):
            rows = rows.get("hits", [])
        result["checks"]["old_reads_core_active_version"] = bool(rows) and all(
            "24个月" in h["text"] and "12个月" not in h["text"] for h in rows
        )
        result["checks"]["legacy_qa_not_misreported_as_core"] = (
            "status" not in result["old_answer"]
            and "generation_evidence" not in result["old_answer"]
        )
    except Exception as exc:
        result['error']=type(exc).__name__+': '+str(exc)[:300]
        raise
    finally:
        if service:
            service.close()
        after = {
            str(p.relative_to(PROJECT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (PROJECT / "src/rag_multidocs/cli").rglob("*.py")
        }
        result["checks"]["interaction_source_unchanged"] = bool(before) and before == after
        result["interaction_hashes"] = {"before": before, "after": after}
        result["status"] = (
            "PASS" if result["checks"] and all(result["checks"].values()) and not result.get('error') else "PARTIAL"
        )
        result["enhanced_cli_integration"] = "PENDING_TEAM"
        (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(result["status"], result["checks"], "enhanced_cli_integration=PENDING_TEAM")


if __name__ == "__main__":
    main()
