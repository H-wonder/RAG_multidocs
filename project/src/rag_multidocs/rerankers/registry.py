"""Manifest-selected rerankers with ID-only output and exact stage fallback.

Trusted host-approved in-process code has NO hard cancellation guarantee.
Other plugins use a killable per-call process, not a security sandbox.
"""

from __future__ import annotations

import copy
import json
import math
import multiprocessing as mp
import time
from pathlib import Path

from ..contracts import RankedCandidate, RerankRequest, SearchHit
from ..extensions import dependency_error, discover, load_entry


def _worker(connection, info, request):
    import resource

    started = time.monotonic()
    try:
        entry = load_entry(info)
        raw = entry(request)
        try:
            raw = _validate(raw, [h.candidate_id for h in request.candidates])
            data = {"status": "ok", "rows": [vars(r) for r in raw]}
        except (ValueError, TypeError, AttributeError):
            data = {"status": "invalid"}
        data["worker_metrics"] = {
            "seconds": time.monotonic() - started,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "diagnostics": entry.diagnostics() if callable(getattr(entry, "diagnostics", None)) else {},
        }
        payload = json.dumps(data).encode()
        if len(payload) > 131072:
            payload = b'{"status":"invalid"}'
        connection.send_bytes(payload)
    except Exception:
        connection.send_bytes(b'{"status":"error"}')
    finally:
        connection.close()


def _validate(raw, ids):
    if not isinstance(raw, list) or len(raw) != len(ids):
        raise ValueError("Result must cover all candidates")
    if any(not isinstance(r, RankedCandidate) for r in raw):
        raise ValueError("Invalid output type")
    if {r.candidate_id for r in raw} != set(ids) or len({r.candidate_id for r in raw}) != len(ids):
        raise ValueError("Invalid candidate IDs")
    if any(type(r.rank) is not int for r in raw) or sorted(r.rank for r in raw) != list(
        range(1, len(ids) + 1)
    ):
        raise ValueError("Ranks must be a permutation of 1..N")
    if any(
        type(r.score) not in (float, int) or not math.isfinite(r.score) or not 0 <= r.score <= 1
        for r in raw
    ):
        raise ValueError("Scores must be finite and in [0,1]")
    ordered = sorted(raw, key=lambda r: r.rank)
    if any(a.score < b.score for a, b in zip(ordered, ordered[1:])):
        raise ValueError("Scores disagree with ranks")
    return ordered


class RerankerRegistry:
    def __init__(self, plugins_dir=None, enabled=None, timeout_s=10, *, trusted_inprocess=(), persistent_paths=(), max_worker_restarts=1):
        bundled = Path(__file__).resolve().parents[3] / "rerankers"
        self.plugins_dir = Path(plugins_dir or bundled).resolve()
        # A manifest cannot grant itself in-process privileges. Only host-approved paths can.
        self.trusted_inprocess = {Path(p).resolve() for p in trusted_inprocess}
        if plugins_dir is None:
            self.trusted_inprocess.add((bundled / "lightweight").resolve())
        self.enabled = set(enabled) if enabled is not None else None
        if not 0 < timeout_s <= 120:
            raise ValueError("timeout_s must be in (0,120]")
        self.timeout_s = timeout_s
        self.discovery_issues = []
        self.last_execution = {}
        self.persistent_paths = {Path(p).resolve() for p in persistent_paths}
        from .persistent import PersistentWorker
        self.persistent_worker = PersistentWorker(max_worker_restarts)

    def close(self):
        self.persistent_worker.close()

    def plugin_infos(self):
        self.discovery_issues = []
        return discover(
            self.plugins_dir, self.enabled, kind="reranker", issues=self.discovery_issues
        )

    def rerank(self, name, query, hits, config=None, *, timeout_s=None):
        baseline = copy.deepcopy(hits)
        self.last_execution = {"mode": "none", "seconds": 0.0}
        started = time.monotonic()

        def fallback(code, message):
            self.last_execution["seconds"] = time.monotonic() - started
            return baseline, [
                {
                    "code": code,
                    "message": message,
                    "plugin": name,
                    "fallback": "rrf",
                    "severity": "warning",
                }
            ]

        if name == "none" or not hits:
            return baseline, []
        infos = [i for i in self.plugin_infos() if i.name == name]
        if len(infos) != 1:
            code = "RERANKER_DUPLICATE" if len(infos) > 1 else "RERANKER_NOT_FOUND"
            return fallback(code, "Plugin missing or name not unique")
        info = infos[0]
        if not info.enabled:
            return fallback("RERANKER_DISABLED", "Plugin disabled")
        if info.contract_version != "1":
            return fallback("RERANKER_VERSION_ERROR", "Unsupported contract version")
        dependency = dependency_error(info)
        if dependency:
            return fallback("RERANKER_" + dependency, "Plugin dependency unavailable")
        try:
            ids = [str(h.get("chunk_id", h.get("id", ""))) for h in hits]
            if (
                len(set(ids)) != len(ids)
                or not all(ids)
                or any(not isinstance(h.get("text"), str) for h in hits)
            ):
                return fallback("RERANKER_INPUT_ERROR", "Candidate ID/text invalid")
            request = RerankRequest(
                query,
                tuple(
                    SearchHit(
                        key,
                        ((h.get("section_path") + "\n") if h.get("section_path") else "") + h["text"],
                        h.get("source", ""),
                        h.get("doc_id", ""),
                        float(h.get("score", 0)),
                        h.get("semantic_score"),
                    )
                    for key, h in zip(ids, hits)
                ),
                copy.deepcopy({**info.config, **(config or {})}),
            )
        except (ValueError, TypeError, KeyError):
            return fallback("RERANKER_INPUT_ERROR", "Invalid candidates/configuration")
        budget = min(self.timeout_s, timeout_s if timeout_s is not None else self.timeout_s)
        if budget <= 0:
            return fallback("RERANKER_TIMEOUT", "Request deadline expired")
        if info.path in self.persistent_paths:
            try:
                raw = self.persistent_worker.execute(info, request, budget)
                self.last_execution = dict(self.persistent_worker.last)
            except TimeoutError:
                self.last_execution = dict(self.persistent_worker.last)
                return fallback("RERANKER_TIMEOUT", "Persistent execution deadline exceeded; worker reaped")
            except Exception as exc:
                self.last_execution = dict(self.persistent_worker.last)
                code = "RERANKER_INVALID_OUTPUT" if "INVALID" in str(exc) or "MISMATCH" in str(exc) else "RERANKER_EXECUTION_ERROR"
                return fallback(code, "Persistent worker failed; baseline retained")
        elif info.execution == "inprocess" and info.path in self.trusted_inprocess:
            self.last_execution["mode"] = "trusted_inprocess_no_hard_timeout"
            try:
                raw = load_entry(info)(request)
            except Exception:
                return fallback("RERANKER_EXECUTION_ERROR", "Trusted plugin failed")
        else:
            self.last_execution["mode"] = "process"
            ctx = mp.get_context("spawn")
            receiver, sender = ctx.Pipe(duplex=False)
            process = ctx.Process(target=_worker, args=(sender, info, request), daemon=True)
            received = False
            try:
                process.start()
                self.last_execution["worker_pid"] = process.pid
                sender.close()
                if not receiver.poll(budget):
                    return fallback("RERANKER_TIMEOUT", "Execution deadline exceeded")
                packet = json.loads(receiver.recv_bytes(maxlength=131072))
                received = True
                self.last_execution["worker_metrics"] = packet.get("worker_metrics", {})
                if packet.get("status") == "invalid":
                    return fallback("RERANKER_INVALID_OUTPUT", "Invalid or oversized output")
                if packet.get("status") != "ok":
                    return fallback("RERANKER_EXECUTION_ERROR", "Plugin failed to load or execute")
                raw = [RankedCandidate(**r) for r in packet["rows"]]
            except (RuntimeError, EOFError, OSError, ValueError, TypeError):
                return fallback("RERANKER_START_ERROR", "Worker could not start or exited")
            finally:
                if process.pid is not None:
                    if received:
                        process.join(timeout=0.5)
                    if process.is_alive():
                        process.terminate()
                    process.join(timeout=0.5)
                    if process.is_alive():
                        process.kill()
                        process.join(timeout=0.5)
                receiver.close()
                sender.close()
                self.last_execution["worker_alive"] = process.is_alive() if process.pid else False
                self.last_execution["seconds"] = time.monotonic() - started
        try:
            ordered = _validate(raw, ids)
        except (ValueError, TypeError, AttributeError):
            return fallback("RERANKER_INVALID_OUTPUT", "Result violates v1 contract")
        originals = dict(zip(ids, baseline))
        self.last_execution["seconds"] = time.monotonic() - started
        return [
            {
                **originals[r.candidate_id],
                "rerank_score": float(r.score),
                "rerank_rank": r.rank,
                "rerank_seconds": self.last_execution["seconds"],
            }
            for r in ordered
        ], []
