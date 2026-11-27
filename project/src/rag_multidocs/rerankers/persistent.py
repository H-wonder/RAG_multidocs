"""One explicitly approved reusable worker, not a pool or plugin hot loader."""
import json
import multiprocessing as mp
import resource
import threading
import time
import uuid
from dataclasses import asdict

from ..contracts import RankedCandidate, RerankRequest, SearchHit
from ..extensions import load_entry


def _serve(connection, info):
    from .registry import _validate

    try:
        rank = load_entry(info)
        while True:
            packet = json.loads(connection.recv_bytes(maxlength=1048576))
            if packet.get("close"):
                break
            started = time.monotonic()
            key = packet["request_id"]
            try:
                request = RerankRequest(packet["query"], tuple(SearchHit(**h) for h in packet["candidates"]), packet["config"])
                rows = _validate(rank(request), [h.candidate_id for h in request.candidates])
                diagnostics = getattr(rank, "diagnostics", None)
                reply = {"status": "ok", "rows": [asdict(r) for r in rows], "diagnostics": diagnostics() if callable(diagnostics) else {}}
            except (ValueError, TypeError, AttributeError):
                reply = {"status": "invalid"}
            except Exception:
                reply = {"status": "error"}
            reply.update(request_id=key, seconds=time.monotonic()-started, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            encoded = json.dumps(reply).encode()
            if len(encoded) > 131072:
                encoded = json.dumps({"status": "invalid", "request_id": key}).encode()
            connection.send_bytes(encoded)
    except (EOFError, OSError, ValueError, KeyError):
        pass
    finally:
        connection.close()


class PersistentWorker:
    """Serial requests; a failed request falls back, restart only on a later call.

    At most initial spawn + max_restarts over this object's lifetime. Startup,
    dispatch and response all share one deadline. Timeout kills the process and
    reaps it before returning; no model inference is left on a Python thread.
    """

    def __init__(self, max_restarts=1):
        if not 0 <= max_restarts <= 5:
            raise ValueError("max_restarts must be in 0..5")
        self.max_starts = max_restarts + 1
        self.starts = 0
        self.process = self.connection = self.identity = None
        self.closed = False
        self.lock = threading.Lock()
        self.last = {}

    def _stop(self, graceful=False):
        if self.process:
            if self.process.pid is not None:
                if graceful and self.process.is_alive() and self.connection:
                    try:
                        self.connection.send_bytes(b'{"close":true}')
                        self.process.join(timeout=1)
                    except (OSError, EOFError):
                        pass
                if self.process.is_alive():
                    self.process.terminate()
                self.process.join(timeout=0.5)
                if self.process.is_alive():
                    self.process.kill()
                    self.process.join(timeout=0.5)
        if self.connection:
            self.connection.close()
        self.process = self.connection = None

    def execute(self, info, request, timeout):
        started = time.monotonic()
        if not self.lock.acquire(timeout=max(0, timeout)):
            raise TimeoutError("PERSISTENT_BUSY")
        self.last = {"mode": "persistent", "request_id": uuid.uuid4().hex, "worker_alive": False}
        try:
            if self.closed:
                raise RuntimeError("PERSISTENT_CLOSED")
            identity = (str(info.path), json.dumps(request.config, sort_keys=True))
            if self.process and (not self.process.is_alive() or self.identity != identity):
                self._stop()
            if self.process is None:
                if self.starts >= self.max_starts:
                    raise RuntimeError("PERSISTENT_RESTART_LIMIT")
                parent, child = mp.get_context("spawn").Pipe()
                process = mp.get_context("spawn").Process(target=_serve, args=(child, info), daemon=True)
                self.process, self.connection = process, parent
                try:
                    process.start()
                finally:
                    child.close()
                self.identity = identity
                self.starts += 1
            self.last.update(worker_pid=self.process.pid, starts=self.starts)
            packet = json.dumps({**asdict(request), "request_id": self.last["request_id"]}).encode()
            if len(packet) > 1048576:
                raise ValueError("PERSISTENT_INPUT_LIMIT")
            errors = []
            connection = self.connection
            def send():
                try:
                    connection.send_bytes(packet)
                except (OSError, EOFError) as exc:
                    errors.append(exc)
            sender = threading.Thread(target=send, daemon=True)
            sender.start()
            sender.join(timeout=max(0, timeout-(time.monotonic()-started)))
            if sender.is_alive():
                self._stop()
                sender.join(timeout=0.5)
                raise TimeoutError("PERSISTENT_DISPATCH_TIMEOUT")
            if errors:
                raise OSError("PERSISTENT_DISPATCH_ERROR")
            left = timeout-(time.monotonic()-started)
            if left <= 0 or not self.connection.poll(left):
                raise TimeoutError("PERSISTENT_TIMEOUT")
            reply = json.loads(self.connection.recv_bytes(maxlength=131072))
            if reply.get("request_id") != self.last["request_id"]:
                raise ValueError("PERSISTENT_REQUEST_ID_MISMATCH")
            self.last.update(reply)
            if reply.get("status") != "ok":
                raise ValueError("PERSISTENT_INVALID_OUTPUT" if reply.get("status") == "invalid" else "PERSISTENT_EXECUTION_ERROR")
            self.last["worker_alive"] = self.process.is_alive()
            return [RankedCandidate(**r) for r in reply["rows"]]
        except Exception:
            self._stop()
            self.last["worker_alive"] = False
            raise
        finally:
            self.last["seconds_total"] = time.monotonic()-started
            self.lock.release()

    def close(self):
        with self.lock:
            self.closed = True
            self._stop(graceful=True)
