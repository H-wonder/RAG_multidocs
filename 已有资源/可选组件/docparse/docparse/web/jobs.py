"""Persistent, single-worker job queue.

Jobs live on disk under ~/.docparse/jobs/<id>/ so history survives a restart.
The worker is serial by design: Ollama runs one request at a time and the 8 GB
GPU should not hold two KV caches.
"""

from __future__ import annotations

import json
import os
import queue
import re
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ..config import ParseOptions
from ..pipeline import CancelledError, Parser
from .. import writers

HOME = Path(os.getenv("DOCPARSE_HOME", "~/.docparse")).expanduser()
JOBS_DIR = HOME / "jobs"
FORMATS = ("md", "json", "html", "txt")


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sanitize_name(name: str) -> str:
    """Keep a single path component; strip traversal and unsafe characters."""
    name = Path(name or "file").name
    name = re.sub(r"[^\w.\-一-鿿 ]+", "_", name, flags=re.UNICODE).strip()
    return name or "file"


@dataclass
class FileTask:
    name: str                      # original display name
    stored: str                    # sanitized name inside input/
    status: str = "queued"         # queued|running|done|failed|cancelled|interrupted
    pages_total: int | None = None
    pages_done: int = 0
    elapsed_ms: int = 0
    warnings: int = 0
    outputs: list[str] = field(default_factory=list)
    error: str = ""
    mode: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Job:
    id: str
    created_at: str
    status: str = "queued"         # queued|running|done|partial|failed|cancelled|interrupted
    options: dict = field(default_factory=dict)
    files: list[FileTask] = field(default_factory=list)
    cancel_requested: bool = False
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "created_at": self.created_at,
            "status": self.status,
            "options": self.options,
            "error": self.error,
            "files": [f.to_dict() for f in self.files],
        }


class JobManager:
    def __init__(self, jobs_dir: Path | str = JOBS_DIR, workers: int = 1):
        self.jobs_dir = Path(jobs_dir)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.jobs: dict[str, Job] = {}
        self.queue: queue.Queue[str | None] = queue.Queue()
        self._load()
        self._threads = [
            threading.Thread(target=self._worker, daemon=True, name=f"docparse-worker-{i}")
            for i in range(max(1, workers))
        ]
        for thread in self._threads:
            thread.start()

    # ── paths ───────────────────────────────────────────────────────────────
    def job_dir(self, job_id: str) -> Path:
        return self.jobs_dir / job_id

    def input_dir(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "input"

    def output_dir(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "output"

    # ── persistence ─────────────────────────────────────────────────────────
    def _load(self) -> None:
        for directory in sorted(self.jobs_dir.iterdir()):
            manifest = directory / "job.json"
            if not manifest.is_file():
                continue
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
                files = [FileTask(**f) for f in data.get("files", [])]
                job = Job(
                    id=data["id"],
                    created_at=data.get("created_at", ""),
                    status=data.get("status", "failed"),
                    options=data.get("options", {}),
                    files=files,
                    error=data.get("error", ""),
                )
                if job.status in {"queued", "running"}:
                    job.status = "interrupted"
                    job.error = "service restarted while this job was running"
                    for task in job.files:
                        if task.status in {"queued", "running"}:
                            task.status = "interrupted"
                self.jobs[job.id] = job
            except Exception:
                continue

    def _save(self, job: Job) -> None:
        with self.lock:
            directory = self.job_dir(job.id)
            directory.mkdir(parents=True, exist_ok=True)
            temp = directory / "job.json.tmp"
            temp.write_text(json.dumps(job.to_dict(), ensure_ascii=False, indent=2),
                            encoding="utf-8")
            os.replace(temp, directory / "job.json")

    # ── public API ──────────────────────────────────────────────────────────
    def submit(self, files: list[tuple[str, bytes]], options: dict) -> Job:
        job_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        job = Job(id=job_id, created_at=_now(), options=options)
        directory = self.job_dir(job_id)
        (directory / "input").mkdir(parents=True, exist_ok=True)
        (directory / "output").mkdir(parents=True, exist_ok=True)

        used: set[str] = set()
        for original, data in files:
            stored = sanitize_name(original)
            stem, suffix = os.path.splitext(stored)
            counter = 1
            while stored in used:
                stored = f"{stem}-{counter}{suffix}"
                counter += 1
            used.add(stored)
            (directory / "input" / stored).write_bytes(data)
            job.files.append(FileTask(name=original or stored, stored=stored))

        with self.lock:
            self.jobs[job.id] = job
        self._save(job)
        self.queue.put(job.id)
        return job

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def list(self, limit: int = 100) -> list[dict]:
        with self.lock:
            jobs = sorted(self.jobs.values(), key=lambda j: j.created_at, reverse=True)
            return [job.to_dict() for job in jobs[:limit]]

    def cancel(self, job_id: str) -> bool:
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                return False
            job.cancel_requested = True
            if job.status == "queued":
                job.status = "cancelled"
                for task in job.files:
                    if task.status == "queued":
                        task.status = "cancelled"
            self._save(job)
            return True

    def retry(self, job_id: str) -> bool:
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None or job.status in {"running", "queued"}:
                return False
            retryable = [t for t in job.files
                         if t.status in {"failed", "cancelled", "interrupted"}]
            if not retryable:
                return False
            for task in retryable:
                task.status = "queued"
                task.error = ""
                task.pages_done = 0
                task.pages_total = None
                task.warnings = 0
                task.outputs = []
            job.cancel_requested = False
            job.status = "queued"
            job.error = ""
            self._save(job)
        self.queue.put(job.id)
        return True

    # ── worker ──────────────────────────────────────────────────────────────
    def _worker(self) -> None:
        while True:
            job_id = self.queue.get()
            if job_id is None:
                return
            job = self.get(job_id)
            if job is None or job.status in {"cancelled", "done", "partial", "failed", "interrupted"}:
                continue
            try:
                self._run(job)
            except Exception as exc:  # never let the worker die
                with self.lock:
                    job.status = "failed"
                    job.error = str(exc)[:500]
                    self._save(job)

    def _run(self, job: Job) -> None:
        with self.lock:
            job.status = "running"
            job.error = ""
            self._save(job)

        opts = self._build_options(job.options)
        formats = [f for f in job.options.get("formats", ["md", "json"]) if f in FORMATS]
        formats = formats or ["md", "json"]
        failed = cancelled = 0

        for task in job.files:
            if task.status == "done":      # retry only re-runs failed/cancelled files
                continue
            if job.cancel_requested:
                with self.lock:
                    task.status = "cancelled"
                    self._save(job)
                cancelled += 1
                continue

            with self.lock:
                task.status = "running"
                task.pages_done = 0
                task.pages_total = None
                task.error = ""
                self._save(job)

            started = time.perf_counter()

            def progress(event: dict, task=task, job=job) -> None:
                with self.lock:
                    if event.get("event") == "page_start":
                        task.pages_total = event.get("total") or task.pages_total
                    elif event.get("event") == "page_done":
                        task.pages_done = event.get("page") or task.pages_done
                        task.pages_total = event.get("total") or task.pages_total
                        task.mode = event.get("mode") or task.mode
                    self._save(job)

            try:
                parser = Parser(opts)
                document = parser.parse(
                    self.input_dir(job.id) / task.stored,
                    progress=progress,
                    should_cancel=lambda job=job: job.cancel_requested,
                )
                paths = writers.write(document, self.output_dir(job.id), formats=formats)
                with self.lock:
                    task.outputs = [path.name for path in paths.values()]
                    task.pages_total = task.pages_total or len(document.pages)
                    task.pages_done = task.pages_total
                    task.elapsed_ms = int((time.perf_counter() - started) * 1000)
                    task.warnings = sum(
                        1 for w in document.warnings
                        if w.get("severity") in {"high", "medium"})
                    task.status = "done"
                    self._save(job)
            except CancelledError:
                with self.lock:
                    task.status = "cancelled"
                    task.elapsed_ms = int((time.perf_counter() - started) * 1000)
                    self._save(job)
                cancelled += 1
            except Exception as exc:
                with self.lock:
                    task.status = "failed"
                    task.error = str(exc)[:500]
                    task.elapsed_ms = int((time.perf_counter() - started) * 1000)
                    self._save(job)
                failed += 1

        with self.lock:
            if job.cancel_requested and cancelled:
                job.status = "cancelled"
            elif failed and (failed + cancelled) == len(job.files):
                job.status = "failed"
            elif failed or cancelled:
                job.status = "partial"
            else:
                job.status = "done"
            self._save(job)

    @staticmethod
    def _build_options(raw: dict) -> ParseOptions:
        raw = raw or {}
        fields = raw.get("fields") or []
        if isinstance(fields, str):
            fields = [f.strip() for f in fields.split(",") if f.strip()]
        max_pages = raw.get("max_pages")
        return ParseOptions(
            mode=raw.get("mode", "auto"),
            dpi=int(raw.get("dpi") or 200),
            max_pages=int(max_pages) if max_pages else None,
            kie_fields=list(fields) or None,
            layout_device=raw.get("layout_device")
            or os.getenv("DOCPARSE_LAYOUT_DEVICE", "cpu"),
        )
