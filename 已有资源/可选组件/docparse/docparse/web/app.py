"""FastAPI application for the local web UI."""

from __future__ import annotations

import importlib.util
import io
import json
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

from .. import __version__, backend
from ..config import BASE_URL
from .jobs import FORMATS, HOME, JobManager, sanitize_name

STATIC = Path(__file__).parent / "static"
MAX_UPLOAD_MB = int(os.getenv("DOCPARSE_MAX_UPLOAD_MB", "200"))
MAX_UPLOAD = MAX_UPLOAD_MB * 1024 * 1024

app = FastAPI(title="docparse", version=__version__)
manager = JobManager()
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


def _layout_installed() -> bool:
    # find_spec avoids importing PaddleX on every health check.
    return importlib.util.find_spec("paddlex") is not None


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC / "index.html").read_text(encoding="utf-8")


@app.get("/api/settings")
def settings() -> dict:
    return {
        "formats": list(FORMATS),
        "defaults": {"mode": "auto", "dpi": 200, "formats": ["md", "json"]},
        "max_upload_mb": MAX_UPLOAD_MB,
    }


@app.get("/api/health")
def health() -> dict:
    reachable = backend.ping()
    models: list[str] = []
    if reachable:
        try:
            with urllib.request.urlopen(BASE_URL + "/models", timeout=3) as resp:
                models = [m.get("id") for m in json.loads(resp.read()).get("data", [])]
        except Exception:
            pass
    usage = shutil.disk_usage(HOME)
    return {
        "version": __version__,
        "ollama": reachable,
        "models": models,
        "layout": _layout_installed(),
        "disk_free_gb": round(usage.free / 1e9, 1),
        "jobs_dir": str(HOME / "jobs"),
    }


@app.post("/api/jobs")
async def create_job(files: list[UploadFile] = File(...), options: str = Form("{}")) -> dict:
    try:
        raw = json.loads(options or "{}")
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"invalid options JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise HTTPException(status_code=400, detail="options must be a JSON object")

    formats = [fmt for fmt in raw.get("formats", ["md", "json"]) if fmt in FORMATS]
    raw["formats"] = formats or ["md", "json"]

    payload: list[tuple[str, bytes]] = []
    for upload in files:
        data = await upload.read()
        if not data:
            continue
        if len(data) > MAX_UPLOAD:
            raise HTTPException(
                status_code=413,
                detail=f"{upload.filename or 'file'} exceeds {MAX_UPLOAD_MB} MB")
        payload.append((upload.filename or "file", data))
    if not payload:
        raise HTTPException(status_code=400, detail="no files uploaded")

    return manager.submit(payload, raw).to_dict()


@app.get("/api/jobs")
def list_jobs(limit: int = 100) -> dict:
    return {"jobs": manager.list(limit=limit)}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job.to_dict()


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict:
    if not manager.cancel(job_id):
        raise HTTPException(status_code=404, detail="job not found")
    return {"ok": True}


@app.post("/api/jobs/{job_id}/retry")
def retry_job(job_id: str) -> dict:
    if manager.get(job_id) is None:
        raise HTTPException(status_code=404, detail="job not found")
    if not manager.retry(job_id):
        raise HTTPException(status_code=409, detail="nothing to retry")
    return {"ok": True}


@app.get("/api/jobs/{job_id}/files/{name}")
def download_file(job_id: str, name: str) -> FileResponse:
    if manager.get(job_id) is None:
        raise HTTPException(status_code=404, detail="job not found")
    outdir = manager.output_dir(job_id).resolve()
    path = (outdir / sanitize_name(name)).resolve()
    if outdir not in path.parents or not path.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(path, filename=path.name)


@app.get("/api/jobs/{job_id}/bundle.zip")
def download_bundle(job_id: str) -> Response:
    if manager.get(job_id) is None:
        raise HTTPException(status_code=404, detail="job not found")
    outdir = manager.output_dir(job_id)
    if not outdir.is_dir():
        raise HTTPException(status_code=404, detail="no output")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(outdir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(outdir))
    if buffer.tell() == 0:
        raise HTTPException(status_code=404, detail="no output files yet")
    return Response(
        buffer.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{job_id}.zip"'},
    )
