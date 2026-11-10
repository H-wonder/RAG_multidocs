"""文档清单（SQLite）：记录文档身份、版本、状态与事件。

来源：``任务二/docflow/manifest.py``。把原来的模块级全局配置改成实例化
``ManifestStore(db_path)``，方便测试和在同一进程里管理多个知识库。

可直接用于课程项目的“增量入库”：
- doc_id 由逻辑相对路径派生，稳定；文件内容变化只升 version；
- sha256 是幂等基础：内容没变就不重跑；
- 状态机 NEW → PARSED → INDEXED（另有 NEEDS_REVIEW / DUPLICATE / FAILED / DELETED）；
- 每次状态变化写一条 events，排障和演示都能查到。
"""

from __future__ import annotations

import hashlib
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

SCHEMA = """
CREATE TABLE IF NOT EXISTS docs (
    doc_id      TEXT PRIMARY KEY,
    rel_path    TEXT NOT NULL,
    abs_path    TEXT NOT NULL,
    sha256      TEXT NOT NULL,
    size        INTEGER NOT NULL,
    mtime       REAL NOT NULL,
    status      TEXT NOT NULL,
    version     INTEGER NOT NULL DEFAULT 1,
    pages       INTEGER DEFAULT 0,
    needs_review INTEGER DEFAULT 0,
    warnings    TEXT DEFAULT '',
    last_error  TEXT DEFAULT '',
    simhash     TEXT DEFAULT '',
    meta_path   TEXT DEFAULT '',
    summary_path TEXT DEFAULT '',
    updated_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_docs_status ON docs(status);

CREATE TABLE IF NOT EXISTS events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id     TEXT NOT NULL,
    event      TEXT NOT NULL,
    detail     TEXT DEFAULT '',
    created_at TEXT NOT NULL
);
"""

STATUSES = {
    "NEW",
    "PARSED",
    "NEEDS_REVIEW",
    "DUPLICATE",
    "INDEXED",
    "FAILED",
    "DELETED",
}

_ALLOWED_FIELDS = {
    "pages",
    "needs_review",
    "warnings",
    "last_error",
    "simhash",
    "meta_path",
    "summary_path",
}


def doc_id_for(rel_path: str, length: int = 12) -> str:
    return hashlib.sha256(rel_path.replace("\\", "/").encode("utf-8")).hexdigest()[:length]


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ManifestStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=30000")
            yield conn
            conn.commit()
        finally:
            conn.close()

    def init_db(self) -> None:
        with self.connect() as conn:
            conn.executescript(SCHEMA)

    def log_event(self, conn: sqlite3.Connection, doc_id: str, event: str, detail: str = "") -> None:
        conn.execute(
            "INSERT INTO events(doc_id, event, detail, created_at) VALUES (?,?,?,?)",
            (doc_id, event, detail, _now()),
        )

    def scan(self, files: Iterable[str | Path], root: str | Path) -> dict:
        """比对文件集合与清单，返回新增/变更/删除三类。

        ``root`` 用来计算稳定的相对路径；文件集合应由 Connector 或目录扫描给出。
        只做清单比对，不解析文件。
        """
        root = Path(root)
        seen: set[str] = set()
        added: list[dict] = []
        changed: list[dict] = []

        paths = sorted(Path(p) for p in files if Path(p).is_file() and not Path(p).name.startswith("."))
        with self.connect() as conn:
            for path in paths:
                try:
                    rel = path.resolve().relative_to(root.resolve()).as_posix()
                except ValueError:
                    rel = path.name
                sha = sha256_file(path)
                did = doc_id_for(rel)
                seen.add(did)
                row = conn.execute("SELECT * FROM docs WHERE doc_id=?", (did,)).fetchone()

                if row is None:
                    stat = path.stat()
                    conn.execute(
                        """INSERT INTO docs(doc_id, rel_path, abs_path, sha256, size, mtime,
                                            status, version, updated_at)
                           VALUES (?,?,?,?,?,?,?,?,?)""",
                        (did, rel, str(path), sha, stat.st_size, stat.st_mtime,
                         "NEW", 1, _now()),
                    )
                    self.log_event(conn, did, "discovered", rel)
                    added.append({"doc_id": did, "rel_path": rel, "version": 1})
                elif row["sha256"] != sha:
                    stat = path.stat()
                    version = int(row["version"]) + 1
                    conn.execute(
                        """UPDATE docs SET sha256=?, size=?, mtime=?, status='NEW',
                                           version=?, updated_at=?, last_error=''
                           WHERE doc_id=?""",
                        (sha, stat.st_size, stat.st_mtime, version, _now(), did),
                    )
                    self.log_event(conn, did, "changed", f"{rel} -> v{version}")
                    changed.append({"doc_id": did, "rel_path": rel, "version": version})

            existing = conn.execute(
                "SELECT doc_id, rel_path FROM docs WHERE status != 'DELETED'"
            ).fetchall()
            removed = [
                {"doc_id": row["doc_id"], "rel_path": row["rel_path"]}
                for row in existing
                if row["doc_id"] not in seen
            ]
            for item in removed:
                conn.execute(
                    "UPDATE docs SET status='DELETED', updated_at=? WHERE doc_id=?",
                    (_now(), item["doc_id"]),
                )
                self.log_event(conn, item["doc_id"], "deleted", item["rel_path"])

        return {"added": added, "changed": changed, "removed": removed}

    def mark(self, doc_id: str, status: str, **fields) -> None:
        if status not in STATUSES:
            raise ValueError(f"未知状态: {status}，可选 {sorted(STATUSES)}")
        sets, values = ["status=?", "updated_at=?"], [status, _now()]
        for key, value in fields.items():
            if key in _ALLOWED_FIELDS:
                sets.append(f"{key}=?")
                values.append(value)
        values.append(doc_id)
        with self.connect() as conn:
            conn.execute(f"UPDATE docs SET {', '.join(sets)} WHERE doc_id=?", values)
            self.log_event(conn, doc_id, f"status:{status}", str(fields)[:500])

    def get(self, doc_id: str) -> dict | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM docs WHERE doc_id=?", (doc_id,)).fetchone()
        return dict(row) if row else None

    def all_docs(self) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM docs").fetchall()
        return [dict(row) for row in rows]

    def pending(self, limit: int = 50) -> list[dict]:
        """待处理：新建或解析过但还没入库的。"""
        with self.connect() as conn:
            rows = conn.execute(
                """SELECT * FROM docs
                   WHERE status IN ('NEW','PARSED','NEEDS_REVIEW')
                   ORDER BY updated_at LIMIT ?""",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def indexed_fingerprints(self, exclude: str | None = None) -> list[tuple[str, str]]:
        """给去重用的基准：只取已成功入库且指纹非空的文档。"""
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT doc_id, simhash FROM docs WHERE status='INDEXED' AND simhash != ''"
            ).fetchall()
        return [(row["doc_id"], row["simhash"]) for row in rows if row["doc_id"] != exclude]

    def events(self, doc_id: str, limit: int = 50) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM events WHERE doc_id=? ORDER BY id DESC LIMIT ?",
                (doc_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]


__all__ = ["ManifestStore", "STATUSES", "doc_id_for", "sha256_file"]
