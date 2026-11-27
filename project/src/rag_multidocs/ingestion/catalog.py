"""Single-machine publication: one writer; pinned reader snapshots block GC.

Index staging precedes SQLite activation. Interrupted staging is reported and
marked failed, never auto-activated. This is NOT a distributed transaction or a
claim of complete physical power-loss safety. Existing catalog formats require
an explicitly chosen new location; no user collection is migrated/deleted.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from types import MappingProxyType

from ..index.contracts import supports_versions


def model_identity(embedder):
    if callable(getattr(embedder, "identity", None)):
        return embedder.identity()
    return {
        "provider": type(embedder).__name__,
        "model": getattr(embedder, "model_name", getattr(embedder, "model", "hashing")),
        "dimension": getattr(embedder, "dim", 0),
        "query_instruction": getattr(embedder, "query_instruction", ""),
        "base_url": getattr(embedder, "base_url", ""),
        "revision": getattr(embedder, "revision", None),
        "normalize_embeddings": getattr(embedder, "normalize_embeddings", True),
    }


def index_identity(index):
    if callable(getattr(index, "identity", None)):
        return index.identity()
    return {
        "backend": getattr(index, "name", type(index).__name__),
        "embedding": model_identity(index.embedder) if hasattr(index, "embedder") else None,
        "path": str(Path(index.path).resolve()) if getattr(index, "path", None) else None,
    }


class KnowledgeBusyError(RuntimeError):
    pass


class DocumentCatalog:
    FORMAT = "rag-catalog-v2"

    def __init__(self, index, path=None):
        if not supports_versions(index):
            raise ValueError("INDEX_CAPABILITY_UNAVAILABLE: version publication")
        self.index = index
        if path is None and getattr(index, "path", None):
            path = Path(index.path).with_suffix(".catalog.sqlite")
        if path is None and getattr(index, "name", "") == "milvus":
            raise ValueError(
                "CATALOG_PATH_REQUIRED: supply an explicit data directory/catalog path"
            )
        self.path = Path(path).resolve() if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self._local_writer = threading.Lock()
        self._snapshot = contextvars.ContextVar(f"catalog_snapshot_{id(self)}", default=None)
        self._readers = 0
        self.recovery_warnings = []
        self.cleanup_warnings = []
        self.db = sqlite3.connect(
            str(self.path) if self.path else ":memory:", timeout=10, check_same_thread=False
        )
        self.db.row_factory = sqlite3.Row
        identity = json.dumps(index_identity(index), sort_keys=True)
        try:
            with self.writer():
                tables = {
                    r[0]
                    for r in self.db.execute("SELECT name FROM sqlite_master WHERE type='table'")
                }
                if "settings" in tables:
                    old = dict(self.db.execute("SELECT key,value FROM settings"))
                    if old.get("catalog_format") != self.FORMAT:
                        raise ValueError(
                            "CATALOG_FORMAT_MISMATCH: use new catalog; migration requires approval"
                        )
                    if old.get("index_identity") != identity:
                        raise ValueError(
                            "INDEX_MODEL_MISMATCH: catalog belongs to another index space"
                        )
                self.db.executescript("""PRAGMA journal_mode=WAL;
                PRAGMA synchronous=FULL;
                CREATE TABLE IF NOT EXISTS documents(doc_id TEXT PRIMARY KEY,source TEXT NOT NULL,sha256 TEXT NOT NULL,fingerprint TEXT NOT NULL,active_version INTEGER NOT NULL,deleted INTEGER NOT NULL DEFAULT 0,payload TEXT NOT NULL DEFAULT '{}');
                CREATE TABLE IF NOT EXISTS versions(doc_id TEXT,version INTEGER,status TEXT,sha256 TEXT,fingerprint TEXT,error TEXT DEFAULT '',PRIMARY KEY(doc_id,version));
                CREATE TABLE IF NOT EXISTS chunks(id TEXT PRIMARY KEY,doc_id TEXT,version INTEGER,position INTEGER,payload TEXT);
                CREATE INDEX IF NOT EXISTS chunk_position ON chunks(doc_id,version,position);
                CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY,value TEXT);
                CREATE TABLE IF NOT EXISTS ingest_cache(key TEXT PRIMARY KEY,signature TEXT,payload TEXT);
                """)
                self.db.executemany(
                    "INSERT OR IGNORE INTO settings VALUES (?,?)",
                    [("index_identity", identity), ("catalog_format", self.FORMAT)],
                )
                self.db.commit()
                self._recover_locked()
        except BaseException:
            self.db.close()
            raise
        index.version_catalog = self

    @contextmanager
    def _file_lock(self, suffix, *, shared=False):
        if self.path is None:
            yield
            return
        import fcntl

        handle = self.path.with_suffix(self.path.suffix + suffix).open("a+b")
        try:
            try:
                fcntl.flock(handle, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
            except BlockingIOError:
                raise KnowledgeBusyError(
                    "KNOWLEDGE_BUSY: writer or pinned readers active"
                ) from None
            yield
        finally:
            handle.close()

    @contextmanager
    def writer(self):
        if not self._local_writer.acquire(blocking=False):
            raise KnowledgeBusyError("KNOWLEDGE_BUSY: one writer per knowledge base")
        try:
            with self._file_lock(".writer.lock"), self.lock:
                yield
        finally:
            self._local_writer.release()

    def _recover_locked(self):
        rows = self.db.execute(
            "SELECT doc_id,version FROM versions WHERE status='staging'"
        ).fetchall()
        for r in rows:
            self.recovery_warnings.append(
                {
                    "code": "STAGING_INTERRUPTED",
                    "severity": "warning",
                    "message": "Unknown completion; not activated",
                    **dict(r),
                }
            )
        self.db.execute(
            "UPDATE versions SET status='failed',error='interrupted; requires reconciliation' WHERE status='staging'"
        )
        self.db.commit()
        return self.recovery_warnings

    def reconcile(self):
        """Read index counts for interrupted/failed versions, without activating them."""
        with self.writer():
            self._recover_locked()
            rows = self.db.execute(
                "SELECT doc_id,version,status FROM versions WHERE status='failed'"
            ).fetchall()
            result = []
            for r in rows:
                data = dict(r)
                try:
                    data["index_rows"] = self.index.count_version(r["doc_id"], r["version"])
                except Exception as exc:
                    data["index_rows"] = None
                    data["error"] = type(exc).__name__
                result.append(data)
            return result

    def active(self):
        pinned = self._snapshot.get()
        if pinned is not None:
            return dict(pinned)
        with self.lock:
            rows = self.db.execute("SELECT doc_id,active_version,deleted FROM documents").fetchall()
            result = {r["doc_id"]: r["active_version"] for r in rows if not r["deleted"]}
            # Legacy unversioned keyword rows are a read-only generation 0. They
            # remain visible until this logical document is published or deleted.
            known = {r["doc_id"] for r in rows}
            for c in getattr(self.index, "chunks", {}).values():
                if c.get("version") is None and c.get("doc_id") not in known:
                    result[c.get("doc_id")] = 0
            return result

    def source_records(self):
        """Active identities from the existing catalog, under the pinned snapshot."""
        active = self.active()
        with self.lock:
            rows = self.db.execute(
                "SELECT doc_id,source,active_version,payload FROM documents WHERE deleted=0"
            ).fetchall()
            result = [dict(title=json.loads(r['payload']).get('title', ''), doc_id=r['doc_id'], source=r['source'],
                           version=r['active_version']) for r in rows
                      if active.get(r['doc_id']) == r['active_version']]
            known = {r['doc_id'] for r in result}
            for c in getattr(self.index, 'chunks', {}).values():
                did = c.get('doc_id')
                if did not in known and active.get(did) == 0:
                    result.append({'doc_id': did, 'source': c.get('source', ''), 'version': 0})
                    known.add(did)
            return result

    @contextmanager
    def read_snapshot(self):
        pinned = self._snapshot.get()
        if pinned is not None:
            yield pinned
            return
        with self._file_lock(".readers.lock", shared=True):
            with self.lock:
                active = MappingProxyType(self.active())
                self._readers += 1
            token = self._snapshot.set(active)
            try:
                yield active
            finally:
                self._snapshot.reset(token)
                with self.lock:
                    self._readers -= 1

    def filter_expr(self, snapshot=None):
        values = self.active() if snapshot is None else snapshot
        return (
            " or ".join(
                f"(doc_id == {json.dumps(d)} and version == {int(v)})" for d, v in values.items()
            )
            or 'doc_id == "__unpublished__"'
        )

    def publish(self, document, chunks, fingerprint):
        did = document.doc_id
        sha = (
            document.metadata.get("raw_sha256")
            or document.metadata.get("sha256")
            or hashlib.sha256(document.content.encode()).hexdigest()
        )
        with self.writer():
            version = None
            try:
                self.db.execute("BEGIN IMMEDIATE")
                old = self.db.execute("SELECT * FROM documents WHERE doc_id=?", (did,)).fetchone()
                if (
                    old
                    and not old["deleted"]
                    and old["sha256"] == sha
                    and old["fingerprint"] == fingerprint
                ):
                    expected = self.db.execute(
                        "SELECT COUNT(*) FROM chunks WHERE doc_id=? AND version=?",
                        (did, old["active_version"]),
                    ).fetchone()[0]
                    if (
                        expected > 0
                        and self.index.count_version(did, old["active_version"]) == expected
                    ):
                        self.db.rollback()
                        return 0, old["active_version"]
                version = self.db.execute(
                    "SELECT COALESCE(MAX(version),0)+1 FROM versions WHERE doc_id=?", (did,)
                ).fetchone()[0]
                self.db.execute(
                    "INSERT INTO versions(doc_id,version,status,sha256,fingerprint) VALUES (?,?,?,?,?)",
                    (did, version, "staging", sha, fingerprint),
                )
                self.db.commit()
                staged = []
                for c in chunks:
                    data = c.as_dict()
                    data.update(
                        id=hashlib.sha256(f"{did}:{version}:{c.id}".encode()).hexdigest(),
                        version=version,
                    )
                    staged.append(data)
                if not staged:
                    raise ValueError("EMPTY_PUBLICATION")
                self.index.stage_version(document, staged, version)
                if self.index.count_version(did, version) != len(staged):
                    raise RuntimeError("INDEX_INCOMPLETE: row count mismatch")
                verifier = getattr(self.index, "verify_version", None)
                if verifier and not verifier(did, version, staged):
                    raise RuntimeError("INDEX_INCOMPLETE: persisted rows mismatch")
                self.db.execute("BEGIN IMMEDIATE")
                self.db.executemany(
                    "INSERT INTO chunks VALUES (?,?,?,?,?)",
                    [
                        (c["id"], did, version, c["index"], json.dumps(c, ensure_ascii=False))
                        for c in staged
                    ],
                )
                self.db.execute(
                    """INSERT INTO documents(doc_id,source,sha256,fingerprint,active_version,deleted,payload) VALUES (?,?,?,?,?,0,?)
                ON CONFLICT(doc_id) DO UPDATE SET source=excluded.source,sha256=excluded.sha256,fingerprint=excluded.fingerprint,active_version=excluded.active_version,deleted=0,payload=excluded.payload""",
                    (
                        did,
                        document.source,
                        sha,
                        fingerprint,
                        version,
                        json.dumps(document.to_dict(), ensure_ascii=False),
                    ),
                )
                self.db.execute(
                    "UPDATE versions SET status='superseded' WHERE doc_id=? AND status='active'",
                    (did,),
                )
                self.db.execute(
                    "UPDATE versions SET status='active' WHERE doc_id=? AND version=?",
                    (did, version),
                )
                self.db.commit()
                return len(staged), version
            except Exception as exc:
                self.db.rollback()
                if version is not None:
                    try:
                        self.db.execute(
                            "UPDATE versions SET status='failed',error=? WHERE doc_id=? AND version=?",
                            (type(exc).__name__, did, version),
                        )
                        self.db.commit()
                    except sqlite3.Error:
                        self.db.rollback()  # leave staging for conservative restart recovery
                raise

    def attach_metadata(self, hits, snapshot=None):
        """Read existing versioned payloads; no Milvus schema migration required."""
        active = self.active() if snapshot is None else snapshot
        result = []
        with self.lock:
            for hit in hits:
                row = self.db.execute("SELECT doc_id,version,payload FROM chunks WHERE id=?", (hit["chunk_id"],)).fetchone()
                data = dict(hit)
                if row and row["doc_id"] == hit.get("doc_id") and row["version"] == hit.get("version") and active.get(row["doc_id"]) == row["version"]:
                    payload = json.loads(row["payload"])
                    if payload["text"] != hit["text"] or payload["source"] != hit.get("source"):
                        raise RuntimeError("INDEX_PAYLOAD_MISMATCH")
                    meta = payload.get("metadata", {})
                    data["metadata"] = meta
                    for key in ["location", "table_data", "table_headers", "semantic_cells", "retrieval_text", "source_table_id"]:
                        if key in meta:
                            data[key] = meta[key]
                result.append(data)
        return result

    def neighbors(self, ids, snapshot=None):
        active = self.active() if snapshot is None else snapshot
        result = {}
        with self.lock:
            for key in ids:
                row = self.db.execute("SELECT * FROM chunks WHERE id=?", (key,)).fetchone()
                if row is None or active.get(row["doc_id"]) != row["version"]:
                    continue
                rows = self.db.execute(
                    "SELECT payload FROM chunks WHERE doc_id=? AND version=? AND position BETWEEN ? AND ?",
                    (row["doc_id"], row["version"], row["position"] - 1, row["position"] + 1),
                )
                for hit in rows:
                    data = json.loads(hit[0])
                    data["chunk_id"] = data["id"]
                    result[data["id"]] = data
        return list(result.values())

    def delete(self, doc_id):
        with self.writer(), self.db:
            old = self.db.execute(
                "SELECT doc_id FROM documents WHERE doc_id=?", (doc_id,)
            ).fetchone()
            if old is None:
                self.db.execute(
                    "INSERT INTO documents(doc_id,source,sha256,fingerprint,active_version,deleted) VALUES (?,'','','',0,1)",
                    (doc_id,),
                )
                self.db.execute(
                    "INSERT OR IGNORE INTO versions(doc_id,version,status,sha256,fingerprint) VALUES (?,0,'superseded','','')",
                    (doc_id,),
                )
            else:
                self.db.execute("UPDATE documents SET deleted=1 WHERE doc_id=?", (doc_id,))
        return {"doc_id": doc_id, "status": "deleted"}

    def cleanup(self, doc_id=None):
        self.cleanup_warnings = []
        try:
            with self.writer(), self._file_lock(".readers.lock"):
                if self._readers:
                    raise KnowledgeBusyError("read snapshot pinned")
                rows = self.db.execute("""SELECT v.doc_id,v.version FROM versions v LEFT JOIN documents d ON d.doc_id=v.doc_id
                WHERE v.status != 'cleaned' AND (v.status IN ('failed','superseded') OR d.deleted=1)""").fetchall()
                removed = 0
                for row in rows:
                    if doc_id and row["doc_id"] != doc_id:
                        continue
                    self.index.remove_version(row["doc_id"], row["version"])
                    if self.index.count_version(row["doc_id"], row["version"]) != 0:
                        raise RuntimeError("CLEANUP_INCOMPLETE")
                    with self.db:
                        self.db.execute(
                            "DELETE FROM chunks WHERE doc_id=? AND version=?", tuple(row)
                        )
                        self.db.execute(
                            "UPDATE versions SET status='cleaned' WHERE doc_id=? AND version=?",
                            tuple(row),
                        )
                    removed += 1
                return removed
        except KnowledgeBusyError:
            self.cleanup_warnings.append(
                {
                    "code": "CLEANUP_PINNED",
                    "message": "Retry after active requests/writer finish",
                    "severity": "warning",
                }
            )
            return 0

    def cached_ingest(self, key, signature):
        with self.lock:
            row = self.db.execute(
                "SELECT payload FROM ingest_cache WHERE key=? AND signature=?", (key, signature)
            ).fetchone()
            if not row:
                return None
            records = json.loads(row[0])
            active = self.active()
            for d in records:
                if (
                    active.get(d["doc_id"]) != d["version"]
                    or self.index.count_version(d["doc_id"], d["version"]) != d["chunks"]
                ):
                    return None
            return [{**d, "status": "unchanged"} for d in records]

    def remember_ingest(self, key, signature, records):
        with self.writer(), self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO ingest_cache VALUES (?,?,?)",
                (key, signature, json.dumps(records, ensure_ascii=False)),
            )

    def near_duplicates(self, document, max_distance=10):
        from .dedup import hamming, simhash64

        from ..model import Document
        from .structured import public_blocks

        def text_of(doc):
            return doc.content or "\n".join(
                str(b.get("content", b.get("text", b.get("caption", ""))))
                for b in public_blocks(doc)
            )

        text = text_of(document)
        if not text.strip():
            return []
        fingerprint = simhash64(text)
        with self.lock:
            rows = self.db.execute(
                "SELECT doc_id,source,payload FROM documents WHERE deleted=0 AND doc_id<>?",
                (document.doc_id,),
            ).fetchall()
        reports = []
        for row in rows:
            other = json.loads(row["payload"])
            other_text = text_of(Document.from_dict(other))
            if not other_text.strip():
                continue
            distance = hamming(fingerprint, simhash64(other_text))
            if distance <= max_distance:
                reports.append(
                    {"doc_id": row["doc_id"], "source": row["source"], "hamming_distance": distance}
                )
        return reports

    def close(self):
        self.db.close()
