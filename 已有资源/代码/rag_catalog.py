"""新文档增量入库：SQLite激活版本，Milvus保留历史版本以避免失败更新覆盖旧数据。"""

from __future__ import annotations

import fcntl
import hashlib
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from chunking import ChunkingOptions, chunk_docparse_ir, chunk_markdown, chunk_text
from docparse_connector import DocparseConnector, DocparseOptions
from document_model import Document, sha256_file, stable_doc_id

SCHEMA = """
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS documents (
 doc_id TEXT PRIMARY KEY, source TEXT UNIQUE NOT NULL, sha256 TEXT NOT NULL,
 active_version INTEGER NOT NULL, chunks INTEGER NOT NULL, pages INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS versions (
 doc_id TEXT NOT NULL, version INTEGER NOT NULL, sha256 TEXT NOT NULL,
 status TEXT NOT NULL, error TEXT DEFAULT '', created_at TEXT DEFAULT CURRENT_TIMESTAMP,
 PRIMARY KEY(doc_id,version));
CREATE TABLE IF NOT EXISTS chunks (
 id TEXT PRIMARY KEY, doc_id TEXT NOT NULL, version INTEGER NOT NULL,
 position INTEGER NOT NULL, payload TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS chunk_document ON chunks(doc_id,version,position);
"""


class Catalog:
    def __init__(self, path, index):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.index = index
        self.options = ChunkingOptions(
            max_tokens=320, overlap_tokens=48, max_table_rows=100000
        )
        with self.connect() as c:
            c.executescript(SCHEMA)
            signature = json.dumps(
                {
                    "collection": index.collection,
                    "uri": index.uri,
                    "db": index.db_name,
                    "model": index.embedder.model_name,
                    "dim": index.embedder.dim,
                    "chunking": "320/48/v1",
                },
                sort_keys=True,
            )
            old = c.execute(
                "SELECT value FROM settings WHERE key='signature'"
            ).fetchone()
            if old and old[0] != signature:
                raise ValueError(
                    "目录绑定的模型/集合/切块策略不一致，请使用新目录及新集合"
                )
            c.execute(
                "INSERT OR IGNORE INTO settings VALUES ('signature',?)", (signature,)
            )

    @contextmanager
    def connect(self):
        c = sqlite3.connect(self.path, timeout=30)
        c.row_factory = sqlite3.Row
        try:
            c.execute("PRAGMA journal_mode=WAL")
            yield c
            c.commit()
        except Exception:
            c.rollback()
            raise
        finally:
            c.close()

    def ingest(self, path, root, *, parser_python="", parser_repo=""):
        path, root = Path(path).resolve(), Path(root).resolve()
        source = path.relative_to(
            root
        ).as_posix()  # 禁止退化成文件名，防止不同目录同名覆盖。
        sha = sha256_file(str(path))
        did = stable_doc_id(source)
        with self.path.with_suffix(".lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            self.index.ensure_ready()
            with self.connect() as c:
                old = c.execute(
                    "SELECT * FROM documents WHERE doc_id=?", (did,)
                ).fetchone()
                if old and old["sha256"] == sha:
                    count = self.index.client().query(
                        self.index.collection,
                        filter=f'doc_id == "{did}" and version == {old["active_version"]}',
                        output_fields=["count(*)"],
                        consistency_level="Strong",
                    )[0]["count(*)"]
                    if count == old["chunks"]:
                        return {
                            "status": "UNCHANGED",
                            "source": source,
                            "doc_id": did,
                            "version": old["active_version"],
                            "chunks": count,
                        }
                version = c.execute(
                    "SELECT COALESCE(MAX(version),0)+1 FROM versions WHERE doc_id=?",
                    (did,),
                ).fetchone()[0]
                c.execute(
                    "INSERT INTO versions(doc_id,version,sha256,status) VALUES (?,?,?,?)",
                    (did, version, sha, "STAGING"),
                )
            try:
                suffix = path.suffix.lower()
                pages = 0
                if suffix in {".txt", ".md", ".markdown"}:
                    doc = Document.from_text(
                        path.read_text(encoding="utf-8"), source, title=path.stem
                    )
                    chunks = (chunk_text if suffix == ".txt" else chunk_markdown)(
                        doc, self.options
                    )
                else:
                    connector = DocparseConnector(
                        DocparseOptions(repo=parser_repo, python=parser_python)
                    )
                    doc, ir = connector.parse(
                        path, self.path.parent / "parsed" / did / str(version)
                    )
                    if any(p.get("mode") == "failed" for p in ir.get("pages", [])):
                        raise ValueError("存在解析失败页，未发布新版本")
                    if ir.get("needs_review"):
                        raise ValueError("解析器要求人工复核，未发布新版本")
                    doc.source = source
                    doc.doc_id = did
                    pages = doc.metadata["pages"]
                    chunks = chunk_docparse_ir(ir, doc, self.options)
                if not chunks:
                    raise ValueError("解析没有可用正文")
                # 校验实际BGE tokenizer长度，禁止模型默默截断大表格/代码。
                model = self.index.embedder._load()
                for chunk in chunks:
                    text = (
                        chunk.section_path + "\n" if chunk.section_path else ""
                    ) + chunk.text
                    if len(model.tokenizer.encode(text)) > model.max_seq_length:
                        raise ValueError(
                            "结构块超过嵌入模型窗口，请拆分原文表格/代码后重试"
                        )
                    chunk.id = hashlib.sha256(
                        f"{did}:{version}:{chunk.id}".encode()
                    ).hexdigest()
                self.index.ingest(
                    did,
                    chunks,
                    version=version,
                    source=source,
                    sha256=sha,
                    replace_document=False,
                )
                actual = self.index.client().query(
                    self.index.collection,
                    filter=f'doc_id == "{did}" and version == {version}',
                    output_fields=["count(*)"],
                    consistency_level="Strong",
                )[0]["count(*)"]
                if actual != len(chunks):
                    raise RuntimeError("Milvus行数验证失败，保留旧激活版本")
                if sha256_file(str(path)) != sha:
                    raise RuntimeError("入库期间文件发生变化，未发布新版本")
                with self.connect() as c:
                    for chunk in chunks:
                        c.execute(
                            "INSERT INTO chunks VALUES (?,?,?,?,?)",
                            (
                                chunk.id,
                                did,
                                version,
                                chunk.index,
                                json.dumps(chunk.as_dict(), ensure_ascii=False),
                            ),
                        )
                    c.execute(
                        "INSERT INTO documents VALUES (?,?,?,?,?,?) ON CONFLICT(doc_id) DO UPDATE SET sha256=excluded.sha256,active_version=excluded.active_version,chunks=excluded.chunks,pages=excluded.pages",
                        (did, source, sha, version, len(chunks), pages),
                    )
                    c.execute(
                        "UPDATE versions SET status='ACTIVE' WHERE doc_id=? AND version=?",
                        (did, version),
                    )
                    c.execute(
                        "UPDATE versions SET status='SUPERSEDED' WHERE doc_id=? AND version<>? AND status='ACTIVE'",
                        (did, version),
                    )
                return {
                    "status": "INDEXED",
                    "source": source,
                    "doc_id": did,
                    "version": version,
                    "chunks": len(chunks),
                    "pages": pages,
                }
            except Exception as exc:
                with self.connect() as c:
                    c.execute(
                        "UPDATE versions SET status='FAILED',error=? WHERE doc_id=? AND version=?",
                        (str(exc), did, version),
                    )
                raise

    def active_filter(self, sources=None):
        with self.connect() as c:
            rows = c.execute("SELECT * FROM documents ORDER BY source").fetchall()
        if sources:
            rows = [r for r in rows if r["source"] in sources]
            if {r["source"] for r in rows} != set(sources):
                raise ValueError("指定来源未入库")
        return (
            " or ".join(
                f'(doc_id == "{r["doc_id"]}" and version == {r["active_version"]})'
                for r in rows
            )
            or 'doc_id == "__empty__"'
        )

    def list_documents(self):
        with self.connect() as c:
            return [
                dict(r) for r in c.execute("SELECT * FROM documents ORDER BY source")
            ]

    def neighbors(self, ids):
        """只加载激活版本的邻块。邻块仍需重排及预算筛选。"""
        result = {}
        with self.connect() as c:
            for key in ids:
                row = c.execute("SELECT * FROM chunks WHERE id=?", (key,)).fetchone()
                if row is None:
                    continue
                rows = c.execute(
                    """SELECT c.payload FROM chunks c JOIN documents d
                    ON c.doc_id=d.doc_id AND c.version=d.active_version
                    WHERE c.doc_id=? AND c.version=? AND c.position BETWEEN ? AND ?""",
                    (
                        row["doc_id"],
                        row["version"],
                        row["position"] - 1,
                        row["position"] + 1,
                    ),
                ).fetchall()
                for item in rows:
                    chunk = json.loads(item["payload"])
                    result[chunk["id"]] = {**chunk, "version": row["version"]}
        return list(result.values())
