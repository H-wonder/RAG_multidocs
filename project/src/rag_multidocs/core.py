"""Public service API for teammate-owned CLI/REPL/Web.

No interaction imports. Embedding/LLM factories are lazy and remain Providers,
not additional plugin extension points.
"""

from __future__ import annotations

from .ingestion.service import IngestService
from .execution import request_scope
from contextlib import nullcontext
from dataclasses import replace
import time
from .qa.llm import answer_with_evidence
from .retrieval.controller import RetrievalController
from .retrieval.profiles import recommended_policies


class RAGCore:
    def __init__(
        self,
        registry,
        index,
        *,
        rerankers=None,
        policy=None,
        chat=None,
        chat_factory=None,
        chat_budget_factory=None,
        rewriter=None,
        chunk_options=None,
        support_level3=True,
        catalog_path=None,
    ):
        self.index = index
        catalog = None
        if catalog_path is not None:
            from .ingestion.catalog import DocumentCatalog
            catalog = DocumentCatalog(index, catalog_path)
        self.ingestion = IngestService(registry, index, chunk_options, catalog=catalog)
        self.controller = RetrievalController(
            index, rerankers, policy or recommended_policies()['stable'], rewriter
        )
        self.support_level3 = support_level3
        self.chat = chat
        self.chat_factory = chat_factory
        self.chat_budget_factory = chat_budget_factory

    def ingest(self, connector, path, options=None):
        return self.ingestion.ingest(connector, path, options)

    def ingest_many(self, connector, paths, options=None):
        return self.ingestion.ingest_many(connector, paths, options)

    def search(self, query, *, doc_ids=None, filters=None, policy=None):
        policy = policy or self.controller.policy
        with request_scope(policy.request_seconds) as budget:
            started = time.perf_counter()
            try:
                result = self.controller.retrieve(query, doc_ids=doc_ids, filters=filters, policy=policy)
            except (TimeoutError, OSError, RuntimeError) as exc:
                result = {
                    "hits": [],
                    "missing_evidence": [query],
                    "retrieval_trace": [],
                    "context_tokens": 0,
                    "index_calls": 0,
                    "retrieval_error": True,
                    "warnings": [
                        {
                            "code": "RETRIEVAL_FAILED",
                            "message": type(exc).__name__,
                            "severity": "error",
                        }
                    ],
                }
            result["usage"] = dict(budget.counters)
            result["retrieval_seconds"] = time.perf_counter() - started
            return result

    def ask(self, query, *, style="professional", doc_ids=None, filters=None):
        catalog = self.ingestion.catalog
        policy = self.controller.policy
        chat_budget = self.chat_budget_factory() if self.chat_budget_factory else getattr(self.chat, "budget", None)
        if chat_budget:
            policy = replace(policy, model_context_tokens=chat_budget.context_window,
                             output_tokens=chat_budget.retry_tokens)
        with request_scope(policy.request_seconds) as budget:
            with catalog.read_snapshot() if catalog else nullcontext(None):
                started = time.perf_counter()
                result = self._ask(query, style=style, doc_ids=doc_ids, filters=filters, policy=policy)
                result["usage"] = dict(budget.counters)
                result["total_seconds"] = time.perf_counter() - started
                if chat_budget:
                    result["token_budget"] = chat_budget.as_trace()
                return result

    def _ask(self, query, *, style="professional", doc_ids=None, filters=None, policy):
        if policy.use_rewrite and self.controller.rewriter is None:
            # CLI/REPL and API must use the same existing semantic rewriter.
            # Keep retrieval-only calls free of implicit generator creation.
            try:
                if self.chat is None and self.chat_factory:
                    self.chat = self.chat_factory()
                if self.chat is not None:
                    from .retrieval.rewrite import SemanticRewriter
                    self.controller.rewriter = SemanticRewriter(self.chat)
            except Exception:
                pass  # Retrieval remains available; generation reports init errors below.
        result = self.search(query, doc_ids=doc_ids, filters=filters, policy=policy)
        common = {
            "retrieval_trace": result["retrieval_trace"],
            "warnings": result["warnings"],
            "context_tokens": result["context_tokens"],
            "evidence": result["hits"],
            "generation_evidence": [],
            "generation_context_tokens_estimate": 0,
            "rewrite_seconds": result.get("rewrite_seconds", 0),
            "retrieval_seconds": result.get("retrieval_seconds", 0),
            "index_calls": result["index_calls"],
            "evidence_support": "heuristic_only",
            "source_constraints": result.get('source_constraints', {}),
        }
        if not result["hits"]:
            return {
                **common,
                "answer": "没有检索到相关依据。",
                "status": "generation_failed"
                if result.get("retrieval_error")
                else "insufficient_evidence",
                "citations": [],
                "used_llm": False,
                "model_calls": 0,
            }
        if self.chat is None and self.chat_factory:
            try:
                self.chat = self.chat_factory()
            except Exception:
                return {
                    **common,
                    "answer": "回答Provider初始化失败。",
                    "status": "generation_failed",
                    "citations": [],
                    "used_llm": False,
                    "model_calls": 0,
                    "warnings": common["warnings"]
                    + [{"code": "LLM_INIT_ERROR", "message": "Provider初始化失败"}],
                }
        if self.chat is None:
            return {
                **common,
                "answer": "未配置回答Provider；检索证据已保留。",
                "status": "generation_failed",
                "citations": [],
                "used_llm": False,
                "model_calls": 0,
            }
        started = time.perf_counter()
        answer = answer_with_evidence(
            query, result['hits'], self.chat, style=style,
            context_tokens=policy.model_context_tokens,
            output_tokens=policy.output_tokens,
            allow_calculation=policy.allow_calculation,
            check_numbers=self.support_level3,
            structured_table_answers=policy.structured_table_answers,
            source_constraints=result.get('source_constraints'),
        )
        return {
            **common,
            **answer,
            "generation_seconds": time.perf_counter() - started,
            "warnings": common["warnings"] + answer["warnings"],
        }

    def delete(self, doc_id):
        if self.ingestion.catalog is None:
            self.index.remove_doc(doc_id)
            self.index.persist()
            return {"doc_id": doc_id, "status": "deleted"}
        return self.ingestion.catalog.delete(doc_id)

    def cleanup(self, doc_id=None):
        if self.ingestion.catalog is None:
            raise ValueError("INDEX_CAPABILITY_UNAVAILABLE: version cleanup")
        return self.ingestion.catalog.cleanup(doc_id)

    def close(self):
        close = getattr(self.controller.rerankers, "close", None)
        if close:
            close()
        if self.ingestion.catalog:
            self.ingestion.catalog.close()
        self.index.close()

    def list_documents(self):
        return self.index.list_docs()

    def stats(self):
        from .index.contracts import supports_versions

        return {
            **self.index.stats(),
            "capabilities": {"version_publication": supports_versions(self.index)},
            "recovery_warnings": self.ingestion.catalog.recovery_warnings
            if self.ingestion.catalog
            else [],
        }

    def plugins(self):
        connectors = self.ingestion.registry.plugin_infos()
        ranks = self.controller.rerankers.plugin_infos()
        return {
            "connectors": [
                {k: v for k, v in i.as_dict().items() if k != "config"} for i in connectors
            ],
            "rerankers": [{k: v for k, v in i.as_dict().items() if k != "config"} for i in ranks],
            "warnings": self.ingestion.registry.discovery_issues
            + self.controller.rerankers.discovery_issues,
        }

    def recover(self):
        if self.ingestion.catalog is None:
            raise ValueError("INDEX_CAPABILITY_UNAVAILABLE: recovery")
        return self.ingestion.catalog.reconcile()

    def answer_evidence(self, question, hits, *, style="professional"):
        """Answer supplied evidence through the same normal Core QA path."""
        if self.chat is None:
            raise ValueError("CHAT_PROVIDER_REQUIRED")
        budget = self.chat_budget_factory() if self.chat_budget_factory else getattr(self.chat, 'budget', None)
        with request_scope(self.controller.policy.request_seconds):
            return answer_with_evidence(
                question, hits, self.chat, style=style,
                context_tokens=budget.context_window if budget else self.controller.policy.model_context_tokens,
                output_tokens=budget.retry_tokens if budget else self.controller.policy.output_tokens,
                allow_calculation=self.controller.policy.allow_calculation,
                check_numbers=self.support_level3,
                structured_table_answers=self.controller.policy.structured_table_answers,
            )

    def document(self, doc_id):
        """Latest source metadata/IR; history is not silently substituted."""
        import json

        catalog = self.ingestion.catalog
        if catalog is None:
            lookup = getattr(self.index, "document", None)
            if lookup:
                return lookup(doc_id)
            raise ValueError("INDEX_CAPABILITY_UNAVAILABLE: document metadata")
        with catalog.read_snapshot() as active, catalog.lock:
            row = catalog.db.execute(
                "SELECT active_version,payload FROM documents WHERE doc_id=? AND deleted=0",
                (doc_id,),
            ).fetchone()
            if row is None:
                return None
            if active.get(doc_id) != row["active_version"]:
                raise RuntimeError("DOCUMENT_SNAPSHOT_CHANGED: retry metadata request")
            return {**json.loads(row["payload"]), "version": row["active_version"]}
