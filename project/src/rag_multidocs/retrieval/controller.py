"""Core-only adaptive retrieval: query plans, RRF, coverage/MMR and budgets."""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass
from contextlib import nullcontext

from ..ingestion.chunking import estimate_tokens
from ..index.contracts import search_index
from ..model import normalize_hit
from ..execution import request_scope, remaining
from ..index.keyword import tokenize
from ..rerankers import RerankerRegistry
from .rewrite import level1_queries, validate_rewrites, constraints, comparison_parts, summary_parts, question_parts
from .needs import EvidenceNeed, build_needs, support, covered as relation_covered, directed_queries
from .sources import resolve_sources, without_source_names


@dataclass
class RetrievalPolicy:
    adaptive: bool = True
    candidate_k: int | None = None
    context_k: int | None = None
    context_tokens: int | None = None
    reranker: str | None = None
    alpha: float = 0.65
    beta: float = 0.20
    gamma: float = 0.15
    use_mmr: bool = True
    use_neighbors: bool = True
    use_recall: bool = True
    use_rewrite: bool = True
    use_coverage: bool = True
    max_index_calls: int = 4
    max_generation_calls: int = 2
    atomic_generation: bool = False
    allow_calculation: bool = False
    structured_table_answers: bool = False
    min_relevance: float = 0.05
    model_context_tokens: int = 8192
    output_tokens: int = 1024
    request_seconds: float = 120

    @classmethod
    def baseline(cls, **overrides):
        values = dict(adaptive=False, reranker="none", use_mmr=False, use_neighbors=False, use_recall=False, use_rewrite=False, use_coverage=False, alpha=1, beta=0, gamma=0)
        values.update(overrides)
        return cls(**values)

    def resources(self, query):
        kind = "fact"
        if re.search(r"比较|对比|区别|分别|跨文档|综合|异同|各自|不同|结合|总结|概括|归纳|分析|compare|versus", query, re.I) or len(re.findall(r"[？?]", query)) > 1:
            kind = "compare"
        elif re.search(r"为什么|如何|解释|原理|原因|怎样", query):
            kind = "explain"
        defaults = {
            "fact": (12, 2, 1000, "lightweight"),
            "explain": (24, 4, 2000, "lightweight"),
            "compare": (48, 6, 3000, "crossencoder"),
        }
        base = defaults[kind] if self.adaptive else (24, 4, 2000, "lightweight")
        if summary_parts(query):
            base = (base[0], 6, 3000, base[3])
        n, k, t, r = [
            override if override is not None else default
            for override, default in zip(
                (self.candidate_k, self.context_k, self.context_tokens, self.reranker), base
            )
        ]
        if not 1 <= n <= 48 or not 1 <= k <= 6 or not 64 <= t <= 3000:
            raise ValueError("resource limit: candidates<=48, context<=6, tokens<=3000")
        if not 1 <= self.max_index_calls <= 4:
            raise ValueError("max_index_calls must be in 1..4")
        if not 1 <= self.max_generation_calls <= 5:
            raise ValueError("max_generation_calls must be in 1..5")
        if (
            min(self.alpha, self.beta, self.gamma) < 0
            or abs(self.alpha + self.beta + self.gamma - 1) > 1e-6
        ):
            raise ValueError("coverage/MMR weights must be nonnegative and sum to 1")
        budget = min(t, self.model_context_tokens - self.output_tokens - 1600)
        if budget < 64:
            raise ValueError("model context has no evidence budget")
        return {
            "kind": kind,
            "candidate_k": n,
            "context_k": k,
            "context_tokens": budget,
            "reranker": r,
        }


def keywords(query):
    text = re.sub(
        r"请问|请|为什么|如何|怎样|是什么|多少|什么|哪个|是否|以及|之间|有什么|的|是|了|吗|呢",
        " ",
        query,
    )
    return list(dict.fromkeys(re.findall(r"[A-Za-z][A-Za-z0-9_+.-]*|[一-鿿]{2,}", text)))


def requirements(query):
    if re.search(r"比较|对比|区别|异同", query):
        clean = re.sub(r"请问|比较|对比", "", query)
        parts = re.split(r"与|和|及|、| versus | vs ", clean, maxsplit=3)
        if len(parts) > 1:
            values = [
                re.split(r"的|区别|差异|有何|有什么|哪个|如何", p)[0].strip(" ？?，,")
                for p in parts
            ]
            return [p for p in values if p][:3]
    return [p.strip() for p in re.split(r"[？?；;]", query) if p.strip()][:3] or [query]


def coverage(text, need):
    compact = re.sub(r"\s+", "", text).lower()
    if re.sub(r"\s+", "", need).lower() in compact:
        return 1.0
    terms = keywords(need)
    if not terms:
        return 0.0
    matched = 0.0
    for term in terms:
        if term.lower() in compact:
            matched += 1
            continue
        if len(term) >= 7 and re.fullmatch("[一-鿿]+", term):
            pairs = {term[i : i + 2] for i in range(len(term) - 1)}
            matched += sum(pair in compact for pair in pairs) / max(1, len(pairs))
    return matched / len(terms)


def similarity(a, b):
    a, b = set(tokenize(a)), set(tokenize(b))
    return len(a & b) / max(1, len(a | b))


def comparison_facets(query, candidates):
    """Literal object/event facets; aliases come only from source headings.

    These are ranking queries, not inferred facts or sufficiency judgments.
    The complete original query remains the retrieval/generation constraint.
    """
    parts = comparison_parts(query)
    if not parts:
        return list(summary_parts(query) or [part + ' ' + query for part in question_parts(query)])
    first, second, event = parts
    # A temporal modifier is retained in the full query. The facet ranks the
    # event itself, so generic era mentions cannot dominate concrete evidence.
    event = re.sub(r"^[^\d，,]{1,8}时期", "", event)
    result = []
    for obj in (first, second):
        aliases = set()
        for hit in candidates:
            for heading in hit.get("section_path", "").split(" > "):
                heading = re.sub(r"^[一二三四五六七八九十\d]+[、.]\s*", "", heading)
                names = heading.split("/")
                if len(names) > 1 and obj in names:
                    aliases.update(name.strip() for name in names if name.strip() != obj)
        alias = max(aliases, key=lambda name: sum(h["text"].count(name) for h in candidates), default="")
        result.append(" ".join(part for part in (obj, alias, event, " ".join(sorted(constraints(query)))) if part))
    return result


def select_context(candidates, needs, resources, policy, required_sources=()):
    selected = []
    remaining = list(candidates)
    covered = set()
    used = 0
    source_counts = dict.fromkeys(required_sources, 0)
    source_quota = max(1, resources["context_k"] // max(1, len(source_counts)))
    selected_groups = set()
    def matched(hit, need):
        if isinstance(need, EvidenceNeed) and need.kind != "topic":
            return support(hit, need)["supported"]
        return coverage(hit["text"], need.label if isinstance(need, EvidenceNeed) else need) >= 0.5
    while remaining and len(selected) < resources["context_k"]:

        def utility(h):
            rel = h.get("selection_relevance", h.get("rerank_score", h.get("normalized_relevance", 0)))
            red = (
                max((similarity(h["text"], s["text"]) for s in selected), default=0)
                if policy.use_mmr
                else 0
            )
            gain = sum(
                i not in covered and matched(h,n) and (not isinstance(n,EvidenceNeed) or n.kind != "topic") for i,n in enumerate(needs)
            ) / max(1, len(needs)) if policy.use_coverage else 0
            return policy.alpha * rel - policy.beta * red + policy.gamma * gain

        reserved = [h for h in remaining if h.get("doc_id") in source_counts
                    and source_counts[h["doc_id"]] < source_quota]
        facet_pool = [h for h in remaining if set(h.get("context_coverage_groups", [])) - selected_groups]
        if facet_pool:
            # First retain actual evidence for each source/subquestion pair;
            # source counts alone do not establish useful source coverage.
            pool = facet_pool
        elif reserved:
            least = min(source_counts[h["doc_id"]] for h in reserved)
            pool = [h for h in reserved if source_counts[h["doc_id"]] == least]
        else:
            pool = remaining
        hit = max(pool, key=utility)
        remaining.remove(hit)
        if policy.use_mmr and any(similarity(hit["text"], s["text"]) > 0.9 and
                                 (not required_sources or hit.get('doc_id') == s.get('doc_id')) for s in selected):
            continue
        cost = (
            estimate_tokens(hit["text"])
            + estimate_tokens(hit.get("source", ""))
            + estimate_tokens(hit.get("section_path", ""))
            + 80
        )
        if used + cost > resources["context_tokens"]:
            continue
        selected.append(hit)
        selected_groups.update(hit.get("context_coverage_groups", []))
        if hit.get("doc_id") in source_counts:
            source_counts[hit["doc_id"]] += 1
        used += cost
        covered.update(i for i, n in enumerate(needs) if matched(hit,n))
    return selected, [n for i, n in enumerate(needs) if i not in covered], used


class RetrievalController:
    def __init__(self, index, rerankers=None, policy=None, rewriter=None):
        self.index = index
        self.rerankers = rerankers or RerankerRegistry()
        self.policy = policy or RetrievalPolicy()
        self.rewriter = rewriter

    @staticmethod
    def search_hits(index, query, top_k=5, doc_ids=None, filters=None):
        """Compatibility path: retains backend fusion/default rank and list return."""
        return search_index(index, query, top_k=top_k, doc_ids=doc_ids, filters=filters)

    def retrieve(self, query, *, doc_ids=None, filters=None, policy=None):
        policy = policy or self.policy
        catalog = getattr(self.index, "version_catalog", None)
        with request_scope(policy.request_seconds) as budget:
            with catalog.read_snapshot() if catalog else nullcontext(None):
                result = self._retrieve(query, doc_ids=doc_ids, filters=filters, policy=policy)
                result["usage"] = dict(budget.counters)
                return result

    def _retrieve(self, query, *, doc_ids=None, filters=None, policy=None):
        policy = policy or self.policy
        if not query.strip() or len(query) > 2000:
            raise ValueError("invalid query")
        catalog = getattr(self.index, "version_catalog", None)
        source_index = catalog or self.index
        records = source_index.source_records() if hasattr(source_index, "source_records") else []
        constraint = resolve_sources(query, records, doc_ids)
        if constraint['blocked']:
            return {'hits': [], 'missing_evidence': [query], 'retrieval_trace': [],
                    'warnings': constraint['warnings'], 'context_tokens': 0, 'index_calls': 0,
                    'source_constraints': constraint}
        doc_ids = constraint['doc_ids']
        original_query = query
        if constraint['matches']:
            # Source identity is already enforced by the index filter. File
            # names and the now-empty "according to" clause are not content.
            query = without_source_names(query, constraint)
            query = re.sub(r"\s+", " ", query).strip()
            query = re.sub(r"^(?:根据|结合|综合|在|从)\s*(?:(?:和|与|及|、)\s*)*[，,：:]\s*", "", query)
            query = query.strip(" ，,：:") or original_query
        resources = policy.resources(query)
        summary_facets = summary_parts(query)
        factual_facets = question_parts(query) if not summary_facets else []
        focused_parts = summary_facets or factual_facets
        needs = build_needs(query)
        warnings, trace, aux, candidates, scores = [], [], [], {}, {}
        rewrite_seconds = 0.0
        rewrite_before = getattr(self.rewriter, "model_calls", 0)
        if policy.use_rewrite:
            started = time.perf_counter()
            aux.extend(level1_queries(query))
            if self.rewriter and resources["kind"] == "compare":
                try:
                    remaining(policy.request_seconds)
                    generated = self.rewriter(query)
                    generated = validate_rewrites(query, generated)
                    aux = generated + aux
                except Exception:
                    warnings.append(
                        {
                            "code": "QUERY_REWRITE_FAILED",
                            "message": "Original query retained",
                            "severity": "warning",
                        }
                    )
            aux = list(dict.fromkeys(q for q in aux if q != query))[:2]
            rewrite_seconds = time.perf_counter() - started
        calls, tokens = 0, 0
        context, missing, ranked = [], list(needs), []
        seen_contributions = set()
        route_leaders = set()
        catalog = getattr(self.index, "version_catalog", None)
        for round_no in range(2 if policy.use_recall else 1):
            if round_no and calls >= policy.max_index_calls:
                break
            started = time.perf_counter()
            remaining(policy.request_seconds)
            if round_no == 0:
                queries = [query] + aux
            else:
                # A query cannot repair a generator omission, nor an already
                # available relation excluded by selection/budget.
                unavailable = [n for n in missing if not any(support(h,n)["supported"] for h in ranked)]
                if missing and not unavailable:
                    warnings.append({"code":"CONTEXT_SELECTION_GAP", "message":"Required relations exist in candidates; no redundant recall", "severity":"warning"})
                    break
                directed = [q for q in directed_queries(unavailable) if q not in aux and q != query][: 3 - len(aux)]
                directed = [q + (" " + " ".join(sorted(constraints(query))) if constraints(query) else "") for q in directed]
                queries = directed or [query]
                aux.extend(directed)
                if not directed:
                    warnings.append(
                        {
                            "code": "GAP_UNCERTAIN",
                            "message": "Expanded original retrieval",
                            "severity": "warning",
                        }
                    )
                if policy.adaptive and (not context or unavailable):
                    resources = {**resources, "escalated": True,
                        "candidate_k": resources["candidate_k"] if policy.candidate_k is not None else min(48,resources["candidate_k"]*2),
                        "context_k": resources["context_k"] if policy.context_k is not None else min(6,max(4,resources["context_k"])),
                        "context_tokens": resources["context_tokens"] if policy.context_tokens is not None else min(3000,policy.model_context_tokens-policy.output_tokens-1600,max(2000,resources["context_tokens"]))}
            cap = min(48, resources["candidate_k"] * (2 if round_no and not policy.adaptive else 1))
            snapshot = catalog.active() if catalog else None
            per_query = []
            jobs = [(q, doc_ids) for q in queries]
            for job_index, (q, scope) in enumerate(jobs):
                if calls >= policy.max_index_calls:
                    warnings.append({"code":"RETRIEVAL_CALL_BUDGET", "message":"Index call limit reached", "severity":"warning"})
                    break
                timeout = remaining(policy.request_seconds)
                hits = search_index(
                    self.index,
                    q,
                    top_k=cap,
                    doc_ids=scope,
                    filters=filters,
                    snapshot=snapshot,
                    fusion=getattr(self.index, "fusion", "rrf"),
                    ann_limit=cap * 4,
                    timeout_s=timeout,
                )
                if catalog:
                    hits = catalog.attach_metadata(hits, snapshot)
                calls += 1
                per_query.append(
                    {"query": q, "returned": min(len(hits), cap), "ann_limit_per_route": cap * 4,
                     "doc_ids": scope}
                )
                for rank, raw in enumerate(hits[:cap], 1):
                    h = normalize_hit(raw)
                    if scope is not None and h.get('doc_id') not in scope:
                        continue
                    if snapshot is not None and snapshot.get(h.get("doc_id")) != (
                        h.get("version") or 0
                    ):
                        continue
                    key = h["chunk_id"]
                    route_budget = min(48, cap * max(len(doc_ids or ()), len(focused_parts or ()), 1)) // policy.max_index_calls
                    if rank <= max(1, route_budget):
                        route_leaders.add(key)
                    candidates[key] = h
                    if (q,key) not in seen_contributions:
                        scores[key] = scores.get(key, 0) + 1 / (60 + rank)
                        seen_contributions.add((q,key))
                if round_no == 0 and job_index == 0 and constraint["explicit"] and (len(doc_ids or ()) > 1 or focused_parts):
                    # Pooled top-k can crowd out an explicitly required source.
                    # Recover it using the same filters, snapshot and index-call budget.
                    represented = {did: sum(h.get('doc_id') == did for h in candidates.values()) for did in doc_ids}
                    facets = comparison_facets(query, list(candidates.values()))
                    focused = facets[0] if facets else query
                    if focused_parts:
                        required_jobs = [(facet + (' ' + query if factual_facets else ''), [did])
                                         for facet in focused_parts for did in doc_ids]
                    else:
                        required_jobs = [(focused, [did]) for did in sorted(doc_ids, key=lambda did: represented[did])
                                         if represented[did] < cap]
                    # Required-source recall precedes optional rewrites. Both
                    # still share the same four-call request budget.
                    jobs[1:1] = required_jobs
            pool_cap = min(48, cap * max(len(doc_ids or ()), len(focused_parts or ()))) if constraint['explicit'] and (len(doc_ids or ()) > 1 or focused_parts) else cap
            fused = sorted(candidates.values(), key=lambda h: scores[h['chunk_id']], reverse=True)
            if constraint['explicit'] and (len(doc_ids or ()) > 1 or focused_parts):
                # RRF must not evict the best result of a focused source route
                # merely because generic passages occur in more query lists.
                fused = ([h for h in fused if h['chunk_id'] in route_leaders]
                         + [h for h in fused if h['chunk_id'] not in route_leaders])
                if len(doc_ids or ()) > 1:
                    # Apply the existing source quota before the pool cutoff,
                    # too. Otherwise repeated generic routes can remove an
                    # entire source's concrete facts before reranking sees it.
                    quota = pool_cap // len(doc_ids)
                    reserved = {h['chunk_id'] for did in doc_ids
                                for h in [x for x in fused if x['doc_id'] == did][:quota]}
                    fused = ([h for h in fused if h['chunk_id'] in reserved]
                             + [h for h in fused if h['chunk_id'] not in reserved])
            ordered = fused[:pool_cap]
            base = [dict(h) for h in ordered]
            maximum = max(scores.values(), default=1)
            for h in ordered:
                h["rrf_score"] = scores[h["chunk_id"]]
                h["rank_score"] = scores[h["chunk_id"]] / maximum
            neighbor_index = catalog or self.index
            coherent_neighbors = set()
            if policy.use_neighbors and hasattr(neighbor_index, "neighbors"):
                neighbors = neighbor_index.neighbors([h["chunk_id"] for h in ordered[:3]], snapshot)
                coherent = len(ordered) >= 3 and len({h.get('doc_id') for h in ordered[:3]}) == 1
                if coherent:
                    coherent_neighbors = {h['chunk_id'] for h in neighbors
                                          if (not doc_ids or h['doc_id'] in doc_ids)
                                          and all(h.get(k) == v for k, v in (filters or {}).items())}
                known = {h["chunk_id"] for h in ordered}
                additions = [
                    normalize_hit(h)
                    for h in neighbors
                    if h["chunk_id"] not in known
                    and (not doc_ids or h["doc_id"] in doc_ids)
                    and all(h.get(k) == v for k, v in (filters or {}).items())
                    and (coherent or max((coverage(h["text"], n.label) for n in needs), default=0) >= 0.5)
                ][:6]
                ordered = ordered[: pool_cap - len(additions)] + additions
            for h in ordered:
                # Lexical overlap is a weak support signal, never an entailment judgment.
                h["support_score"] = coverage(h["text"], query)
                h["support_signal"] = "lexical_overlap_uncalibrated"
                h["normalized_relevance"] = h.get(
                    "rank_score", 0
                )  # compatibility ranking alias only
            embedder = getattr(self.index, "embedder", None)
            if embedder and ordered and resources["reranker"] != "none":
                try:
                    remaining(policy.request_seconds)
                    vectors = embedder.embed([h["text"] for h in ordered])
                    qv = embedder.embed([query], query=True)[0]
                    import math

                    for h, v in zip(ordered, vectors):
                        norm = math.sqrt(sum(x * x for x in qv) * sum(x * x for x in v)) or 1
                        h["semantic_score"] = max(
                            0.0, min(1.0, sum(x * y for x, y in zip(qv, v)) / norm)
                        )
                        h["support_score"] = h["semantic_score"]
                        h["support_signal"] = "cosine_uncalibrated"
                except Exception:
                    warnings.append(
                        {
                            "code": "SEMANTIC_SCORING_FAILED",
                            "message": "Using lexical support signal, not RRF threshold",
                            "severity": "warning",
                        }
                    )
            rerank_started = time.perf_counter()
            if isinstance(self.rerankers, RerankerRegistry):
                ranked, issues = self.rerankers.rerank(
                    resources["reranker"],
                    query,
                    ordered,
                    timeout_s=remaining(policy.request_seconds),
                )
            else:
                ranked, issues = self.rerankers.rerank(resources["reranker"], query, ordered)
            warnings.extend(issues)
            rerank_seconds = time.perf_counter()-rerank_started
            if coherent_neighbors and all(h['chunk_id'] in coherent_neighbors for h in ranked[:3]):
                # The leading hits agree on one document. Keep their existing
                # adjacent source passages ahead of unrelated tail hits, under
                # the same context count/token budget. This is a context signal,
                # never a claim that the passages prove an answer.
                ranked = ([h for h in ranked if h['chunk_id'] in coherent_neighbors]
                          + [h for h in ranked if h['chunk_id'] not in coherent_neighbors])
                for hit in ranked:
                    if hit['chunk_id'] in coherent_neighbors:
                        hit['context_neighbor'] = True
            for ordinal, hit in enumerate(ranked):
                hit.pop('context_coverage_groups', None)
                ordinal_relevance = 1 - ordinal / max(1,len(ranked)-1)
                relational = [n for n in needs if n.kind != "topic"]
                explicit_match = any(support(hit,n)["supported"] for n in relational)
                hit["selection_relevance"] = (ordinal_relevance + float(explicit_match))/2 if policy.use_coverage and relational else ordinal_relevance
                hit["selection_relevance_kind"] = "ordinal_plus_explicit_relation" if policy.use_coverage and relational else "ordinal_rank_only"
            facet_trace = []
            if constraint['explicit'] and (len(constraint['doc_ids'] or ()) > 1 or focused_parts):
                facets = comparison_facets(query, ranked)
                for facet_no, facet in enumerate(facets):
                    for did in constraint['doc_ids']:
                        scoped = [h for h in ranked if h['doc_id'] == did]
                        if re.search(r'已经?完成|已取得|仍然|现存|当前|目前', facet):
                            scoped = [h for h in scoped if not re.search(r'下一步|后续(?:计划|安排)|工作计划', h.get('section_path', ''))
                                      or re.search(r'已经?完成|完成了|仍然存在|目前存在', h['text'])]
                        if summary_facets and re.search(r'问题|挑战|困难|不足', facet):
                            explicit_sections = [h for h in scoped if re.search(r'问题|挑战|困难|不足', h.get('section_path', ''))]
                            scoped = explicit_sections or scoped
                        if not scoped:
                            continue
                        if resources['reranker'] == 'none':
                            scoped = sorted(scoped, key=lambda h: coverage(h['text'], facet), reverse=True)
                        else:
                            kwargs = ({'timeout_s': remaining(policy.request_seconds)}
                                      if isinstance(self.rerankers, RerankerRegistry) else {})
                            scoped, facet_issues = self.rerankers.rerank(
                                resources['reranker'], facet, scoped, **kwargs)
                            warnings.extend(facet_issues)
                        for member, picked in enumerate(scoped[:2 if summary_facets else 1]):
                            chosen = next(h for h in ranked if h['chunk_id'] == picked['chunk_id'])
                            group = f'{did}:{member}:{facet_no}'
                            chosen.setdefault('context_coverage_groups', []).append(group)
                        facet_trace.append({'query': facet, 'doc_id': did, 'selected_ids': [h['chunk_id'] for h in scoped[:2 if summary_facets else 1]],
                                            'ranked_ids': [h['chunk_id'] for h in scoped]})
            rerank_seconds = time.perf_counter() - rerank_started
            context, missing, tokens = select_context(ranked, needs, resources, policy,
                                                     constraint['doc_ids'] if constraint['explicit'] else ())
            trace.append(
                {
                    "round": round_no + 1,
                    "original_query": original_query,
                    "rewrite_output": getattr(self.rewriter, "last_output", None),
                    "source_facet_rankings": facet_trace,
                    "source_doc_ids": doc_ids,
                    "queries": queries,
                    "per_query": per_query,
                    "candidate_count": len(ordered),
                    "rerank_seconds": rerank_seconds,
                    "reranker_execution": {k:v for k,v in getattr(self.rerankers,'last_execution',{}).items() if k not in {'rows'}},
                    "base_candidates": base,
                    "reranked_candidates": [dict(h) for h in ranked],
                    "candidates": ranked,
                    "context_ids": [h["chunk_id"] for h in context],
                    "missing_evidence": [n.label for n in missing],
                    "evidence_needs": [n.as_dict() for n in needs],
                    "finite_relation_coverage": {stage: sorted(relation_covered(hits,needs)) for stage,hits in [("base",base),("reranked",ranked),("reranked_top_context",ranked[:resources["context_k"]]),("context",context)]},
                    "gap_stages": [{"need":n.label,"stage":"context_selection_or_budget" if any(support(h,n)["supported"] for h in ranked) else "initial_recall_or_unknown_relation"} for n in missing],
                    "coverage_kind": "explicit_numeric_relations_with_separate_theme_signals_not_general_semantics",
                    "context_tokens": tokens,
                    "resources": {**resources, "ann_limit_per_route": cap * 4},
                    "seconds": time.perf_counter() - started,
                }
            )
            if context and not missing:
                break
        if missing and policy.use_coverage:
            warnings.append(
                {
                    "code": "EVIDENCE_GAP",
                    "message": "Lexical needs not covered; not semantic proof",
                    "needs": [n.label for n in missing],
                    "severity": "warning",
                }
            )
        return {
            "hits": context,
            "source_constraints": constraint,
            "missing_evidence": [n.label for n in missing],
            "retrieval_trace": trace,
            "warnings": warnings,
            "context_tokens": tokens,
            "index_calls": calls,
            "rewrite_seconds": rewrite_seconds,
            "rewrite_model_calls": getattr(self.rewriter, "model_calls", 0) - rewrite_before,
            "auxiliary_queries": aux,
            "policy": asdict(policy),
            "evidence_support": "heuristic_only",
        }
