"""RAG策略：候选重排、多样性选择、上下文预算和可审计重召回。"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, replace

from chunking import estimate_tokens
from grounded_answer import generate_answer
from keyword_index import KeywordIndex, tokenize


@dataclass
class Strategy:
    candidate_k: int = 24
    context_k: int = 3
    context_tokens: int = 1800
    min_dense: float = 0.35
    mmr_lambda: float = 0.8
    reranker: str = "lightweight"
    rerank_model: str = "BAAI/bge-reranker-base"
    tone: str = "professional"
    retry_recall: bool = True

    def __post_init__(self):
        if (
            not 1 <= self.context_k <= 10
            or not self.context_k <= self.candidate_k <= 100
        ):
            raise ValueError("context_k需为1~10，candidate_k需介于context_k和100之间")
        if not 128 <= self.context_tokens <= 3000 or not 0 <= self.mmr_lambda <= 1:
            raise ValueError("上下文预算需为128~3000，MMR权重需为0~1")
        if not -1 <= self.min_dense <= 1:
            raise ValueError("min_dense需为-1~1")
        if self.reranker not in {"lightweight", "crossencoder"}:
            raise ValueError("未知重排方式")
        if self.tone not in {"professional", "teaching", "concise"}:
            raise ValueError("未知回答口吻")


def overlap(a: str, b: str) -> float:
    x, y = set(tokenize(a)), set(tokenize(b))
    return len(x & y) / max(1, len(x | y))


class Reranker:
    def __init__(self, embedder, config: Strategy):
        self.embedder, self.config, self.cross = embedder, config, None

    def rank(self, question: str, hits: list[dict]) -> list[dict]:
        if not hits:
            return []
        vectors = self.embedder.embed([h["text"] for h in hits])
        q = self.embedder.embed([question], query=True)[0]
        lexical = KeywordIndex()
        lexical.add([{**h, "id": h["id"]} for h in hits])
        bm25 = {
            h["chunk_id"]: h["score"] for h in lexical.search(question, top_k=len(hits))
        }
        ceiling = max(bm25.values(), default=1) or 1
        cross_scores = None
        if self.config.reranker == "crossencoder":
            if self.cross is None:
                from sentence_transformers import CrossEncoder
                from torch import nn

                self.cross = CrossEncoder(
                    self.config.rerank_model,
                    device="cpu",
                    max_length=512,
                    activation_fn=nn.Identity(),
                )
            cross_scores = self.cross.predict(
                [(question, h["text"]) for h in hits], batch_size=8
            )
        ranked = []
        for i, (h, v) in enumerate(zip(hits, vectors)):
            norm = math.sqrt(sum(x * x for x in q) * sum(x * x for x in v)) or 1
            dense = sum(x * y for x, y in zip(q, v)) / norm
            sparse = bm25.get(h["id"], 0) / ceiling
            score = 0.8 * dense + 0.2 * sparse
            if cross_scores is not None:
                score = 1 / (1 + math.exp(-max(-60, min(60, float(cross_scores[i])))))
            ranked.append(
                {
                    **h,
                    "dense_score": dense,
                    "lexical_score": sparse,
                    "rerank_score": score,
                }
            )
        return sorted(ranked, key=lambda h: h["rerank_score"], reverse=True)


def select_context(ranked: list[dict], config: Strategy) -> list[dict]:
    """MMR减少重复；整块装箱，不截断表格和公式，超预算跳过。"""
    remaining, selected, used = list(ranked), [], 0
    while remaining and len(selected) < config.context_k:

        def utility(hit):
            redundant = max(
                (overlap(hit["text"], other["text"]) for other in selected), default=0
            )
            return (
                config.mmr_lambda * hit["rerank_score"]
                - (1 - config.mmr_lambda) * redundant
            )

        hit = max(remaining, key=utility)
        remaining.remove(hit)
        if any(overlap(hit["text"], other["text"]) > 0.9 for other in selected):
            continue
        cost = (
            estimate_tokens(hit["text"])
            + estimate_tokens(str(hit.get("source", "")))
            + 40
        )
        if used + cost > config.context_tokens:
            continue
        selected.append(hit)
        used += cost
    return selected


class RagEngine:
    def __init__(
        self,
        index,
        config: Strategy | None = None,
        answer_fn=generate_answer,
        neighbor_fn=None,
    ):
        self.index = index
        self.config = config or Strategy()
        self.reranker = Reranker(index.embedder, self.config)
        self.answer_fn = answer_fn
        self.neighbor_fn = neighbor_fn

    def ask(
        self,
        question: str,
        *,
        filter_expr: str,
        model="gemma3:4b",
        base_url="http://127.0.0.1:11434",
        history=None,
    ):
        if not question.strip() or len(question) > 1000:
            raise ValueError("问题需为1~1000字符")
        # 仅将最近两个用户问题用于指代消解，历史助手回答从不作为证据。
        history = history or []
        if not isinstance(history, list) or any(
            not isinstance(m, dict) or not isinstance(m.get("content"), str)
            for m in history
        ):
            raise ValueError("history必须是包含字符串content的消息列表")
        previous = [m["content"][-200:] for m in history if m.get("role") == "user"][
            -2:
        ]
        retrieval_question = question
        generation_question = question
        if previous and any(
            word in question for word in ["它", "这个", "上述", "那么", "为什么呢"]
        ):
            retrieval_question = "；".join(previous + [question])
            generation_question = f"当前问题：{question}\n用户此前的问题（只用于理解指代，不是事实证据）：{'；'.join(previous)}"
        trace = []
        answer = {
            "status": "INSUFFICIENT_EVIDENCE",
            "answer": "当前知识库没有足够证据回答。",
            "citations": [],
        }
        for attempt in range(2 if self.config.retry_recall else 1):
            started = time.perf_counter()
            query = retrieval_question
            if attempt:
                # 重召回提高候选数并增加关键词视角，不生成凭空扩写的事实。
                keywords = " ".join(dict.fromkeys(tokenize(question)))[:300]
                query = retrieval_question + " " + keywords
            result = self.index.query(
                query,
                top_k=min(100, self.config.candidate_k * (attempt + 1)),
                fusion="rrf",
                filter_expr=filter_expr,
            )
            recalled_at = time.perf_counter()
            ranked = self.reranker.rank(retrieval_question, result["hits"])
            recalled_ids = {h["id"] for h in ranked}
            neighbor_ids = []
            if self.neighbor_fn and ranked:
                extra = [
                    h
                    for h in self.neighbor_fn([h["id"] for h in ranked[:3]])
                    if h["id"] not in recalled_ids
                ]
                neighbor_ids = [h["id"] for h in extra]
                if extra:
                    ranked = self.reranker.rank(retrieval_question, ranked + extra)
            eligible = [h for h in ranked if h["dense_score"] >= self.config.min_dense]
            # 第二轮扩大召回，但压缩最终证据以减少无关上下文干扰。
            selection = (
                replace(self.config, context_k=min(2, self.config.context_k))
                if attempt
                else self.config
            )
            context = select_context(eligible, selection)
            step = {
                "recalled_candidates": result["hits"],
                "recall_seconds": round(recalled_at - started, 4),
                "rerank_context_seconds": round(time.perf_counter() - recalled_at, 4),
                "neighbor_ids": neighbor_ids,
                "attempt": attempt + 1,
                "retrieval_question": query,
                "candidates": ranked,
                "context_ids": [h["id"] for h in context],
                "context_tokens": sum(
                    estimate_tokens(h["text"])
                    + estimate_tokens(str(h.get("source", "")))
                    + 40
                    for h in context
                ),
            }
            trace.append(step)
            if context:
                generation_started = time.perf_counter()
                try:
                    candidate = self.answer_fn(
                        generation_question,
                        context,
                        model=model,
                        base_url=base_url,
                        tone=self.config.tone,
                        max_evidence=self.config.context_k,
                        max_chars=None,
                    )
                except ValueError as exc:
                    step["generation_error"] = str(exc)
                    continue
                step["generation_seconds"] = round(
                    time.perf_counter() - generation_started, 4
                )
                step["answer_status"] = candidate["status"]
                step["generation"] = candidate
                step["discarded_claims"] = len(candidate.get("discarded_claims", []))
                # 重试不能以更差的拒答覆盖已有的有依据回答。
                if candidate["status"] == "ANSWERED":
                    answer = candidate
                    if not candidate.get("discarded_claims"):
                        break
                elif answer["status"] != "ANSWERED":
                    answer = candidate
        if answer["status"] != "ANSWERED" and any(
            "generation_error" in step for step in trace
        ):
            answer = {
                "status": "GENERATION_FAILED",
                "answer": "检索到了候选证据，但回答未通过引用校验，请查看策略记录或重试。",
                "citations": [],
            }
        return {
            "question": question,
            "strategy": asdict(self.config),
            "answer": answer,
            "trace": trace,
        }
