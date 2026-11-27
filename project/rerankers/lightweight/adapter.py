"""Independent stdlib plugin; only the public contract is imported."""

import math
import re
from collections import Counter
from rag_multidocs.contracts import RankedCandidate


def tokens(text):
    return re.findall(r"[a-z0-9]+|[一-鿿]", text.lower())


def rank(request):
    q = set(tokens(request.query))
    counts = [Counter(tokens(h.text)) for h in request.candidates]
    values = []
    for i, (h, c) in enumerate(zip(request.candidates, counts)):
        bm = sum(
            math.log(
                1
                + (len(counts) - sum(t in other for other in counts) + 0.5)
                / (sum(t in other for other in counts) + 0.5)
            )
            * c[t]
            * 2.2
            / (c[t] + 1.2)
            for t in q
            if c[t]
        )
        values.append(bm)
    max_bm = max(values, default=0) or 1
    scored = []
    for i, h in enumerate(request.candidates):
        semantic = h.semantic_score if h.semantic_score is not None else 1 / (i + 1)
        scored.append((h.candidate_id, 0.8 * max(0, min(1, semantic)) + 0.2 * values[i] / max_bm))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [RankedCandidate(key, i + 1, float(score)) for i, (key, score) in enumerate(scored)]
