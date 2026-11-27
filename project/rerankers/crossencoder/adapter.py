import math
from rag_multidocs.contracts import RankedCandidate

_cached = None
_identity = None
_loads = 0


def diagnostics():
    return {"model_loads": _loads, "device": "cpu", "batch_size": 8, "max_length": 512}


def rank(request):
    global _cached, _identity, _loads
    from sentence_transformers import CrossEncoder
    from torch import nn

    # Explicit local-only by default: optional model is never downloaded implicitly.
    identity = (request.config.get("model", "BAAI/bge-reranker-base"), request.config.get("local_files_only", True))
    if _cached is None or _identity != identity:
        _cached = CrossEncoder(identity[0], device="cpu", max_length=512, activation_fn=nn.Identity(), local_files_only=identity[1])
        _identity = identity
        _loads += 1
    model = _cached
    scores = model.predict([(request.query, h.text) for h in request.candidates], batch_size=8)
    scored = [
        (h.candidate_id, 1 / (1 + math.exp(-max(-60, min(60, float(s))))))
        for h, s in zip(request.candidates, scores)
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [RankedCandidate(key, i + 1, score) for i, (key, score) in enumerate(scored)]


rank.diagnostics = diagnostics
