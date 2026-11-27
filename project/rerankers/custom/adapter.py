"""Architecture example: prefer passages containing a configured term.
Not a generally better retrieval algorithm and not a teammate Review artifact.
"""

from rag_multidocs.contracts import RankedCandidate


def rank(request):
    term = request.config.get("preferred_term", "表格")
    ordered = sorted(request.candidates, key=lambda h: term in h.text, reverse=True)
    return [
        RankedCandidate(h.candidate_id, i + 1, 1 - i / max(1, len(ordered)))
        for i, h in enumerate(ordered)
    ]
