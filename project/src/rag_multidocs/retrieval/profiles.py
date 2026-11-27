"""Shared retrieval policy presets."""
from dataclasses import replace

from .controller import RetrievalPolicy


def recommended_policies():
    stable = RetrievalPolicy.baseline(use_neighbors=True, atomic_generation=True, structured_table_answers=True, allow_calculation=True)
    lightweight = replace(stable, reranker='lightweight')
    quality = replace(stable, reranker='crossencoder')
    experimental = RetrievalPolicy(atomic_generation=True, structured_table_answers=True,
                                   allow_calculation=True, max_generation_calls=4)
    return {'stable': stable, 'lightweight': lightweight, 'quality': quality, 'experimental': experimental}
