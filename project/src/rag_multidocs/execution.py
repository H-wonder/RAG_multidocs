"""Cooperative request deadlines and observable Provider accounting.

Network clients cap each attempt to remaining time. Arbitrary synchronous
third-party Python calls and trusted in-process plugins cannot be forcibly killed.
"""

from __future__ import annotations

import contextvars
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

_current = contextvars.ContextVar("rag_request_budget", default=None)


@dataclass
class RequestBudget:
    seconds: float
    started: float = field(default_factory=time.monotonic)
    counters: dict = field(
        default_factory=lambda: {
            "embedding_calls": 0,
            "embedding_items": 0,
            "rewrite_calls": 0,
            "generation_calls": 0,
            "transport_attempts": 0,
            "transport_retries": 0,
        }
    )

    def remaining(self):
        left = self.seconds - (time.monotonic() - self.started)
        if left <= 0:
            raise TimeoutError("REQUEST_DEADLINE_EXCEEDED")
        return left


def current_budget():
    return _current.get()


def remaining(default):
    budget = current_budget()
    return min(default, budget.remaining()) if budget else default


def record(name, count=1):
    budget = current_budget()
    if budget:
        budget.counters[name] = budget.counters.get(name, 0) + count


@contextmanager
def request_scope(seconds=120):
    if seconds <= 0:
        raise ValueError("request_seconds must be positive")
    inherited = current_budget()
    if inherited:
        inherited.remaining()
        yield inherited
        return
    budget = RequestBudget(seconds)
    token = _current.set(budget)
    try:
        yield budget
    finally:
        _current.reset(token)
