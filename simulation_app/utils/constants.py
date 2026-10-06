"""Shared constants for the simulation app.

Canonical topic stop-word set (from the CLAUDE.md "Stop-word pattern").
Several modules still carry local copy-pasted copies of this set; new code
should import ``TOPIC_STOP_WORDS`` from here instead of re-declaring it.
"""
from __future__ import annotations

from typing import FrozenSet

TOPIC_STOP_WORDS: FrozenSet[str] = frozenset({
    'this', 'that', 'about', 'what', 'your', 'please', 'describe',
    'explain', 'question', 'context', 'study', 'topic', 'condition',
    'think', 'feel', 'have', 'some', 'with', 'from', 'very', 'really',
})

__all__ = ["TOPIC_STOP_WORDS"]
