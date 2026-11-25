"""Bounded rolling session memory with post-synthesis injection filtering."""

from collections import deque
from harness.config import MEMORY_ANSWER_MAX_CHARS, MEMORY_MAX_TURNS
from harness.gates import (
    contains_injection,
    strip_invisible_unicode,
    truncate_cleanly,
)


class SessionMemory:
    """
    Fixed-window conversational memory storing only verified (query, answer, source) turns.

    Raw retrieved chunks, raw web snippets, greetings, and abstentions are never stored,
    keeping multi-turn token usage bounded to a small constant ceiling.
    """

    def __init__(self, max_turns: int = MEMORY_MAX_TURNS) -> None:
        self.max_turns = max_turns
        self._turns: deque[dict[str, str]] = deque(maxlen=max_turns)

    def __len__(self) -> int:
        return len(self._turns)

    def is_empty(self) -> bool:
        return len(self._turns) == 0

    def clear(self) -> None:
        """Reset all stored conversation turns."""
        self._turns.clear()

    def add(self, query: str, bounded_answer: str, source: str) -> bool:
        """
        Validate and record a completed turn into the rolling window.

        Skips saving if either the query or the synthesized answer contains
        prompt-injection phrasing, preventing poisoned web turns from persisting.

        Returns:
            True if the turn was saved, False if skipped.
        """
        clean_query = strip_invisible_unicode(query).strip()
        clean_answer = strip_invisible_unicode(bounded_answer).strip()

        if not clean_query or not clean_answer:
            return False

        if contains_injection(clean_query) or contains_injection(clean_answer):
            return False

        self._turns.append(
            {
                "query": clean_query,
                "answer": truncate_cleanly(clean_answer, MEMORY_ANSWER_MAX_CHARS),
                "source": source,
            }
        )
        return True

    def format_for_prompt(self, tag_name: str) -> str:
        """Format stored turns inside a per-request nonce tag for Turn 2+ query rewriting."""
        if not self._turns:
            return ""

        lines = [f"<{tag_name}>"]
        for idx, turn in enumerate(self._turns, start=1):
            lines.append(
                f"Turn {idx} [{turn['source']}]:\n"
                f"User: {turn['query']}\n"
                f"Tutor: {turn['answer']}"
            )
        lines.append(f"</{tag_name}>")
        return "\n".join(lines)