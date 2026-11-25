"""Deterministic input gates, retrieval thresholds, and output validators."""

import re
from harness.config import (
    MAX_ANSWER_CHARS,
    MAX_INPUT_CHARS,
    MIN_SCORE_FLOOR,
)
from harness.llm import GeminiClient, RAGAnswer

# Fast-path pattern for conversational greetings handled without an LLM call
GREETING_PATTERN = re.compile(
    r"^(hi|hello|hey|yo|good\s+(morning|afternoon|evening)|thanks|thank\s+you|bye|goodbye)[!.\s]*$",
    re.IGNORECASE,
)

# Shared prompt-injection and delimiter-spoofing signatures
INJECTION_PATTERNS = [
    re.compile(r"(ignore|disregard)\s+(all\s+|any\s+)?(previous|prior|above)\s+instructions", re.IGNORECASE),
    re.compile(r"forget\s+(everything|all\s+instructions)\s+(above|before)", re.IGNORECASE),
    re.compile(r"reveal\s+(your\s+)?(system\s+prompt|hidden\s+instructions)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(in\s+)?(developer|jailbreak|dan)\s+mode", re.IGNORECASE),
    re.compile(r"system\s+prompt\s+override", re.IGNORECASE),
    re.compile(r"</?\s*(system|instruction|web_context|context|history)\b", re.IGNORECASE),
]

# Zero-width spaces and bidirectional override characters used in hidden-text attacks
INVISIBLE_UNICODE_PATTERN = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff]")

# Link and image patterns stripped from generated text before rendering
MD_IMAGE_PATTERN = re.compile(r"!\[[^\]]*\]\([^)]*\)")
MD_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\((?:https?://|www\.)[^)]+\)", re.IGNORECASE)
RAW_URL_PATTERN = re.compile(r"https?://\S+", re.IGNORECASE)

# Basic PII patterns redacted before any outbound API call
EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
PHONE_PATTERN = re.compile(r"\b(?:\+?\d[\d\s-]{8,14}\d)\b")

ZERO_USAGE = {
    "input_tokens": 0,
    "output_tokens": 0,
    "thinking_tokens": 0,
    "total_tokens": 0,
}


def strip_invisible_unicode(text: str) -> str:
    """Remove zero-width and bidirectional control characters."""
    return INVISIBLE_UNICODE_PATTERN.sub("", text)


def contains_injection(text: str) -> bool:
    """Return True if text matches known prompt-injection or tag-spoofing patterns."""
    normalized = strip_invisible_unicode(text)
    return any(pattern.search(normalized) for pattern in INJECTION_PATTERNS)


def strip_smuggled_links(text: str) -> str:
    """Remove raw URLs, Markdown links, and image beacons from generated answer text."""
    text = MD_IMAGE_PATTERN.sub("", text)
    text = MD_LINK_PATTERN.sub(r"\1", text)
    text = RAW_URL_PATTERN.sub("", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def redact_pii(text: str) -> str:
    """Replace email addresses and phone numbers with placeholder tokens."""
    text = EMAIL_PATTERN.sub("[REDACTED_EMAIL]", text)
    text = PHONE_PATTERN.sub("[REDACTED_PHONE]", text)
    return text


def run_input_gate(
    query: str,
    llm: GeminiClient,
    history_block: str = "",
    history_tag: str = "",
) -> tuple[str, str, dict]:
    """
    Validate, classify, and optionally rewrite a user query before retrieval.

    Deterministic checks (empty input, length, injection, greetings) run first at
    zero token cost; remaining queries are PII-redacted, checked for CS scope,
    and resolved into a standalone query when session history is present.

    Returns:
        (status, payload, usage) where status is one of
        'BLOCKED', 'GREETING', 'OUT_OF_SCOPE', or 'ALLOW'.
    """
    cleaned = strip_invisible_unicode(query).strip()

    if not cleaned:
        return "BLOCKED", "Please enter a CS fundamentals question.", dict(ZERO_USAGE)

    if len(cleaned) > MAX_INPUT_CHARS:
        return (
            "BLOCKED",
            f"Query too long ({len(cleaned)} chars). Keep questions under {MAX_INPUT_CHARS} characters.",
            dict(ZERO_USAGE),
        )

    if contains_injection(cleaned):
        return (
            "BLOCKED",
            "I cannot override my instructions. Please ask a Computer Science fundamentals question.",
            dict(ZERO_USAGE),
        )

    if GREETING_PATTERN.match(cleaned):
        return (
            "GREETING",
            "Hello! Ask me any Computer Science fundamentals question and I'll check your local notes.",
            dict(ZERO_USAGE),
        )

    cleaned = redact_pii(cleaned)

    verdict, usage = llm.check_scope(
        cleaned,
        history_block=history_block,
        history_tag=history_tag,
    )
    if not verdict.in_scope:
        return (
            "OUT_OF_SCOPE",
            "Out of scope: I only answer Computer Science fundamentals questions.",
            usage,
        )

    resolved = strip_invisible_unicode(verdict.standalone_query).strip() or cleaned
    resolved = redact_pii(resolved)
    return "ALLOW", resolved, usage


def run_retrieval_gate(
    chunks: list[dict],
    floor: float = MIN_SCORE_FLOOR,
) -> tuple[bool, float, str]:
    """
    Verify that the top local retrieval match clears the minimum cosine similarity floor.

    Checks only chunks[0] so that once a match qualifies, all top-k chunks remain
    available to keep multi-chunk tables and explanations intact.
    """
    if not chunks:
        return False, 0.0, "No chunks retrieved from the local index."

    top_score = float(chunks[0]["score"])
    if top_score < floor:
        return (
            False,
            top_score,
            f"Top local match score ({top_score:.4f}) is below floor ({floor:.2f}).",
        )

    return True, top_score, "Passed local similarity floor."


def truncate_cleanly(text: str, max_chars: int = MAX_ANSWER_CHARS) -> str:
    """Truncate text to max_chars at the nearest clean sentence or word boundary."""
    text = text.strip()
    if len(text) <= max_chars:
        return text

    clipped = text[: max_chars - 3]
    last_period = clipped.rfind(". ")
    if last_period > max_chars // 2:
        return clipped[: last_period + 1]

    last_space = clipped.rfind(" ")
    if last_space > 0:
        return clipped[:last_space] + "..."
    return clipped + "..."


def validate_output(
    rag_answer: RAGAnswer,
    retrieved_chunks: list[dict],
) -> tuple[str, str, str]:
    """
    Validate a local RAG answer against retrieved chunks and format citations.

    Rejects unverified chunk IDs, strips raw links, enforces MAX_ANSWER_CHARS,
    and returns (status, formatted_output, bounded_answer).
    """
    valid_map = {c["chunk_id"]: c for c in retrieved_chunks}

    if not rag_answer.sufficient_context or not rag_answer.answer.strip():
        msg = "[Abstain] Your local CS notes do not contain enough information to answer this."
        return "DONE_ABSTAIN_INSUFFICIENT", msg, ""

    verified_ids = [cid for cid in rag_answer.cited_ids if cid in valid_map]
    if not verified_ids:
        msg = "[Abstain] Could not verify citations against your local CS notes."
        return "DONE_ABSTAIN_UNVERIFIED", msg, ""

    sanitized_answer = strip_smuggled_links(rag_answer.answer)
    bounded_answer = truncate_cleanly(sanitized_answer, MAX_ANSWER_CHARS)
    citation_labels = ", ".join(
        f"{cid} ({valid_map[cid]['source']} p.{valid_map[cid]['page']})"
        for cid in dict.fromkeys(verified_ids)
    )
    formatted_output = f"[Local] {bounded_answer}\n\nSources: {citation_labels}"
    return "DONE_LOCAL", formatted_output, bounded_answer


def validate_web_output(
    rag_answer: RAGAnswer,
    web_results: list[dict],
) -> tuple[str, str, str]:
    """
    Validate a web-synthesized answer and resolve verified source IDs to original URLs.

    Strips any links emitted inside the answer text so only verified URLs from the
    search client are displayed. Returns (status, formatted_output, bounded_answer).
    """
    valid_map = {r["source_id"]: r for r in web_results}

    if not rag_answer.sufficient_context or not rag_answer.answer.strip():
        msg = "[Abstain] Web snippets did not contain enough verified information to answer this."
        return "DONE_ABSTAIN_WEB_INSUFFICIENT", msg, ""

    verified_ids = [sid for sid in rag_answer.cited_ids if sid in valid_map]
    if not verified_ids:
        msg = "[Abstain] Could not verify web citations against fetched search results."
        return "DONE_ABSTAIN_WEB_UNVERIFIED", msg, ""

    sanitized_answer = strip_smuggled_links(rag_answer.answer)
    bounded_answer = truncate_cleanly(sanitized_answer, MAX_ANSWER_CHARS)

    source_lines = "\n".join(
        f"- [{sid}] {valid_map[sid]['title']} — {valid_map[sid]['url']}"
        for sid in dict.fromkeys(verified_ids)
    )
    formatted_output = f"[Web] {bounded_answer}\n\nSources:\n{source_lines}"
    return "DONE_WEB", formatted_output, bounded_answer