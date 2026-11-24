"""Bounded Tavily web search client with session caching and snippet filtering."""

import os
import re
from dotenv import load_dotenv
from tavily import TavilyClient
from harness.config import (
    WEB_MAX_RESULTS,
    WEB_SNIPPET_MAX_CHARS,
    WEB_TIMEOUT_SECONDS,
)
from harness.gates import (
    contains_injection,
    strip_invisible_unicode,
    truncate_cleanly,
)

load_dotenv()

HTML_TAG_PATTERN = re.compile(r"<[^>]+>")


def _sanitize_snippet(text: str) -> str:
    """Strip invisible Unicode, residual HTML tags, and cap snippet length."""
    text = strip_invisible_unicode(text)
    text = HTML_TAG_PATTERN.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return truncate_cleanly(text, WEB_SNIPPET_MAX_CHARS)


class WebSearcher:
    """Read-only web search wrapper that caches queries and filters unsafe snippets."""

    def __init__(self) -> None:
        api_key = (os.getenv("TAVILY_API_KEY") or "").strip()
        if not api_key or api_key == "paste_your_tavily_key_here":
            raise RuntimeError(
                "Missing TAVILY_API_KEY. Set your Tavily API key in .env first."
            )
        self.client = TavilyClient(api_key=api_key)
        self._cache: dict[str, tuple[list[dict], int]] = {}

    def search(self, query: str) -> tuple[list[dict], int, bool]:
        """
        Fetch and sanitize bounded web snippets for a query.

        Returns:
            (safe_results, dropped_count, cache_hit)
        """
        cache_key = query.strip().lower()
        if cache_key in self._cache:
            cached_results, cached_dropped = self._cache[cache_key]
            return cached_results, cached_dropped, True

        # Explicitly pin basic search depth and disable raw HTML/auto-upgrades
        # to keep credit consumption, payload size, and cloaking exposure minimal.
        response = self.client.search(
            query=query,
            search_depth="basic",
            topic="general",
            max_results=WEB_MAX_RESULTS,
            chunks_per_source=1,
            include_answer=False,
            include_raw_content=False,
            include_images=False,
            auto_parameters=False,
            timeout=WEB_TIMEOUT_SECONDS,
        )

        raw_results = response.get("results", []) if isinstance(response, dict) else []
        safe_results: list[dict] = []
        dropped_count = 0

        for item in raw_results:
            raw_title = strip_invisible_unicode(str(item.get("title") or "Untitled")).strip()
            raw_content = str(item.get("content") or "")
            url = str(item.get("url") or "").strip()

            if not raw_content.strip() or not url:
                continue

            # Scan full untrusted title and content before truncation to drop
            # snippets carrying prompt-injection payloads anywhere in the text.
            if contains_injection(raw_title) or contains_injection(raw_content):
                dropped_count += 1
                continue

            snippet = _sanitize_snippet(raw_content)
            if not snippet:
                continue

            safe_results.append(
                {
                    "source_id": f"web_{len(safe_results) + 1}",
                    "title": truncate_cleanly(raw_title, 120),
                    "url": url,
                    "snippet": snippet,
                }
            )

        self._cache[cache_key] = (safe_results, dropped_count)
        return safe_results, dropped_count, False