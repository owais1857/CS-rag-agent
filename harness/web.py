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


def _normalize_web_text(text: str) -> str:
    """Strip invisible Unicode, HTML tags/comments, and irregular whitespace."""
    text = strip_invisible_unicode(text)
    text = HTML_TAG_PATTERN.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


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
            # Normalize into final plain-text form before scanning so obfuscated
            # tags or irregular whitespace cannot bypass injection detection.
            norm_title = _normalize_web_text(str(item.get("title") or "")) or "Untitled"
            norm_content = _normalize_web_text(str(item.get("content") or ""))
            url = str(item.get("url") or "").strip()

            if not norm_content or not url:
                continue

            # Scan full normalized title and content before truncation
            if contains_injection(norm_title) or contains_injection(norm_content):
                dropped_count += 1
                continue

            safe_results.append(
                {
                    "source_id": f"web_{len(safe_results) + 1}",
                    "title": truncate_cleanly(norm_title, 120),
                    "url": url,
                    "snippet": truncate_cleanly(norm_content, WEB_SNIPPET_MAX_CHARS),
                }
            )

        self._cache[cache_key] = (safe_results, dropped_count)
        return safe_results, dropped_count, False