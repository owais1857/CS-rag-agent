# CS Fundamentals RAG Agent

A deterministic, token-efficient Computer Science study assistant built with local FAISS retrieval, a 2-stage sufficiency gate, and a human-in-the-loop (HITL) web search fallback.

## Architecture

- **Phase 1 — Offline Ingestion & Retrieval (`rag/`):** Converts CS lecture PDFs to Markdown (`pymupdf4llm`), cleans headers/HTML noise, performs cross-page chunking with orphan-heading merging, embeds with `BAAI/bge-small-en-v1.5`, and builds a read-only FAISS cosine-similarity index (`top_k=3`).
- **Phase 2 — Gated Local Harness (`harness/`):**
  - Deterministic input gate (length limit, prompt-injection patterns, 0-token greeting fast-path, PII redaction) + lightweight CS scope classifier (`gemini-3.5-flash-lite`, `MINIMAL` thinking).
  - **2-Stage Sufficiency Gate:** Stage 1 short-circuits clear local misses below a `0.67` cosine similarity floor (0 synthesis tokens); Stage 2 uses a Pydantic `sufficient_context` flag on synthesis (`gemini-4-hydrogen`, `LOW` thinking) to catch topics mentioned by name without explanation.
  - Programmatic citation verification and 900-character output capping.
- **Phase 3A — HITL Web Fallback (`harness/web.py`):** On local misses, invokes a UI-independent `confirm_web` callback (fail-closed by default), queries Tavily (`basic` depth, `max_results=3`, session-cached), filters untrusted snippets for invisible Unicode and prompt-injection patterns, isolates context with per-request nonce tags, and strips smuggled links before resolving verified URLs.
- **Phase 3B — Bounded Session Memory *(Planned)*:** Fixed-window `deque(maxlen=3)` storing only verified, capped `(query, answer, source)` turns with a post-synthesis write filter.

## Project Structure

```text
agent/
├── data/
│   ├── rawdata/          # Place your CS PDF notes here (gitignored)
│   └── index/            # Generated cs.faiss & chunks.json (gitignored)
├── rag/
│   ├── ingest.py         # Offline PDF-to-FAISS indexing pipeline
│   └── retriever.py      # Read-only LocalRetriever
├── harness/
│   ├── config.py         # Models, thinking levels, thresholds, and caps
│   ├── llm.py            # Provider-isolated google-genai wrapper
│   ├── gates.py          # Input gate, 0.67 floor gate, citation/link validators
│   ├── web.py            # Tavily search client, cache, and snippet filter
│   └── runner.py         # UI-independent orchestrator
├── main.py               # Interactive CLI entry point
├── .env.example          # Required environment variables template
└── requirements.txt