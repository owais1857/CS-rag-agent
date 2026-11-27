# CS Fundamentals RAG Agent

A deterministic, token-efficient Computer Science study assistant built with local FAISS retrieval, a 2-stage sufficiency gate, a human-in-the-loop (HITL) web search fallback, bounded session memory, transient retry/failover resilience, privacy-safe runtime/audit logging, and offline evaluation.

## Architecture

- **Phase 1 — Offline Ingestion & Retrieval (`rag/`):** Converts CS lecture PDFs to Markdown (`pymupdf4llm`), cleans headers/HTML noise, performs cross-page chunking with orphan-heading merging, embeds with `BAAI/bge-small-en-v1.5`, and builds a read-only FAISS cosine-similarity index (`top_k=3`).
- **Phase 2 — Gated Local Harness (`harness/`):**
  - Deterministic input gate (length limit, prompt-injection patterns, 0-token greeting fast-path, PII redaction) + lightweight CS scope classifier (`gemini-3.5-flash-lite`, `MINIMAL` thinking).
  - **2-Stage Sufficiency Gate:** Stage 1 short-circuits clear local misses below a `0.67` cosine similarity floor (0 synthesis tokens); Stage 2 uses a Pydantic `sufficient_context` flag on synthesis (`gemini-4-hydrogen`, `LOW` thinking) to catch topics mentioned by name without explanation.
  - Programmatic citation verification and 900-character output capping.
- **Phase 3A — HITL Web Fallback (`harness/web.py`):** On local misses, invokes a UI-independent `confirm_web` callback (fail-closed by default), queries Tavily (`basic` depth, `max_results=3`, session-cached), normalizes untrusted snippets (stripping invisible Unicode, HTML tags, and irregular whitespace) before scanning for prompt-injection patterns, isolates context with per-request nonce tags, and strips smuggled links before resolving verified URLs.
- **Phase 3B — Bounded Session Memory (`harness/memory.py`):** Fixed-window `deque(maxlen=3)` storing only verified, capped `(query, answer, source)` turns after a post-synthesis injection write filter. Feeds history inside a nonce-delimited tag into the Flash-Lite gate to resolve follow-up pronouns in zero extra API calls, with `/clear` reset support in the CLI.
- **Phase 4A — Resilience & Model Failover (`harness/config.py`, `harness/llm.py`):** Configures per-request HTTP timeouts (`60s`), bounded exponential backoff with jitter (`2s`, `4s`) strictly on transient errors (`408`, `429`, `500`, `502`, `503`, `504`, or network timeouts), single-shot failover to `gemini-3.8-flash` when primary synthesis exhausts retries on a transient error, and structured `DONE_LLM_ERROR` reporting without crashing turns.
- **Phase 4B — Runtime Telemetry & Security Audit Logs (`harness/telemetry.py`, `harness/runner.py`):** Emits two append-only, thread-safe JSONL streams under `logs/` (gitignored), correlated by `(session_id, turn_id, query_sha256)` without storing raw user text or PII: `logs/runtime.jsonl` for per-turn latency, scores, tokens, retries/failover, web cache flags, and memory occupancy; `logs/audit.jsonl` for blocks, PII redactions, HITL decisions, filtered web snippets, memory write rejections/clears, failovers, and errors.
- **Phase 4C — Offline Evaluation (`eval/`):** Provides a default 0-token retrieval evaluation (`Hit Rate@k`, `Recall@k`, `MRR`, and similarity floor pass rate) plus an opt-in `--judge` mode (`gemini-3.5-flash-lite` claim-level groundedness) with `--limit` support, telemetry disabled during eval runs, and auto-updated reporting to `logs/eval_report.json`.

## Project Structure

```text
agent/
├── data/
│   ├── rawdata/          # Place your CS PDF notes here (gitignored)
│   └── index/            # Generated cs.faiss & chunks.json (gitignored)
├── rag/
│   ├── ingest.py         # Offline PDF-to-FAISS indexing pipeline
│   └── retriever.py      # Read-only LocalRetriever
├── eval/
│   ├── __init__.py
│   ├── golden_set.json   # Curated CS evaluation queries & expected chunk IDs
│   ├── judge.py          # Opt-in claim-level groundedness judge
│   └── run_eval.py       # Retrieval metrics (default) & opt-in --judge runner
├── harness/
│   ├── config.py         # Models, thinking levels, thresholds, caps, resilience & eval flags
│   ├── llm.py            # Provider-isolated google-genai wrapper with retry & failover
│   ├── gates.py          # Input gate, 0.67 floor gate, citation/link validators
│   ├── web.py            # Tavily search client, cache, and snippet filter
│   ├── memory.py         # Bounded rolling session memory and write filter
│   ├── telemetry.py      # Privacy-safe runtime.jsonl and audit.jsonl writers
│   └── runner.py         # UI-independent orchestrator and telemetry hook
├── logs/                 # Generated runtime.jsonl, audit.jsonl & eval_report.json (gitignored)
├── main.py               # Interactive CLI entry point
├── .env.example          # Required environment variables template
└── requirements.txt
```

## Quick Start

1. **Create a virtual environment and install dependencies:**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   ```

2. **Configure API keys:**
   Copy `.env.example` to `.env` and fill in your `GEMINI_API_KEY` and `TAVILY_API_KEY`.

3. **Build the local vector index:**
   Drop your CS PDF notes into `data/rawdata/` and run:
   ```powershell
   python rag/ingest.py
   ```

4. **Run the agent:**
   ```powershell
   python main.py
   ```

5. **Run offline evaluation (optional):**
   ```powershell
   # Retrieval-only (0 API tokens): Hit Rate@3, Recall@3, MRR, floor pass rate
   python -m eval.run_eval

   # Opt-in groundedness judge on a bounded slice (saves to logs/eval_report.json)
   python -m eval.run_eval --judge --limit 5
   ```