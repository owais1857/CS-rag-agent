"""Deterministic orchestrator for gated local RAG, web fallback, memory, and telemetry."""

import secrets
import time
from typing import Callable
from harness.config import MAX_INPUT_CHARS, TOP_K
from harness.gates import (
    ZERO_USAGE,
    contains_injection,
    redact_pii,
    run_input_gate,
    run_retrieval_gate,
    strip_invisible_unicode,
    validate_output,
    validate_web_output,
)
from harness.llm import GeminiClient, LLMCallError
from harness.memory import SessionMemory
from harness.telemetry import TelemetryLogger, query_fingerprint
from harness.web import WebSearcher
from rag.retriever import LocalRetriever

# Approval callback signature: (pii_redacted_query, miss_reason) -> bool
WebConfirmCallback = Callable[[str, str], bool]


def format_local_context(chunks: list[dict], tag_name: str) -> str:
    """Wrap retrieved local chunks in per-request nonce-delimited tags."""
    blocks = []
    for c in chunks:
        blocks.append(
            f'<{tag_name} chunk_id="{c["chunk_id"]}" source="{c["source"]}" '
            f'page="{c["page"]}" score="{c["score"]}">\n{c["text"]}\n</{tag_name}>'
        )
    return "\n\n".join(blocks)


def format_web_context(web_results: list[dict], tag_name: str) -> str:
    """
    Wrap sanitized web snippets in per-request nonce-delimited tags.

    Raw URLs are omitted from the prompt; verified source IDs ('web_1') are
    resolved back to original URLs programmatically during output validation.
    """
    blocks = []
    for r in web_results:
        blocks.append(
            f'<{tag_name} source_id="{r["source_id"]}" title="{r["title"]}">\n'
            f'{r["snippet"]}\n</{tag_name}>'
        )
    return "\n\n".join(blocks)


def merge_usage(u1: dict, u2: dict) -> dict:
    """Sum token usage counters across multiple LLM calls in a single turn."""
    return {
        k: u1.get(k, 0) + u2.get(k, 0)
        for k in ("input_tokens", "output_tokens", "thinking_tokens", "total_tokens")
    }


def merge_llm_meta(gate_meta: dict, *synth_metas: dict | None) -> dict:
    """Combine gate and synthesis resilience metadata into a single turn-level summary."""
    total_retries = gate_meta.get("retries", 0)
    synthesis_model = None
    fallback_used = False

    for meta in synth_metas:
        if not meta:
            continue
        total_retries += meta.get("retries", 0)
        if meta.get("model"):
            synthesis_model = meta["model"]
        if meta.get("fallback_used"):
            fallback_used = True

    return {
        "gate_model": gate_meta.get("model"),
        "synthesis_model": synthesis_model,
        "retries": total_retries,
        "fallback_used": fallback_used,
    }


def _classify_block_reason(cleaned_query: str) -> str:
    """Categorize why the deterministic input gate blocked a query (without logging raw text)."""
    if not cleaned_query:
        return "empty_input"
    if len(cleaned_query) > MAX_INPUT_CHARS:
        return "max_input_chars"
    if contains_injection(cleaned_query):
        return "prompt_injection"
    return "blocked"


class CSAgentHarness:
    """UI-independent controller coordinating retrieval, gates, synthesis, web fallback, and memory."""

    def __init__(self, telemetry: TelemetryLogger | None = None) -> None:
        self.retriever = LocalRetriever()
        self.llm = GeminiClient()
        self.web = WebSearcher()
        self.memory = SessionMemory()
        self.telemetry = telemetry or TelemetryLogger()

    def clear_memory(self) -> None:
        """Clear all stored conversation turns from session memory."""
        cleared_turns = len(self.memory)
        self.memory.clear()
        self.telemetry.log_audit(
            "MEMORY_CLEARED",
            details={"cleared_turns": cleared_turns},
        )

    def _elapsed_ms(self, start_time: float, human_wait_sec: float = 0.0) -> int:
        """Compute execution latency in milliseconds, excluding interactive user wait time."""
        return round((time.perf_counter() - start_time - human_wait_sec) * 1000)

    def _memory_telemetry(self) -> dict:
        return {
            "memory_turns": len(self.memory),
            "memory_max": self.memory.max_turns,
        }

    def _handle_missing_context(
        self,
        *,
        turn_id: int,
        query_sha256: str,
        query: str,
        abstain_status: str,
        reason: str,
        usage: dict,
        gate_meta: dict,
        local_synth_meta: dict | None,
        start_time: float,
        top_score: float,
        confirm_web: WebConfirmCallback | None,
        rewritten_query: str | None,
    ) -> dict:
        """
        Handle local retrieval misses via an optional human-in-the-loop web fallback.

        Fails closed (abstains without searching) if confirm_web is None or returns False.
        Web results are never written back into the local FAISS index.
        """
        approved = False
        human_wait_sec = 0.0

        if confirm_web is None:
            hitl_decision = "fail_closed"
        else:
            wait_start = time.perf_counter()
            approved = bool(confirm_web(query, reason))
            human_wait_sec = time.perf_counter() - wait_start
            hitl_decision = "approved" if approved else "declined"

        self.telemetry.log_audit(
            "WEB_HITL_DECISION",
            turn_id=turn_id,
            query_sha256=query_sha256,
            details={
                "decision": hitl_decision,
                "miss_stage": abstain_status,
                "top_score": round(top_score, 4),
            },
        )

        if not approved:
            return {
                "status": abstain_status,
                "output": f"[Abstain] Not enough information in your local CS notes. ({reason})",
                "usage": usage,
                "latency_ms": self._elapsed_ms(start_time, human_wait_sec),
                "top_score": top_score,
                "rewritten_query": rewritten_query,
                "llm_meta": merge_llm_meta(gate_meta, local_synth_meta),
                **self._memory_telemetry(),
            }

        # 1. Fetch and pre-filter web snippets (query is already standalone and PII-redacted)
        try:
            web_results, dropped_count, cache_hit = self.web.search(query)
        except Exception as exc:
            self.telemetry.log_audit(
                "WEB_ERROR",
                turn_id=turn_id,
                query_sha256=query_sha256,
                details={"error_type": type(exc).__name__},
            )
            return {
                "status": "DONE_WEB_ERROR",
                "output": f"[Web Error] Search failed: {exc}",
                "usage": usage,
                "latency_ms": self._elapsed_ms(start_time, human_wait_sec),
                "top_score": top_score,
                "rewritten_query": rewritten_query,
                "llm_meta": merge_llm_meta(gate_meta, local_synth_meta),
                **self._memory_telemetry(),
            }

        if dropped_count > 0:
            self.telemetry.log_audit(
                "WEB_SNIPPETS_FILTERED",
                turn_id=turn_id,
                query_sha256=query_sha256,
                details={
                    "dropped_snippets": dropped_count,
                    "remaining_snippets": len(web_results),
                    "all_dropped": len(web_results) == 0,
                    "cache_hit": cache_hit,
                },
            )

        # Short-circuit without synthesis if all returned snippets failed injection checks
        if not web_results and dropped_count > 0:
            return {
                "status": "DONE_ABSTAIN_WEB_UNSAFE",
                "output": (
                    "[Abstain] Web results were filtered out due to unsafe instruction-like "
                    "content. Skipping synthesis."
                ),
                "usage": usage,
                "latency_ms": self._elapsed_ms(start_time, human_wait_sec),
                "top_score": top_score,
                "rewritten_query": rewritten_query,
                "web_meta": {"cache_hit": cache_hit, "dropped_snippets": dropped_count},
                "llm_meta": merge_llm_meta(gate_meta, local_synth_meta),
                **self._memory_telemetry(),
            }

        if not web_results:
            return {
                "status": "DONE_ABSTAIN_WEB_EMPTY",
                "output": "[Abstain] Web search returned no usable snippets for this query.",
                "usage": usage,
                "latency_ms": self._elapsed_ms(start_time, human_wait_sec),
                "top_score": top_score,
                "rewritten_query": rewritten_query,
                "web_meta": {"cache_hit": cache_hit, "dropped_snippets": dropped_count},
                "llm_meta": merge_llm_meta(gate_meta, local_synth_meta),
                **self._memory_telemetry(),
            }

        # 2. Single-pass web synthesis using a fresh per-request nonce delimiter
        web_tag = f"web_context_{secrets.token_hex(4)}"
        web_block = format_web_context(web_results, web_tag)
        try:
            web_answer, web_synth_usage, web_synth_meta = self.llm.synthesize_web_answer(
                web_block, query, tag_name=web_tag
            )
        except LLMCallError as exc:
            err_meta = {
                "model": exc.model,
                "retries": exc.retries,
                "fallback_used": exc.fallback_attempted,
            }
            return {
                "status": "DONE_LLM_ERROR",
                "output": f"[LLM Error] Web synthesis failed after retries: {exc}",
                "usage": usage,
                "latency_ms": self._elapsed_ms(start_time, human_wait_sec),
                "top_score": top_score,
                "rewritten_query": rewritten_query,
                "web_meta": {"cache_hit": cache_hit, "dropped_snippets": dropped_count},
                "llm_meta": {
                    **merge_llm_meta(gate_meta, local_synth_meta, err_meta),
                    "error_code": exc.status_code,
                    "error_stage": "web_synthesis",
                },
                **self._memory_telemetry(),
            }

        total_usage = merge_usage(usage, web_synth_usage)

        # 3. Verify cited web IDs, strip smuggled links, and resolve original URLs
        final_status, formatted_output, bounded_answer = validate_web_output(
            web_answer, web_results
        )

        # 4. Save verified web turn into session memory (passes post-synthesis injection filter)
        if final_status == "DONE_WEB":
            saved = self.memory.add(query, bounded_answer, source="web")
            if not saved:
                self.telemetry.log_audit(
                    "MEMORY_WRITE_REJECTED",
                    turn_id=turn_id,
                    query_sha256=query_sha256,
                    details={"source": "web"},
                )

        return {
            "status": final_status,
            "output": formatted_output,
            "usage": total_usage,
            "latency_ms": self._elapsed_ms(start_time, human_wait_sec),
            "top_score": top_score,
            "rewritten_query": rewritten_query,
            "web_meta": {"cache_hit": cache_hit, "dropped_snippets": dropped_count},
            "llm_meta": merge_llm_meta(gate_meta, local_synth_meta, web_synth_meta),
            **self._memory_telemetry(),
        }

    def _execute_ask(
        self,
        user_query: str,
        confirm_web: WebConfirmCallback | None,
        turn_id: int,
        query_sha256: str,
    ) -> dict:
        """Run the deterministic RAG pipeline for a single turn."""
        start = time.perf_counter()
        cleaned_preview = strip_invisible_unicode(user_query).strip()

        # Format rolling session history inside a per-request nonce tag when non-empty
        history_tag = ""
        history_block = ""
        if not self.memory.is_empty():
            history_tag = f"history_{secrets.token_hex(4)}"
            history_block = self.memory.format_for_prompt(history_tag)

        # 1. Input gate (length, injection, greetings, PII redaction, CS scope + rewrite)
        try:
            gate_status, payload, gate_usage, gate_meta = run_input_gate(
                user_query,
                self.llm,
                history_block=history_block,
                history_tag=history_tag,
            )
        except LLMCallError as exc:
            return {
                "status": "DONE_LLM_ERROR",
                "output": f"[LLM Error] Scope check failed after retries: {exc}",
                "usage": dict(ZERO_USAGE),
                "latency_ms": self._elapsed_ms(start),
                "top_score": None,
                "rewritten_query": None,
                "llm_meta": {
                    "gate_model": exc.model,
                    "synthesis_model": None,
                    "retries": exc.retries,
                    "fallback_used": False,
                    "error_code": exc.status_code,
                    "error_stage": "scope_gate",
                },
                **self._memory_telemetry(),
            }

        if gate_status == "BLOCKED":
            self.telemetry.log_audit(
                "INPUT_BLOCKED",
                turn_id=turn_id,
                query_sha256=query_sha256,
                details={
                    "reason": _classify_block_reason(cleaned_preview),
                    "query_chars": len(cleaned_preview),
                },
            )
        elif gate_status in ("ALLOW", "OUT_OF_SCOPE") and redact_pii(cleaned_preview) != cleaned_preview:
            self.telemetry.log_audit(
                "PII_REDACTED",
                turn_id=turn_id,
                query_sha256=query_sha256,
                details={"gate_status": gate_status},
            )

        if gate_status != "ALLOW":
            return {
                "status": f"DONE_{gate_status}",
                "output": payload,
                "usage": gate_usage,
                "latency_ms": self._elapsed_ms(start),
                "top_score": None,
                "rewritten_query": None,
                "llm_meta": merge_llm_meta(gate_meta),
                **self._memory_telemetry(),
            }

        rewritten_query = (
            payload if payload.lower() != user_query.strip().lower() else None
        )

        # 2. Local FAISS retrieval (read-only)
        chunks = self.retriever.search(payload, top_k=TOP_K)

        # 3. Stage 1 gate: deterministic similarity floor (skips synthesis on clear misses)
        passed_floor, top_score, floor_reason = run_retrieval_gate(chunks)
        if not passed_floor:
            return self._handle_missing_context(
                turn_id=turn_id,
                query_sha256=query_sha256,
                query=payload,
                abstain_status="DONE_ABSTAIN_LOW_SCORE",
                reason=floor_reason,
                usage=gate_usage,
                gate_meta=gate_meta,
                local_synth_meta=None,
                start_time=start,
                top_score=top_score,
                confirm_web=confirm_web,
                rewritten_query=rewritten_query,
            )

        # 4. Grounded local synthesis with per-request nonce tags
        local_tag = f"context_{secrets.token_hex(4)}"
        context_block = format_local_context(chunks, local_tag)
        try:
            rag_answer, synth_usage, synth_meta = self.llm.synthesize_answer(
                context_block, payload, tag_name=local_tag
            )
        except LLMCallError as exc:
            err_meta = {
                "model": exc.model,
                "retries": exc.retries,
                "fallback_used": exc.fallback_attempted,
            }
            return {
                "status": "DONE_LLM_ERROR",
                "output": f"[LLM Error] Local synthesis failed after retries: {exc}",
                "usage": gate_usage,
                "latency_ms": self._elapsed_ms(start),
                "top_score": top_score,
                "rewritten_query": rewritten_query,
                "llm_meta": {
                    **merge_llm_meta(gate_meta, err_meta),
                    "error_code": exc.status_code,
                    "error_stage": "local_synthesis",
                },
                **self._memory_telemetry(),
            }

        total_usage = merge_usage(gate_usage, synth_usage)

        # 5. Stage 2 gate: catches topics mentioned by name but not explained in notes
        if not rag_answer.sufficient_context:
            return self._handle_missing_context(
                turn_id=turn_id,
                query_sha256=query_sha256,
                query=payload,
                abstain_status="DONE_ABSTAIN_INSUFFICIENT",
                reason="Retrieved passages were topically related but lacked the answer.",
                usage=total_usage,
                gate_meta=gate_meta,
                local_synth_meta=synth_meta,
                start_time=start,
                top_score=top_score,
                confirm_web=confirm_web,
                rewritten_query=rewritten_query,
            )

        # 6. Output validation (citation verification, link stripping, character cap)
        final_status, formatted_output, bounded_answer = validate_output(rag_answer, chunks)

        # 7. Save verified local turn into session memory
        if final_status == "DONE_LOCAL":
            saved = self.memory.add(payload, bounded_answer, source="local")
            if not saved:
                self.telemetry.log_audit(
                    "MEMORY_WRITE_REJECTED",
                    turn_id=turn_id,
                    query_sha256=query_sha256,
                    details={"source": "local"},
                )

        return {
            "status": final_status,
            "output": formatted_output,
            "usage": total_usage,
            "latency_ms": self._elapsed_ms(start),
            "top_score": top_score,
            "rewritten_query": rewritten_query,
            "llm_meta": merge_llm_meta(gate_meta, synth_meta),
            **self._memory_telemetry(),
        }

    def ask(
        self,
        user_query: str,
        confirm_web: WebConfirmCallback | None = None,
    ) -> dict:
        """
        Process a user query through input gating, follow-up resolution, local RAG, and web fallback.

        Every turn writes a privacy-safe record to logs/runtime.jsonl; security, HITL,
        failover, and error events are recorded separately in logs/audit.jsonl.
        """
        turn_id = self.telemetry.next_turn_id()
        query_sha256 = query_fingerprint(user_query)

        result = self._execute_ask(
            user_query=user_query,
            confirm_web=confirm_web,
            turn_id=turn_id,
            query_sha256=query_sha256,
        )

        self.telemetry.log_runtime(
            turn_id=turn_id,
            query_sha256=query_sha256,
            query_chars=len(user_query.strip()),
            result=result,
        )

        llm_meta = result.get("llm_meta") or {}
        if result.get("status") == "DONE_LLM_ERROR":
            self.telemetry.log_audit(
                "LLM_ERROR",
                turn_id=turn_id,
                query_sha256=query_sha256,
                details={
                    "stage": llm_meta.get("error_stage"),
                    "error_code": llm_meta.get("error_code"),
                    "retries": llm_meta.get("retries"),
                    "fallback_attempted": llm_meta.get("fallback_used"),
                },
            )
        elif llm_meta.get("fallback_used"):
            self.telemetry.log_audit(
                "LLM_FAILOVER",
                turn_id=turn_id,
                query_sha256=query_sha256,
                details={
                    "synthesis_model": llm_meta.get("synthesis_model"),
                    "retries": llm_meta.get("retries"),
                    "final_status": result.get("status"),
                },
            )

        return result