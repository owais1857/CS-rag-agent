import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from harness.config import EVAL_JUDGE_ENABLED, EVAL_REPORT_PATH, MIN_SCORE_FLOOR, TOP_K
from rag.retriever import LocalRetriever

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = Path(__file__).resolve().parent / "golden_set.json"
DEFAULT_REPORT_PATH = PROJECT_ROOT / EVAL_REPORT_PATH


def compute_retrieval_metrics(retrieved_ids: list[str], expected_ids: list[str]) -> dict:
    """Compute Hit Rate@k, Recall@k, and Reciprocal Rank for one query."""
    expected = set(expected_ids)
    if not expected:
        raise ValueError("expected_chunk_ids must contain at least one chunk_id.")

    hit = 1.0 if any(cid in expected for cid in retrieved_ids) else 0.0
    recall = len(expected.intersection(retrieved_ids)) / len(expected)
    rr = 0.0
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in expected:
            rr = 1.0 / rank
            break

    return {"hit": hit, "recall": recall, "rr": rr}


def _extract_local_answer(formatted_output: str) -> str:
    """Strip the '[Local]' prefix and 'Sources:' footer to isolate answer text for judging."""
    text = formatted_output.strip()
    if text.startswith("[Local]"):
        text = text[len("[Local]"):].strip()
    if "\n\nSources:" in text:
        text = text.split("\n\nSources:", 1)[0].strip()
    return text


def run_evaluation(
    dataset_path: Path,
    k: int,
    use_judge: bool,
    limit: int | None = None,
) -> dict:
    cases = json.loads(dataset_path.read_text(encoding="utf-8"))
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be a positive integer.")
        cases = cases[:limit]
    if not cases:
        raise ValueError(f"No evaluation cases found in {dataset_path}")

    harness = None
    judge = None
    if use_judge:
        from eval.judge import GroundednessJudge
        from harness.runner import CSAgentHarness
        from harness.telemetry import TelemetryLogger

        # Disable runtime.jsonl writes so eval traffic does not mix with live logs
        harness = CSAgentHarness(telemetry=TelemetryLogger(enabled=False))
        retriever = harness.retriever
        judge = GroundednessJudge()
    else:
        retriever = LocalRetriever()

    rows = []
    for case in cases:
        query = case["query"]
        expected_ids = case["expected_chunk_ids"]

        chunks = retriever.search(query, top_k=k)
        retrieved_ids = [c["chunk_id"] for c in chunks]
        top_score = float(chunks[0]["score"]) if chunks else 0.0
        ret = compute_retrieval_metrics(retrieved_ids, expected_ids)

        row = {
            "id": case["id"],
            "query": query,
            "top_score": round(top_score, 4),
            "passed_floor": top_score >= MIN_SCORE_FLOOR,
            "retrieved_ids": retrieved_ids,
            "expected_ids": expected_ids,
            **ret,
            "status": None,
            "groundedness": None,
        }

        if use_judge and harness is not None and judge is not None:
            harness.clear_memory()
            result = harness.ask(query, confirm_web=None)
            row["status"] = result["status"]

            if result["status"] == "DONE_LOCAL":
                effective_query = result.get("rewritten_query") or query
                synth_chunks = (
                    chunks
                    if effective_query == query
                    else retriever.search(effective_query, top_k=k)
                )
                answer_text = _extract_local_answer(result["output"])
                row["groundedness"] = judge.evaluate(effective_query, answer_text, synth_chunks)

        rows.append(row)

    n = len(rows)
    summary = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "k": k,
        "cases": n,
        "hit_rate_at_k": round(sum(r["hit"] for r in rows) / n, 4),
        "recall_at_k": round(sum(r["recall"] for r in rows) / n, 4),
        "mrr": round(sum(r["rr"] for r in rows) / n, 4),
        "floor_pass_rate": round(sum(1 for r in rows if r["passed_floor"]) / n, 4),
        "judge_enabled": use_judge,
    }

    if use_judge:
        judged = [r["groundedness"] for r in rows if r["groundedness"] is not None]
        summary["answered_local"] = len(judged)
        summary["mean_groundedness"] = (
            round(sum(g["score"] for g in judged) / len(judged), 4) if judged else None
        )
        summary["total_judge_tokens"] = sum(g["judge_tokens"] for g in judged)

    return {"summary": summary, "results": rows}


def _print_report(report: dict) -> None:
    s = report["summary"]
    k = s["k"]
    print(f"\n=== CS RAG Evaluation ({s['cases']} cases | k={k} | judge={'ON' if s['judge_enabled'] else 'OFF'}) ===\n")

    for r in report["results"]:
        floor_flag = "OK" if r["passed_floor"] else "BELOW_FLOOR"
        line = (
            f"[{r['id']}] hit={int(r['hit'])} | recall={r['recall']:.2f} | "
            f"rr={r['rr']:.2f} | top_score={r['top_score']:.4f} ({floor_flag}) | "
            f"got={r['retrieved_ids']}"
        )
        if s["judge_enabled"]:
            g = r["groundedness"]
            if g is not None:
                line += f" | grounded={g['score']:.2f} ({g['supported_claims']}/{g['total_claims']})"
            else:
                line += f" | grounded=SKIPPED ({r['status']})"
        print(line)

        if s["judge_enabled"] and r["groundedness"] and r["groundedness"]["unsupported_claims"]:
            for u in r["groundedness"]["unsupported_claims"]:
                print(f"    ! Unsupported: {u['claim']}  -->  {u['reason']}")

    print("\n--- Summary ---")
    print(f"Hit Rate@{k}   : {s['hit_rate_at_k']:.4f}")
    print(f"Recall@{k}     : {s['recall_at_k']:.4f}")
    print(f"MRR           : {s['mrr']:.4f}")
    print(f"Floor Pass    : {s['floor_pass_rate']:.4f} (floor={MIN_SCORE_FLOOR})")
    if s["judge_enabled"]:
        mg = s["mean_groundedness"]
        mg_str = f"{mg:.4f}" if mg is not None else "N/A"
        print(f"Groundedness  : {mg_str} (judged {s['answered_local']}/{s['cases']} DONE_LOCAL)")
        print(f"Judge Tokens  : {s['total_judge_tokens']}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run offline RAG evaluation for CS RAG Agent.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--k", type=int, default=TOP_K)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of cases to run from the dataset.",
    )
    parser.add_argument(
        "--judge",
        action=argparse.BooleanOptionalAction,
        default=EVAL_JUDGE_ENABLED,
        help="Enable synthesis + LLM-as-a-judge groundedness (default: off).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Path to write JSON eval report (default: logs/eval_report.json).",
    )
    args = parser.parse_args()

    report = run_evaluation(args.dataset, k=args.k, use_judge=args.judge, limit=args.limit)
    _print_report(report)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved report to {args.out}")


if __name__ == "__main__":
    main()