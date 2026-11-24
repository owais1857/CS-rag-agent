"""Interactive CLI entry point for the CS Fundamentals RAG Agent."""

from harness.runner import CSAgentHarness


def prompt_web_approval(query: str, reason: str) -> bool:
    """Prompt the user in the terminal before issuing an external web search."""
    print(f"\n[Local Miss] {reason}")
    choice = input(f"Search the web via Tavily for '{query}'? [y/n]: ").strip().lower()
    return choice in {"y", "yes"}


def main() -> None:
    print("Loading local FAISS index, Gemini client, and Tavily web client...")
    agent = CSAgentHarness()
    print("Ready! Type a CS fundamentals question (or 'exit').\n")

    while True:
        query = input("You: ").strip()
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            break

        try:
            result = agent.ask(query, confirm_web=prompt_web_approval)
        except Exception as exc:
            print(f"\n[Error] {exc}\n")
            continue

        print(f"\n{result['output']}")
        u = result["usage"]

        extras = []
        if result["top_score"] is not None:
            extras.append(f"top_score={result['top_score']:.4f}")
        web_meta = result.get("web_meta")
        if web_meta:
            extras.append(f"web_cache={web_meta['cache_hit']}")
            if web_meta["dropped_snippets"]:
                extras.append(f"dropped={web_meta['dropped_snippets']}")

        extra_str = f" | {' | '.join(extras)}" if extras else ""
        print(
            f"\n--- [{result['status']}{extra_str} | {result['latency_ms']}ms | "
            f"tokens in={u['input_tokens']}, think={u['thinking_tokens']}, "
            f"out={u['output_tokens']}, total={u['total_tokens']}] ---\n"
        )


if __name__ == "__main__":
    main()