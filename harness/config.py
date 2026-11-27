"""Central configuration for models, retrieval thresholds, memory, and safety limits."""

# Gemini model IDs and thinking budgets
# Hydrogen defaults to high thinking; LOW keeps synthesis fast and token-efficient.
# Flash-Lite supports MINIMAL thinking for low-latency scope and query rewriting.
SYNTHESIS_MODEL = "gemini-4-hydrogen"
SYNTHESIS_THINKING = "LOW"

GATE_MODEL = "gemini-3.5-flash-lite"
GATE_THINKING = "MINIMAL"

FALLBACK_SYNTHESIS_MODEL = "gemini-3.8-flash"
FALLBACK_SYNTHESIS_THINKING = "LOW"

# Local vector retrieval and deterministic similarity cutoff
TOP_K = 3
MIN_SCORE_FLOOR = 0.67  # Minimum cosine similarity required to invoke synthesis

# Input and output size guardrails (character counts)
MAX_INPUT_CHARS = 500
MAX_ANSWER_CHARS = 900  # ~150-180 words; enforced in Python to avoid JSON truncation

# Tavily web search limits (pinned to basic depth to keep latency and credit usage low)
WEB_MAX_RESULTS = 3
WEB_SNIPPET_MAX_CHARS = 900
WEB_TIMEOUT_SECONDS = 10

# Bounded session memory settings (constant O(1) context overhead)
MEMORY_MAX_TURNS = 3
MEMORY_ANSWER_MAX_CHARS = 900

# Permitted domain scope for the input classifier
CS_TOPICS = (
    "Data Structures, Algorithms, Computer Networks, Operating Systems, "
    "Database Management Systems (DBMS), Computer Organization & Architecture, "
    "Object-Oriented Programming (OOP), and Theory of Computation / Software Engineering"
)
# --- Phase 4A: Resilience, Timeouts & Model Failover ---
# 60s covers observed gemini-4-hydrogen latency peaks (~48s) while preventing hung sockets
GEMINI_TIMEOUT_MS = 60_000
MAX_TRANSIENT_RETRIES = 2          # Up to 2 retries on transient errors (2s, 4s + jitter)
RETRY_BASE_DELAY_SEC = 2.0
RETRY_JITTER_SEC = 0.25
RETRYABLE_HTTP_CODES = frozenset({408, 429, 500, 502, 503, 504})

# --- Phase 4B: Structured Telemetry & Security Audit Logging ---
TELEMETRY_ENABLED = True
RUNTIME_LOG_PATH = "logs/runtime.jsonl"
AUDIT_LOG_PATH = "logs/audit.jsonl"

# --- Phase 4C: Offline Evaluation & Opt-In Groundedness Judge ---
EVAL_JUDGE_ENABLED = False             # Master switch: OFF by default (enable via --judge)
EVAL_JUDGE_MODEL = "gemini-3.5-flash-lite"
EVAL_JUDGE_THINKING = "MINIMAL"
EVAL_REPORT_PATH = "logs/eval_report.json"