"""Central configuration for models, retrieval thresholds, and safety limits."""

# Gemini model IDs and thinking budgets
# Hydrogen defaults to high thinking; LOW keeps synthesis fast and token-efficient.
# Flash-Lite supports MINIMAL thinking for low-latency scope classification.
SYNTHESIS_MODEL = "gemini-4-hydrogen"
SYNTHESIS_THINKING = "LOW"

GATE_MODEL = "gemini-3.5-flash-lite"
GATE_THINKING = "MINIMAL"

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

# Permitted domain scope for the input classifier
CS_TOPICS = (
    "Data Structures, Algorithms, Computer Networks, Operating Systems, "
    "Database Management Systems (DBMS), Computer Organization & Architecture, "
    "Object-Oriented Programming (OOP), and Theory of Computation / Software Engineering"
)