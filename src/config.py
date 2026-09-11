"""
src/config.py
──────────────
Single authoritative config loader for the entire project.
All modules import `cfg` from here — never call os.getenv directly.

Validates required keys at startup and raises a clear ValueError if any are missing.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (two levels up from this file: src/config.py)
_project_root = Path(__file__).parent.parent
_env_path     = _project_root / ".env"
load_dotenv(dotenv_path=_env_path, override=True)


@dataclass(frozen=True)
class Config:
    # ── Gemini ─────────────────────────────────────────────────────────────
    google_api_key:   str
    classifier_model: str
    drafter_model:    str
    judge_model:      str

    # ── Qdrant Cloud ───────────────────────────────────────────────────────
    qdrant_url:        str
    qdrant_api_key:    str
    qdrant_collection: str

    # ── Retrieval ──────────────────────────────────────────────────────────
    top_k_retrieval: int
    embed_model:     str
    embed_dim:       int

    # ── Data Paths (relative to project root) ─────────────────────────────
    raw_csv:       Path
    threads_jsonl: Path
    golden_csv:    Path
    results_dir:   Path

    # ── Data Limits ────────────────────────────────────────────────────────
    max_sample_threads: int


def load_config() -> Config:
    """
    Load and validate configuration from environment variables.
    Raises ValueError with clear instructions if required keys are missing.
    """
    missing: list[str] = []

    def _require(key: str) -> str:
        val = os.getenv(key, "").strip()
        if not val or val.startswith("your_"):
            missing.append(key)
        return val

    def _optional(key: str, default: str) -> str:
        val = os.getenv(key, "").strip()
        return val if val else default

    google_api_key = _require("GOOGLE_API_KEY")
    # Qdrant is local-first: default to a local Docker instance so reviewers
    # need no cloud account. Cloud users set QDRANT_URL + QDRANT_API_KEY.
    qdrant_url     = _optional("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = _optional("QDRANT_API_KEY", "")

    if missing:
        raise ValueError(
            "\n\nMISSING required environment variables: " + str(missing) + "\n"
            "  -> Copy .env.example to .env and fill in your keys.\n"
            "  -> Gemini key : https://aistudio.google.com/app/apikey\n"
            "  -> Qdrant key : https://cloud.qdrant.io\n"
        )

    return Config(
        # Gemini
        google_api_key   = google_api_key,
        classifier_model = _optional("CLASSIFIER_MODEL", "gemini-3.1-flash-lite"),
        drafter_model    = _optional("DRAFTER_MODEL",    "gemini-3.1-flash-lite"),
        # Default judge model is gemini-3.1-flash-lite (same family for benchmark
        # reproduction; can be overridden to models/gemini-2.5-pro or others via env).
        judge_model      = _optional("JUDGE_MODEL",      "gemini-3.1-flash-lite"),

        # Qdrant Cloud
        qdrant_url        = qdrant_url,
        qdrant_api_key    = qdrant_api_key,
        qdrant_collection = _optional("QDRANT_COLLECTION", "amazon_support_history"),

        # Retrieval
        top_k_retrieval = int(_optional("TOP_K_RETRIEVAL", "5")),
        # NOTE: text-embedding-004 was retired (404 on v1beta as of 2026).
        # gemini-embedding-001 is the current replacement; we request
        # output_dimensionality=768 (Matryoshka truncation) to keep the
        # existing Qdrant collection dim unchanged.
        embed_model     = "models/gemini-embedding-001",
        embed_dim       = 768,

        # Paths (always relative to project root, not cwd)
        raw_csv       = _project_root / "data" / "raw"       / "twcs.csv",
        threads_jsonl = _project_root / "data" / "processed" / "amazon_threads.jsonl",
        golden_csv    = _project_root / "golden_set"         / "golden_250.csv",
        results_dir   = _project_root / "results",

        # Limits
        max_sample_threads = int(_optional("MAX_SAMPLE_THREADS", "10000")),
    )


# Singleton — import this everywhere
cfg = load_config()
