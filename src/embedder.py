"""
src/embedder.py
───────────────
Gemini gemini-embedding-001 wrapper using the new google-genai SDK.
Handles batching (100 texts/call) and exponential-backoff retries.
"""

import time
from typing import List

from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential

from config import cfg

# Single authenticated client — reused across all calls
_client = genai.Client(api_key=cfg.google_api_key)

_BATCH_SIZE = 100   # Gemini embedding API batch limit


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=30))
def _embed_batch(texts: List[str], task_type: str) -> List[List[float]]:
    """Embed a single batch via the new google-genai SDK."""
    response = _client.models.embed_content(
        model   = cfg.embed_model,
        contents = texts,
        config  = types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=cfg.embed_dim,
        ),
    )
    return [e.values for e in response.embeddings]


def embed_texts(
    texts: List[str],
    task_type: str = "RETRIEVAL_DOCUMENT",
    show_progress: bool = False,
) -> List[List[float]]:
    """
    Embed a list of strings in batches of 100.

    Args:
        texts        : strings to embed
        task_type    : "RETRIEVAL_DOCUMENT" (indexing) or "RETRIEVAL_QUERY" (search)
        show_progress: show tqdm progress bar

    Returns:
        List of 768-dim float vectors, one per input text
    """
    batches = [texts[i: i + _BATCH_SIZE] for i in range(0, len(texts), _BATCH_SIZE)]

    if show_progress:
        from tqdm import tqdm
        batches = tqdm(batches, desc="Embedding")

    embeddings: List[List[float]] = []
    for batch in batches:
        embeddings.extend(_embed_batch(batch, task_type))
        time.sleep(0.05)

    return embeddings


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=30))
def embed_query(text: str) -> List[float]:
    """Embed a single query string (RETRIEVAL_QUERY for better recall)."""
    response = _client.models.embed_content(
        model    = cfg.embed_model,
        contents = [text],
        config   = types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=cfg.embed_dim,
        ),
    )
    return response.embeddings[0].values


# ── Smoke-test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    vecs = embed_texts(["Where is my order?", "I need a refund."], show_progress=True)
    print(f"OK — embedded {len(vecs)} texts, dim={len(vecs[0])}")
