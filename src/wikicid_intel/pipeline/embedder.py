from __future__ import annotations

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from wikicid_intel.config import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)


def encode_texts(texts: list[str], batch_size: int = 64) -> np.ndarray:
    model = get_model()
    if not texts:
        return np.zeros((0, model.get_sentence_embedding_dimension()), dtype=np.float32)
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 200,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return emb.astype(np.float32)


def cosine_sim_matrix(query_emb: np.ndarray, corpus_emb: np.ndarray) -> np.ndarray:
    """query_emb: (d,) o (1,d); corpus_emb: (n,d) normalizados."""
    q = np.atleast_2d(query_emb).astype(np.float32)
    c = corpus_emb.astype(np.float32)
    return (c @ q.T).ravel()
