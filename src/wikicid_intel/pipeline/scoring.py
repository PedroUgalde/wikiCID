from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import requests

from wikicid_intel.config import (
    IMPACT_ANCHORS_ES,
    INDUSTRY_ANCHORS_ES,
    WEIGHT_IMPACT,
    WEIGHT_INDUSTRY,
    WEIGHT_MATURITY,
)
from wikicid_intel.pipeline.embedder import encode_texts


def _anchor_max_similarity(text_emb: np.ndarray, anchor_embs: np.ndarray) -> float:
    """text_emb (d,), anchor_embs (k,d) normalizados -> escalar en [0,1]."""
    sims = anchor_embs @ text_emb
    return float(np.clip(np.max(sims), 0.0, 1.0))


def website_reachable(host: str, timeout: float = 2.5) -> bool:
    if not host:
        return False
    for scheme in ("https", "http"):
        url = f"{scheme}://{host}"
        try:
            r = requests.head(
                url,
                timeout=timeout,
                allow_redirects=True,
                headers={"User-Agent": "WikiCIDIntel/1.0"},
            )
            if r.status_code < 400:
                return True
        except requests.RequestException:
            continue
    return False


def batch_website_reachable(hosts: list[str], max_workers: int = 12) -> dict[str, bool]:
    """Una petición por host único; desactivar con WIKICID_PROBE_WEB=0."""
    if os.environ.get("WIKICID_PROBE_WEB", "1").lower() in ("0", "false", "no"):
        return {h: False for h in set(hosts) if h}
    unique = sorted({h for h in hosts if h})
    out: dict[str, bool] = {h: False for h in unique}

    def check(h: str) -> tuple[str, bool]:
        return h, website_reachable(h)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(check, h) for h in unique]
        for fut in as_completed(futures):
            h, ok = fut.result()
            out[h] = ok
    return out


def maturity_score(
    description: str,
    web_host: str,
    extra_filled_fields: int,
    url_ok: bool = False,
) -> float:
    """
    0-100: sitio alcanzable, descripción suficiente, campos de negocio completos.
    """
    desc = (description or "").strip()
    nchars = len(desc)
    words = len(desc.split())

    url_pts = 0.0
    if web_host:
        url_pts += 25.0
        if url_ok:
            url_pts += 25.0

    desc_pts = 0.0
    if nchars >= 200 and words >= 30:
        desc_pts = 35.0
    elif nchars >= 120 and words >= 18:
        desc_pts = 28.0
    elif nchars >= 60 and words >= 10:
        desc_pts = 18.0
    elif nchars >= 30:
        desc_pts = 10.0

    # Hasta 15 puntos por riqueza de fila (mercados, ingresos, fondeo, etc.)
    richness = min(15.0, float(extra_filled_fields) * 3.0)

    raw = url_pts + desc_pts + richness
    return float(min(100.0, raw))


def compute_radar_vectors(
    embedding_texts: list[str],
    descriptions: list[str],
    web_hosts: list[str],
    extra_counts: list[int],
    industry_embs: np.ndarray,
    impact_embs: np.ndarray,
    url_ok_by_host: dict[str, bool] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Retorna tupla (industry_0_100, impact_0_100, maturity_0_100, radar_total_0_100)
    cada uno array (n,).
    """
    doc_embs = encode_texts(embedding_texts)
    n = len(embedding_texts)
    ind = np.zeros(n, dtype=np.float32)
    imp = np.zeros(n, dtype=np.float32)
    mat = np.zeros(n, dtype=np.float32)

    url_map = url_ok_by_host or {}
    for i in range(n):
        ind[i] = 100.0 * _anchor_max_similarity(doc_embs[i], industry_embs)
        imp[i] = 100.0 * _anchor_max_similarity(doc_embs[i], impact_embs)
        host = web_hosts[i]
        mat[i] = maturity_score(
            descriptions[i],
            host,
            extra_counts[i],
            url_ok=url_map.get(host, False),
        )

    radar = (
        WEIGHT_INDUSTRY * ind + WEIGHT_IMPACT * imp + WEIGHT_MATURITY * mat
    ).astype(np.float32)
    return ind, imp, mat, radar


def precompute_anchor_embeddings() -> tuple[np.ndarray, np.ndarray]:
    ind = encode_texts(list(INDUSTRY_ANCHORS_ES))
    imp = encode_texts(list(IMPACT_ANCHORS_ES))
    return ind, imp
