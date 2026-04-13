from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from wikicid_intel.config import (
    SEMANTIC_COLUMNS,
    WEIGHT_RADAR_IN_RANK,
    WEIGHT_SEMANTIC_IN_RANK,
)
from wikicid_intel.pipeline.clustering import assign_clusters
from wikicid_intel.pipeline.embedder import cosine_sim_matrix, encode_texts
from wikicid_intel.pipeline.excel_loader import load_empresas
from wikicid_intel.pipeline.scoring import (
    batch_website_reachable,
    compute_radar_vectors,
    precompute_anchor_embeddings,
)
from wikicid_intel.pipeline.text_normalize import clean_url, fingerprint_row, text_for_embedding


EXTRA_FIELDS_FOR_MATURITY = [
    "Tamaño",
    "Fondeo",
    "Ingresos",
    "Valuación",
    "Oportunidades",
    "Sede",
    "Alianzas",
    "Mercados",
]


@dataclass
class SearchHit:
    rank: int
    empresa: str
    descripcion: str
    pagina_web: str
    mercados: str
    cluster: int
    similitud_semantica: float
    score_industria: float
    score_impacto: float
    score_madurez: float
    radar_total: float
    score_combinado: float


def _row_embedding_parts(row: pd.Series) -> list[str]:
    parts: list[str] = []
    for col in SEMANTIC_COLUMNS:
        if col not in row.index:
            continue
        parts.append(row[col])
    return parts


def _extra_filled_count(row: pd.Series) -> int:
    n = 0
    for col in EXTRA_FIELDS_FOR_MATURITY:
        if col not in row.index:
            continue
        v = row[col]
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        s = str(v).strip()
        if s and s.lower() != "nan":
            n += 1
    return n


def build_frame_from_excel(
    excel_path: str | Path,
    sheet_name: str | None = None,
) -> pd.DataFrame:
    raw = load_empresas(excel_path, sheet_name=sheet_name)
    rows: list[dict] = []
    for idx, row in raw.iterrows():
        empresa = row.get("Empresa", "")
        if pd.isna(empresa):
            empresa = ""
        parts = _row_embedding_parts(row)
        emb_text = text_for_embedding([str(p) if p is not None else "" for p in parts])
        host = clean_url(row.get("Página Web"))
        desc = row.get("Descripción", "")
        desc = "" if pd.isna(desc) else str(desc)
        fp = fingerprint_row(str(empresa), host, emb_text)
        rows.append(
            {
                "_orig_idx": idx,
                "Empresa": str(empresa).strip() if empresa is not None else "",
                "Descripción": desc,
                "Página Web": str(row.get("Página Web", "") or ""),
                "web_host": host,
                "Mercados": str(row.get("Mercados", "") or ""),
                "embedding_text": emb_text,
                "fingerprint": fp,
                "_extra_n": _extra_filled_count(row),
            }
        )

    df = pd.DataFrame(rows)
    before = len(df)
    df = df.drop_duplicates(subset=["fingerprint"], keep="first").reset_index(drop=True)
    df.attrs["rows_before_dedup"] = before
    df.attrs["rows_after_dedup"] = len(df)
    return df


class CompanyIndex:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.embeddings: np.ndarray | None = None
        self.industry: np.ndarray | None = None
        self.impact: np.ndarray | None = None
        self.maturity: np.ndarray | None = None
        self.radar: np.ndarray | None = None
        self.clusters: np.ndarray | None = None

    def build(self) -> None:
        texts = self.df["embedding_text"].tolist()
        self.embeddings = encode_texts(texts)
        ind_emb, imp_emb = precompute_anchor_embeddings()
        descs = self.df["Descripción"].tolist()
        hosts = self.df["web_host"].tolist()
        extras = self.df["_extra_n"].astype(int).tolist()
        url_ok = batch_website_reachable(hosts)
        ind, imp, mat, radar = compute_radar_vectors(
            texts, descs, hosts, extras, ind_emb, imp_emb, url_ok_by_host=url_ok
        )
        self.industry = ind
        self.impact = imp
        self.maturity = mat
        self.radar = radar

        ind_n = ind / 100.0
        imp_n = imp / 100.0
        mat_n = mat / 100.0
        self.clusters = assign_clusters(self.embeddings, ind_n, imp_n, mat_n)

    def search(
        self,
        query: str,
        top_k: int = 15,
        weight_semantic: float | None = None,
        weight_radar: float | None = None,
    ) -> list[SearchHit]:
        if self.embeddings is None or self.radar is None:
            raise RuntimeError("Índice no construido. Llama a .build().")

        w_s = weight_semantic if weight_semantic is not None else WEIGHT_SEMANTIC_IN_RANK
        w_r = weight_radar if weight_radar is not None else WEIGHT_RADAR_IN_RANK
        total_w = w_s + w_r
        if total_w <= 0:
            w_s, w_r = 0.5, 0.5
        else:
            w_s, w_r = w_s / total_w, w_r / total_w

        q_emb = encode_texts([query.strip()])[0]
        sims = cosine_sim_matrix(q_emb, self.embeddings)
        radar_n = self.radar / 100.0
        combined = w_s * sims + w_r * radar_n
        order = np.argsort(-combined)[:top_k]

        hits: list[SearchHit] = []
        for rank, i in enumerate(order, start=1):
            row = self.df.iloc[int(i)]
            hits.append(
                SearchHit(
                    rank=rank,
                    empresa=row["Empresa"],
                    descripcion=row["Descripción"][:500]
                    + ("…" if len(str(row["Descripción"])) > 500 else ""),
                    pagina_web=row["Página Web"],
                    mercados=str(row["Mercados"]),
                    cluster=int(self.clusters[int(i)]),
                    similitud_semantica=float(sims[int(i)]),
                    score_industria=float(self.industry[int(i)]),
                    score_impacto=float(self.impact[int(i)]),
                    score_madurez=float(self.maturity[int(i)]),
                    radar_total=float(self.radar[int(i)]),
                    score_combinado=float(combined[int(i)]),
                )
            )
        return hits


def load_index(excel_path: str | Path, sheet_name: str | None = None) -> CompanyIndex:
    df = build_frame_from_excel(excel_path, sheet_name=sheet_name)
    idx = CompanyIndex(df)
    idx.build()
    return idx
