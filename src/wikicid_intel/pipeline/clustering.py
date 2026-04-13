from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def assign_clusters(
    embeddings: np.ndarray,
    industry_n: np.ndarray,
    impact_n: np.ndarray,
    maturity_n: np.ndarray,
    random_state: int = 42,
) -> np.ndarray:
    """
    Clusters que mezclan geometría semántica (PCA de embeddings) con los tres ejes Radar normalizados.
    industry_n, impact_n, maturity_n en [0,1].
    """
    n = embeddings.shape[0]
    if n < 4:
        return np.zeros(n, dtype=np.int32)

    n_comp = min(16, embeddings.shape[1], max(2, n // 3))
    pca = PCA(n_components=n_comp, random_state=random_state)
    reduced = pca.fit_transform(embeddings)

    feat = np.hstack(
        [
            reduced,
            industry_n.reshape(-1, 1),
            impact_n.reshape(-1, 1),
            maturity_n.reshape(-1, 1),
        ]
    )
    scaled = StandardScaler().fit_transform(feat)

    k = int(max(3, min(24, round((n / 2) ** 0.5))))
    k = min(k, n - 1)
    if k < 2:
        return np.zeros(n, dtype=np.int32)

    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    return km.fit_predict(scaled).astype(np.int32)
