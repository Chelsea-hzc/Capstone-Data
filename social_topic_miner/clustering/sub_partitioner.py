"""
KMeans sub-perspective partitioning.

Within each selected topic we run KMeans to identify distinct sub-angles of
discussion (e.g. inside "AI models" you might find "pricing concerns",
"benchmark results", "safety debate").  The optimal k is chosen by
silhouette score.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..config import PartitionConfig

logger = logging.getLogger(__name__)


class SubPartitioner:
    """
    Parameters
    ----------
    config:
        PartitionConfig with min_k, max_k, and min_posts_per_perspective.
    """

    def __init__(self, config: PartitionConfig | None = None) -> None:
        self.config = config or PartitionConfig()

    # ------------------------------------------------------------------

    def partition(
        self,
        df: pd.DataFrame,
        embeddings: np.ndarray,
        topic_ids: list[int],
    ) -> pd.DataFrame:
        """
        Add a ``sub_perspective`` integer column to ``df``.

        Parameters
        ----------
        df:
            The unified post DataFrame with a ``topic_id`` column already set.
        embeddings:
            Full embedding matrix aligned with ``df`` (same row order).
        topic_ids:
            The topic IDs to partition (typically the top-N selected topics).

        Returns
        -------
        pd.DataFrame
            The same ``df`` with ``sub_perspective`` column added in-place.
        """
        df = df.copy()
        df["sub_perspective"] = -1

        for tid in topic_ids:
            mask = df["topic_id"] == tid
            indices = df.index[mask].tolist()
            topic_embeds = embeddings[indices]
            best_k = self._find_best_k(topic_embeds)

            if best_k == 1:
                labels = np.zeros(len(indices), dtype=int)
            else:
                from sklearn.cluster import KMeans
                km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
                labels = km.fit_predict(topic_embeds)

            for idx, label in zip(indices, labels):
                df.loc[idx, "sub_perspective"] = int(label)

            sizes = pd.Series(labels).value_counts().sort_index().to_dict()
            logger.info(
                "Topic %d → %d sub-perspectives, sizes: %s", tid, best_k, sizes
            )

        return df

    # ------------------------------------------------------------------

    def _find_best_k(self, topic_embeddings: np.ndarray) -> int:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        cfg = self.config
        n = len(topic_embeddings)
        actual_max_k = min(cfg.max_k, n // cfg.min_posts_per_perspective)

        if actual_max_k < cfg.min_k:
            return 1

        best_k, best_score = 1, -1.0
        for k in range(cfg.min_k, actual_max_k + 1):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(topic_embeddings)
            # silhouette_score requires at least 2 distinct labels
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(topic_embeddings, labels, metric="cosine")
            if score > best_score:
                best_score, best_k = score, k

        return best_k
