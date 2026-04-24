"""
BERTopic wrapper.

The TopicClusterer owns the UMAP / HDBSCAN / c-TF-IDF configuration and
exposes a clean fit() / get_topic_info() interface.  Passing a custom
embedding model is optional — if omitted the BERTopic default is used and
the caller must pass pre-computed embeddings to fit().
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import ClusteringConfig
from ..embedders.base import BaseEmbedder

logger = logging.getLogger(__name__)


@dataclass
class TopicResult:
    """Return value of TopicClusterer.fit()."""
    topic_ids: list[int]
    """Per-document topic assignment (-1 = outlier)."""

    probabilities: np.ndarray
    """Per-document topic probability vector."""

    topic_info: pd.DataFrame
    """Summary DataFrame from BERTopic.get_topic_info()."""

    model: object
    """The underlying BERTopic model (for advanced use)."""


class TopicClusterer:
    """
    Parameters
    ----------
    config:
        ClusteringConfig with UMAP, HDBSCAN, and c-TF-IDF knobs.
    embedding_model:
        Optional BaseEmbedder instance.  When provided it is passed to
        BERTopic so you only embed once.  When omitted, pass pre-computed
        ``embeddings`` to ``fit()``.
    """

    def __init__(
        self,
        config: ClusteringConfig | None = None,
        embedding_model: BaseEmbedder | None = None,
    ) -> None:
        self.config = config or ClusteringConfig()
        self._embedding_model = embedding_model
        self._model = None  # built lazily in fit()

    # ------------------------------------------------------------------

    def fit(
        self,
        docs: list[str],
        embeddings: np.ndarray | None = None,
    ) -> TopicResult:
        """
        Cluster ``docs`` and return a TopicResult.

        Parameters
        ----------
        docs:
            Cleaned post texts.
        embeddings:
            Pre-computed embedding matrix (n_docs × dim).  Required when
            no embedding_model was passed to the constructor.
        """
        try:
            from bertopic import BERTopic
            from bertopic.vectorizers import ClassTfidfTransformer
            from hdbscan import HDBSCAN
            from sklearn.feature_extraction.text import CountVectorizer
            from umap import UMAP
        except ImportError as exc:
            raise ImportError(
                "bertopic, umap-learn, hdbscan are required. "
                "Install with: pip install bertopic umap-learn hdbscan"
            ) from exc

        cfg = self.config
        umap_cfg = cfg.umap
        hdb_cfg = cfg.hdbscan

        umap_model = UMAP(
            n_neighbors=umap_cfg.n_neighbors,
            n_components=umap_cfg.n_components,
            min_dist=umap_cfg.min_dist,
            metric=umap_cfg.metric,
            random_state=umap_cfg.random_state,
        )
        hdbscan_model = HDBSCAN(
            min_cluster_size=hdb_cfg.min_cluster_size,
            min_samples=hdb_cfg.min_samples,
            metric=hdb_cfg.metric,
            cluster_selection_method=hdb_cfg.cluster_selection_method,
            prediction_data=True,
        )
        vectorizer_model = CountVectorizer(
            stop_words="english",
            ngram_range=cfg.ngram_range,
            min_df=2,
        )
        ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

        self._model = BERTopic(
            embedding_model=self._embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            ctfidf_model=ctfidf_model,
            min_topic_size=cfg.min_topic_size,
            top_n_words=cfg.top_n_words,
            verbose=True,
        )

        topics, probs = self._model.fit_transform(docs, embeddings)
        topic_info = self._model.get_topic_info()

        logger.info(
            "BERTopic found %d topics (%d outliers / %d docs)",
            len(topic_info) - 1,
            sum(1 for t in topics if t == -1),
            len(docs),
        )
        return TopicResult(
            topic_ids=list(topics),
            probabilities=np.array(probs) if probs is not None else np.array([]),
            topic_info=topic_info,
            model=self._model,
        )

    def get_keywords(self, topic_id: int, n: int = 8) -> list[str]:
        """Return the top-n keywords for a topic (requires fit() first)."""
        if self._model is None:
            raise RuntimeError("Call fit() before get_keywords().")
        return [word for word, _ in self._model.get_topic(topic_id)[:n]]

    def get_representative_docs(self, topic_id: int) -> list[str]:
        if self._model is None:
            raise RuntimeError("Call fit() before get_representative_docs().")
        return self._model.get_representative_docs(topic_id) or []
