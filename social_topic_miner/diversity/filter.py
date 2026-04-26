"""
Section 3 — Diversity filter.

Receives a list of posts fetched by the full-stack backend using the
Section 2 queries, scores them for relevance + diversity, applies a
cutoff threshold, and returns a curated set of posts suitable for the
"break your echo chamber" page.

PLACEHOLDER — the scoring methods are stubs.  The final scoring strategy
(embedding-based divergence, stance detection, source diversity, etc.)
will be decided once the product direction is confirmed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DiversityFilterConfig:
    # --- Cutoff ---
    min_diversity_score: float = 0.30
    """Posts scoring below this are dropped.  0 = keep all, 1 = keep none."""

    max_posts_out: int = 20
    """Hard cap on how many posts are returned per call."""

    # --- Scoring weights (must sum to 1.0) ---
    weight_relevance: float  = 0.40
    """How on-topic the post is relative to the original bubble topics."""

    weight_divergence: float = 0.40
    """How semantically different the post is from the bubble centroid."""

    weight_recency: float    = 0.20
    """Prefer recent posts."""

    # --- Embedding ---
    use_embeddings: bool = True
    """
    When True, compute cosine divergence from the bubble centroid.
    When False, fall back to keyword-overlap heuristic (no model needed).
    """


@dataclass
class DiversityResult:
    posts: list[dict]
    """Filtered and ranked posts, highest diversity score first."""

    scores: list[float]
    """Diversity score for each post (aligned with posts list)."""

    dropped: int
    """How many posts were removed by the cutoff."""

    metadata: dict = field(default_factory=dict)


class DiversityFilter:
    """
    Section 3 entry-point.

    Parameters
    ----------
    config:
        DiversityFilterConfig with scoring weights and cutoff threshold.
    embedder:
        Optional BaseEmbedder — used when config.use_embeddings is True
        to encode the incoming posts and compute divergence from the
        bubble centroid.

    Example
    -------
    >>> from social_topic_miner.diversity import DiversityFilter
    >>> f = DiversityFilter()
    >>> result = f.filter(
    ...     new_posts=[{"text": "...", "platform": "twitter", ...}],
    ...     bubble_keywords=["AI", "safety", "deployment"],
    ... )
    >>> for post, score in zip(result.posts, result.scores):
    ...     print(f"{score:.2f}  {post['text'][:80]}")
    """

    def __init__(
        self,
        config: DiversityFilterConfig | None = None,
        embedder=None,     # BaseEmbedder — typed loosely to avoid circular import
    ) -> None:
        self.config = config or DiversityFilterConfig()
        self.embedder = embedder

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def filter(
        self,
        new_posts: list[dict],
        bubble_keywords: list[str] | None = None,
        bubble_embeddings: np.ndarray | None = None,
    ) -> DiversityResult:
        """
        Score and filter a list of posts fetched via Section 2 queries.

        Parameters
        ----------
        new_posts:
            Raw posts from the social-media API (same schema as Section 1
            input: must have at least a ``text`` key).
        bubble_keywords:
            Top keywords from Section 1 — used for relevance scoring and
            the keyword-overlap fallback.
        bubble_embeddings:
            Embedding matrix of the original bubble posts (n × dim).
            When provided, used to compute the bubble centroid for
            divergence scoring.
        """
        if not new_posts:
            return DiversityResult(posts=[], scores=[], dropped=0)

        texts = [p.get("text", "") for p in new_posts]
        scores = self._score(texts, bubble_keywords or [], bubble_embeddings)

        cfg = self.config
        before = len(new_posts)
        pairs = [
            (post, score)
            for post, score in zip(new_posts, scores)
            if score >= cfg.min_diversity_score
        ]

        # Sort highest-diversity first, then cap
        pairs.sort(key=lambda x: -x[1])
        pairs = pairs[: cfg.max_posts_out]

        kept_posts  = [p for p, _ in pairs]
        kept_scores = [s for _, s in pairs]

        logger.info(
            "DiversityFilter: %d → %d posts (dropped %d below threshold %.2f)",
            before, len(kept_posts), before - len(kept_posts), cfg.min_diversity_score,
        )
        return DiversityResult(
            posts=kept_posts,
            scores=kept_scores,
            dropped=before - len(kept_posts),
        )

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score(
        self,
        texts: list[str],
        bubble_keywords: list[str],
        bubble_embeddings: np.ndarray | None,
    ) -> list[float]:
        relevance  = self._score_relevance(texts, bubble_keywords)
        divergence = self._score_divergence(texts, bubble_embeddings)
        recency    = [0.5] * len(texts)          # placeholder — no timestamps here yet

        cfg = self.config
        combined = (
            cfg.weight_relevance  * np.array(relevance)
            + cfg.weight_divergence * np.array(divergence)
            + cfg.weight_recency    * np.array(recency)
        )
        return combined.tolist()

    def _score_relevance(
        self, texts: list[str], bubble_keywords: list[str]
    ) -> list[float]:
        """
        PLACEHOLDER — keyword overlap as a simple relevance proxy.

        TODO: replace with embedding cosine similarity to topic centroid
              once we have confirmed the embedding approach.
        """
        if not bubble_keywords:
            return [0.5] * len(texts)

        kw_set = {k.lower() for k in bubble_keywords}
        scores = []
        for text in texts:
            words = set(text.lower().split())
            overlap = len(words & kw_set) / max(len(kw_set), 1)
            scores.append(min(overlap, 1.0))
        return scores

    def _score_divergence(
        self,
        texts: list[str],
        bubble_embeddings: np.ndarray | None,
    ) -> list[float]:
        """
        PLACEHOLDER — semantic divergence from the bubble centroid.

        High divergence = post is far from the user's existing bubble
        (good for breaking echo chambers).

        Current logic:
        - If embedder + bubble_embeddings available: 1 - cosine_similarity
          to bubble centroid.
        - Otherwise: uniform 0.5 (neutral).

        TODO: consider stance detection or framing analysis as a richer signal.
        """
        if self.embedder is None or bubble_embeddings is None:
            return [0.5] * len(texts)

        from sklearn.metrics.pairwise import cosine_similarity

        new_embeds   = self.embedder.embed(texts)                         # (n, dim)
        bubble_centroid = bubble_embeddings.mean(axis=0, keepdims=True)   # (1, dim)
        similarities = cosine_similarity(new_embeds, bubble_centroid).flatten()
        # divergence = 1 - similarity (clip to [0, 1])
        return np.clip(1.0 - similarities, 0.0, 1.0).tolist()
