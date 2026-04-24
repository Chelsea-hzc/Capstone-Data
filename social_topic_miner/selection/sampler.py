"""
Representative post sampler.

For each selected topic we:
1. Filter by recency and a minimum engagement percentile.
2. Score survivors by cosine similarity to their sub-perspective centroid
   (representativeness) + normalised engagement (quality signal).
3. Sample via temperature-scaled softmax so near-centroid posts are favoured
   but some diversity is preserved.
4. Trim to the configured budget if over the maximum.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from ..config import SamplingConfig

logger = logging.getLogger(__name__)


@dataclass
class SampledTopic:
    topic_id: int
    selected_indices: list[int]
    """DataFrame indices of the selected posts."""

    n_perspectives: int
    keywords: list[str]


def _softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    x = np.asarray(x, dtype=float) / max(temperature, 1e-8)
    x = x - x.max()
    exp_x = np.exp(x)
    return exp_x / exp_x.sum()


class PostSampler:
    """
    Parameters
    ----------
    config:
        SamplingConfig with recency window, engagement floor, budgets, and weights.
    """

    def __init__(self, config: SamplingConfig | None = None) -> None:
        self.config = config or SamplingConfig()

    # ------------------------------------------------------------------

    def sample(
        self,
        df: pd.DataFrame,
        embeddings: np.ndarray,
        selected_topic_ids: list[int],
        now: datetime | None = None,
    ) -> list[SampledTopic]:
        """
        Sample representative posts for each topic in ``selected_topic_ids``.

        Parameters
        ----------
        df:
            Unified DataFrame with ``topic_id``, ``sub_perspective``,
            ``engagement_norm``, and ``created_at`` columns.
        embeddings:
            Full embedding matrix aligned with ``df``.
        selected_topic_ids:
            Ordered list of topic IDs to process.
        now:
            Reference timestamp for recency filtering (defaults to UTC now).
        """
        cfg = self.config
        now = now or datetime.now(timezone.utc)
        recency_cutoff = now - timedelta(hours=cfg.recency_window_hours)

        results: list[SampledTopic] = []

        for tid in selected_topic_ids:
            topic_mask = df["topic_id"] == tid
            indices = df.index[topic_mask].tolist()
            topic_embeds = embeddings[indices]

            n_perspectives = int(df.loc[indices, "sub_perspective"].max()) + 1
            total_budget = max(
                cfg.target_min,
                min(cfg.target_max, n_perspectives * cfg.posts_per_perspective),
            )
            per_budget = max(1, total_budget // n_perspectives)

            selected: list[int] = []

            for p_id in range(n_perspectives):
                p_mask = np.array(
                    [df.loc[idx, "sub_perspective"] == p_id for idx in indices]
                )
                p_indices = [indices[i] for i, m in enumerate(p_mask) if m]
                if not p_indices:
                    continue

                eligible = self._apply_filters(df, p_indices, recency_cutoff)

                # Fallback: if all posts were filtered out, keep the most recent
                if not eligible:
                    fallback = max(p_indices, key=lambda i: df.loc[i, "created_at"])
                    selected.append(fallback)
                    continue

                if len(eligible) <= per_budget:
                    selected.extend(eligible)
                    continue

                # Score: cosine sim to perspective centroid + engagement
                p_local_positions = [indices.index(i) for i in eligible]
                p_embeds = topic_embeds[p_local_positions]
                centroid = p_embeds.mean(axis=0, keepdims=True)
                sim_scores = cosine_similarity(p_embeds, centroid).flatten()
                eng_scores = df.loc[eligible, "engagement_norm"].values.astype(float)
                eng_max = eng_scores.max() or 1.0
                combined = (
                    cfg.representativeness_weight * sim_scores
                    + cfg.engagement_weight * (eng_scores / eng_max)
                )
                probs = _softmax(combined, cfg.temperature)
                n_sample = min(per_budget, len(eligible))
                chosen_local = np.random.choice(len(eligible), size=n_sample, replace=False, p=probs)
                selected.extend(eligible[i] for i in chosen_local)

            # Trim if over budget
            if len(selected) > cfg.target_max:
                selected = self._trim_to_budget(selected, indices, topic_embeds, df, cfg.target_max)

            results.append(SampledTopic(
                topic_id=tid,
                selected_indices=selected,
                n_perspectives=n_perspectives,
                keywords=[],  # filled in by pipeline after clustering
            ))

        return results

    # ------------------------------------------------------------------

    def _apply_filters(
        self,
        df: pd.DataFrame,
        indices: list[int],
        recency_cutoff: datetime,
    ) -> list[int]:
        cfg = self.config
        return [
            idx for idx in indices
            if (
                df.loc[idx, "created_at"] >= recency_cutoff
                and df.loc[idx, "engagement_norm"] >= cfg.engagement_floor_percentile
            )
        ]

    def _trim_to_budget(
        self,
        selected: list[int],
        all_indices: list[int],
        topic_embeds: np.ndarray,
        df: pd.DataFrame,
        budget: int,
    ) -> list[int]:
        centroid = topic_embeds.mean(axis=0, keepdims=True)
        scored = []
        for idx in selected:
            local_pos = all_indices.index(idx)
            sim = cosine_similarity(
                topic_embeds[local_pos : local_pos + 1], centroid
            ).flatten()[0]
            eng = df.loc[idx, "engagement_norm"]
            scored.append((idx, 0.7 * sim + 0.3 * eng))
        scored.sort(key=lambda x: -x[1])
        return [idx for idx, _ in scored[:budget]]
