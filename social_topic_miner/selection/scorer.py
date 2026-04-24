"""
Engagement scoring and topic ranking.

Engagement values differ wildly between platforms (a Reddit post with 10k upvotes
and a tweet with 10k likes carry different social signals).  We normalise within
each platform via percentile rank so scores are directly comparable, then compute
a composite topic score that balances post count, total engagement heat, and
per-post quality.
"""

from __future__ import annotations

import logging

import pandas as pd

from ..config import SelectionConfig

logger = logging.getLogger(__name__)


class EngagementScorer:
    """
    Parameters
    ----------
    config:
        SelectionConfig with platform weights and composite-score weights.
    """

    def __init__(self, config: SelectionConfig | None = None) -> None:
        self.config = config or SelectionConfig()

    # ------------------------------------------------------------------

    def add_engagement_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute ``engagement_raw`` and ``engagement_norm`` columns on ``df``.

        ``engagement_norm`` is a 0-1 within-platform percentile rank so a post
        at the 90th percentile on Twitter and one at the 90th on Reddit both
        get ~0.9, regardless of absolute counts.
        """
        cfg = self.config
        df = df.copy()
        df["engagement_raw"] = df.apply(self._raw_score, axis=1)
        df["engagement_norm"] = (
            df.groupby("platform")["engagement_raw"].rank(pct=True)
        )
        return df

    def rank_topics(
        self, df: pd.DataFrame, exclude_outlier: bool = True
    ) -> pd.DataFrame:
        """
        Return a DataFrame of topics sorted by composite score (descending).

        Columns: topic_id, n_posts, total_engagement, avg_engagement,
                 composite_score.
        """
        if "engagement_norm" not in df.columns:
            df = self.add_engagement_columns(df)

        subset = df[df["topic_id"] != -1] if exclude_outlier else df

        stats = (
            subset.groupby("topic_id")
            .agg(
                n_posts=("post_id", "count"),
                total_engagement=("engagement_norm", "sum"),
                avg_engagement=("engagement_norm", "mean"),
            )
            .reset_index()
        )

        for col in ("n_posts", "total_engagement", "avg_engagement"):
            lo, hi = stats[col].min(), stats[col].max()
            stats[f"{col}_norm"] = (
                (stats[col] - lo) / (hi - lo) if hi > lo else 0.0
            )

        cfg = self.config
        stats["composite_score"] = (
            cfg.weight_size            * stats["n_posts_norm"]
            + cfg.weight_total_engagement * stats["total_engagement_norm"]
            + cfg.weight_avg_engagement   * stats["avg_engagement_norm"]
        )

        return stats.sort_values("composite_score", ascending=False).reset_index(drop=True)

    def top_topic_ids(self, df: pd.DataFrame) -> list[int]:
        ranked = self.rank_topics(df)
        return ranked.head(self.config.top_n_topics)["topic_id"].tolist()

    # ------------------------------------------------------------------

    def _raw_score(self, row: pd.Series) -> float:
        cfg = self.config
        if row["platform"] == "twitter":
            return (
                row["engagement_comments"] * cfg.twitter_reply_weight
                + row["engagement_shares"] * cfg.twitter_share_weight
                + row["engagement_likes"]  * cfg.twitter_like_weight
            )
        if row["platform"] == "reddit":
            return (
                row["engagement_comments"] * cfg.reddit_comment_weight
                + row["engagement_likes"]  * cfg.reddit_like_weight
            )
        # Generic fallback
        return float(
            row.get("engagement_likes", 0)
            + row.get("engagement_comments", 0)
            + row.get("engagement_shares", 0)
        )
