"""
Platform schema normalisation.

Converts raw Twitter or Reddit JSON records into a unified DataFrame with
consistent column names and dtypes so the rest of the pipeline is
platform-agnostic.

Unified schema
--------------
post_id              : str
author               : str | None
author_id            : str | None
text                 : str          (raw; cleaned copy added later as text_clean)
created_at           : pd.Timestamp (UTC)
engagement_likes     : int
engagement_comments  : int
engagement_shares    : int
impressions          : int
possibly_sensitive   : bool
platform             : str          ("twitter" | "reddit" | ...)
subreddit            : str | None
permalink            : str | None
"""

from __future__ import annotations

import pandas as pd

from ..config import PreprocessingConfig
from .cleaner import TextCleaner


class PostNormalizer:
    """
    Loads, normalises, cleans, and filters a raw social-media dataset.

    Parameters
    ----------
    config:
        Preprocessing knobs (min_word_count, drop_nsfw, dedup).
    cleaner:
        Optional custom TextCleaner instance (defaults to a fresh one).
    """

    UNIFIED_COLS = [
        "post_id", "author", "author_id", "text", "created_at",
        "engagement_likes", "engagement_comments", "engagement_shares",
        "impressions", "possibly_sensitive", "platform", "subreddit", "permalink",
    ]

    def __init__(
        self,
        config: PreprocessingConfig | None = None,
        cleaner: TextCleaner | None = None,
    ) -> None:
        self.config = config or PreprocessingConfig()
        self.cleaner = cleaner or TextCleaner()

    # ------------------------------------------------------------------
    # Public entry-points
    # ------------------------------------------------------------------

    def from_json(self, path: str) -> pd.DataFrame:
        """Load the combined timeline JSON produced by the data-collection step."""
        import json

        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)

        posts = raw.get("posts", raw) if isinstance(raw, dict) else raw
        df_raw = pd.DataFrame(posts)
        return self.from_dataframe(df_raw)

    def from_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Accept a raw DataFrame that may contain mixed platform rows and return
        a normalised, cleaned, filtered DataFrame.
        """
        parts: list[pd.DataFrame] = []

        if "platform" in df.columns:
            twitter_mask = df["platform"] == "twitter"
            reddit_mask = df["platform"] == "reddit"
        else:
            # Infer platform from available columns
            twitter_mask = df.columns.isin(["tweet_id"]).any()
            reddit_mask = ~twitter_mask

        if twitter_mask.any():
            parts.append(self._normalise_twitter(df[twitter_mask].copy()))
        if reddit_mask.any():
            parts.append(self._normalise_reddit(df[reddit_mask].copy()))

        # Support arbitrary extra platforms via extend_platforms()
        for platform_name, platform_df in self._extra_platform_rows(df).items():
            parts.append(platform_df)

        if not parts:
            raise ValueError("No recognisable platform data found in DataFrame.")

        unified = pd.concat(parts, ignore_index=True)
        unified = unified[unified["post_id"].notna()].copy()
        return self._clean_and_filter(unified)

    # ------------------------------------------------------------------
    # Platform-specific normalisers
    # ------------------------------------------------------------------

    def _normalise_twitter(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({
            "post_id":             df.get("tweet_id", df.get("post_id", pd.Series(dtype=str))).astype(str),
            "author":              df.get("username"),
            "author_id":           df.get("author_id", df.get("user_id")).astype(str),
            "text":                df.get("text", df.get("full_text", pd.Series(dtype=str))),
            "created_at":          pd.to_datetime(df.get("created_at"), utc=True, errors="coerce"),
            "engagement_likes":    df.get("like_count", pd.Series(0)).fillna(0).astype(int),
            "engagement_comments": df.get("reply_count", pd.Series(0)).fillna(0).astype(int),
            "engagement_shares":   df.get("retweet_count", pd.Series(0)).fillna(0).astype(int),
            "impressions":         df.get("impression_count", pd.Series(0)).fillna(0).astype(int),
            "possibly_sensitive":  df.get("possibly_sensitive", pd.Series(False)).fillna(False).astype(bool),
            "platform":            "twitter",
            "subreddit":           None,
            "permalink":           None,
        })

    def _normalise_reddit(self, df: pd.DataFrame) -> pd.DataFrame:
        title = df.get("title", pd.Series("", index=df.index)).fillna("").astype(str)
        selftext = df.get("selftext", pd.Series("", index=df.index)).fillna("").astype(str)
        combined = title.where(selftext.str.strip() == "", title + "\n\n" + selftext)

        return pd.DataFrame({
            "post_id":             df.get("reddit_id", df.get("id", pd.Series(dtype=str))).astype(str),
            "author":              df.get("author"),
            "author_id":           df.get("author"),
            "text":                combined,
            "created_at":          pd.to_datetime(
                                       df.get("created_utc"), unit="s", utc=True, errors="coerce"
                                   ),
            "engagement_likes":    df.get("ups", pd.Series(0)).fillna(0).astype(int),
            "engagement_comments": df.get("num_comments", pd.Series(0)).fillna(0).astype(int),
            "engagement_shares":   0,
            "impressions":         0,
            "possibly_sensitive":  df.get("over_18", pd.Series(False)).fillna(False).astype(bool),
            "platform":            "reddit",
            "subreddit":           df.get("subreddit"),
            "permalink":           df.get("permalink"),
        })

    def _extra_platform_rows(self, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """
        Override in a subclass to add support for additional platforms
        (e.g. Mastodon, Bluesky).  Return a mapping of platform_name → normalised df.
        """
        return {}

    # ------------------------------------------------------------------
    # Cleaning & filtering
    # ------------------------------------------------------------------

    def _clean_and_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        # Preserve raw text
        df["text_raw"] = df["text"]
        df["text"] = self.cleaner.clean_series(df["text_raw"])

        # Drop empty
        df = df[df["text"].str.strip().str.len() > 0].copy()

        # Dedup
        if self.config.dedup:
            df["_dedup_key"] = df["text"].str.lower()
            df = df.drop_duplicates(subset="_dedup_key", keep="first").drop(columns="_dedup_key")

        # NSFW
        if self.config.drop_nsfw:
            df = df[~df["possibly_sensitive"].astype(bool)].copy()

        # Minimum word count
        df["word_count"] = df["text"].str.split().str.len()
        df = df[df["word_count"] >= self.config.min_word_count].copy()

        # Sort newest-first
        df = df.sort_values("created_at", ascending=False).reset_index(drop=True)
        return df
