"""
Text cleaning utilities for social media posts.

Designed to be called after platform normalisation so both Twitter and Reddit
text passes through the same pipeline.
"""

from __future__ import annotations

import html
import re
import unicodedata

_URL_RE = re.compile(r"https?://\S+|www\.\S+|t\.co/\S+", flags=re.IGNORECASE)
# Match @mention plus any immediately-following colon/comma so "RT @foo:" → ""
_MENTION_RE = re.compile(r"@[A-Za-z0-9_]{1,15}[,:]*")
_SUBREDDIT_RE = re.compile(r"\br/[A-Za-z0-9_]+", flags=re.IGNORECASE)
_USER_REF_RE = re.compile(r"\bu/[A-Za-z0-9_-]+", flags=re.IGNORECASE)
_RT_PREFIX_RE = re.compile(r"^RT\s+", flags=re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")


class TextCleaner:
    """
    Stateless text normaliser.

    Steps applied in order:
    1. HTML unescape
    2. Unicode NFKC normalisation
    3. Strip leading "RT " (Twitter retweets)
    4. Remove URLs and @mentions
    5. Remove Reddit-style r/subreddit and u/username references
    6. Keep hashtag words, drop the '#' symbol
    7. Collapse whitespace
    """

    def clean(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = html.unescape(text)
        text = unicodedata.normalize("NFKC", text)
        text = _RT_PREFIX_RE.sub("", text)
        text = _URL_RE.sub(" ", text)
        text = _MENTION_RE.sub(" ", text)
        text = _SUBREDDIT_RE.sub(" ", text)
        text = _USER_REF_RE.sub(" ", text)
        text = text.replace("#", "")
        return _WHITESPACE_RE.sub(" ", text).strip()

    def clean_series(self, series: "pd.Series") -> "pd.Series":  # noqa: F821
        return series.apply(self.clean)
