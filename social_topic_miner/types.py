"""
Stable input/output types for the TopicMinerAPI.

These TypedDicts define the contract between this package and the
full-stack backend.  The backend should import these to validate
the shapes it sends and receives.

The *required* fields are the minimum the package needs to function.
Optional fields are used when available but silently ignored when absent.
"""

from __future__ import annotations

from typing import TypedDict


# ---------------------------------------------------------------------------
# Section 1 — input
# ---------------------------------------------------------------------------

class PostIn(TypedDict, total=False):
    """
    A single social-media post.  Send as many fields as you have;
    only ``text`` and ``platform`` are required.
    """
    # Required
    text: str
    platform: str           # "twitter" | "reddit"

    # Twitter fields
    tweet_id: str
    author_id: str
    username: str
    created_at: str         # ISO-8601 string, e.g. "2026-04-17T18:46:14.000Z"
    like_count: int
    reply_count: int
    retweet_count: int
    impression_count: int
    possibly_sensitive: bool

    # Reddit fields
    reddit_id: str
    author: str
    title: str              # used in place of / in addition to text
    selftext: str
    ups: int
    num_comments: int
    over_18: bool
    subreddit: str
    permalink: str
    created_utc: int        # Unix timestamp


# ---------------------------------------------------------------------------
# Section 1 — output
# ---------------------------------------------------------------------------

class RepresentativePost(TypedDict):
    post_id: str
    text: str
    platform: str
    sub_perspective: int
    engagement_norm: float
    created_at: str


class TopicOut(TypedDict):
    topic_id: int
    headline: str           # LLM-generated; empty string when no summarizer
    category: str           # LLM-generated; "Unknown" when no summarizer
    keywords: list[str]
    key_points: list[str]   # LLM-generated; empty list when no summarizer
    n_posts: int
    n_perspectives: int
    representative_posts: list[RepresentativePost]


class Section1Response(TypedDict):
    topics: list[TopicOut]
    total_posts_processed: int


# ---------------------------------------------------------------------------
# Section 2 — input / output
# ---------------------------------------------------------------------------

class Section2Request(TypedDict, total=False):
    """Pass either ``topics`` (from Section 1) or bare ``headlines``/``keywords``."""
    topics: list[TopicOut]      # preferred — direct output of section1()
    headlines: list[str]        # alternative when topics are not available
    keywords: list[str]         # alternative when topics are not available


class SearchQueryOut(TypedDict):
    query_string: str
    platform: str               # "twitter" | "reddit" | "any"
    intent: str                 # "opposing" | "diverse" | "related" | "factual"
    source_topic_id: int
    source_keywords: list[str]
    metadata: dict


class Section2Response(TypedDict):
    queries: list[SearchQueryOut]
    source_topic_ids: list[int]


# ---------------------------------------------------------------------------
# Section 3 — input / output
# ---------------------------------------------------------------------------

class Section3Request(TypedDict, total=False):
    """
    new_posts is required.
    bubble_keywords improves relevance scoring (pass from Section 1 keywords).
    """
    new_posts: list[PostIn]     # required
    bubble_keywords: list[str]  # optional but recommended


class Section3Response(TypedDict):
    posts: list[PostIn]
    scores: list[float]         # diversity score per post, same order as posts
    dropped: int                # posts removed by the cutoff threshold


# ---------------------------------------------------------------------------
# Full pipeline — input / output
# ---------------------------------------------------------------------------

class FullPipelineRequest(TypedDict, total=False):
    posts: list[PostIn]         # required — user's existing feed
    new_posts: list[PostIn]     # optional — diverse posts already fetched externally


class FullPipelineResponse(TypedDict):
    section1: Section1Response
    section2: Section2Response
    section3: Section3Response | None   # None when new_posts was not provided
