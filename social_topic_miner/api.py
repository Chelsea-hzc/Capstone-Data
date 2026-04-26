"""
Public API layer — the full-stack backend calls these functions directly.

Each section maps to one logical page in the product:

  Page 1  section1()  — "Your bubble"       : cluster & summarise the user's feed
  Page 2  section2()  — "Other perspectives": generate queries to break echo chamber
  Page 3  section3()  — "Diverse content"   : filter & rank the fetched diverse posts
  All     run_full()  — single call that chains 1 → 2 → 3

Request/response types are plain dataclasses so they serialise cleanly to
JSON (e.g. via dataclasses.asdict()) without any framework dependency.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field

import numpy as np

from .config import TopicMinerConfig
from .diversity.filter import DiversityFilter, DiversityFilterConfig, DiversityResult
from .echo_breaker.query_builder import QueryBuilder, QueryBuilderConfig, SearchQuery
from .embedders.base import BaseEmbedder
from .embedders.sentence_transformer import SentenceTransformerEmbedder
from .pipeline import PipelineResult, TopicMinerPipeline
from .summarizers.base import BaseSummarizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared response types
# ---------------------------------------------------------------------------

@dataclass
class TopicOut:
    """Section 1 output for a single topic."""
    topic_id: int
    headline: str
    category: str
    keywords: list[str]
    key_points: list[str]
    n_posts: int
    n_perspectives: int
    representative_posts: list[dict]


@dataclass
class Section1Response:
    topics: list[TopicOut]
    total_posts_processed: int


@dataclass
class Section2Response:
    """
    Queries the full-stack backend should execute against social-media APIs
    to fetch diverse/opposing content for Section 3.
    """
    queries: list[dict]          # SearchQuery as dicts for easy JSON serialisation
    source_topic_ids: list[int]


@dataclass
class Section3Response:
    posts: list[dict]
    scores: list[float]
    dropped: int


@dataclass
class FullPipelineResponse:
    """Single-call response carrying data for all three pages."""
    section1: Section1Response
    section2: Section2Response
    section3: Section3Response | None   # None when no new_posts were provided


# ---------------------------------------------------------------------------
# Main API class
# ---------------------------------------------------------------------------

class TopicMinerAPI:
    """
    Stateless façade over the three-section pipeline.

    Instantiate once (e.g. at app startup) and call the section methods
    as needed.

    Parameters
    ----------
    config:
        Master TopicMinerConfig (all defaults are sensible out-of-the-box).
    embedder:
        Shared embedder reused across all sections so texts are embedded
        in the same vector space.
    summarizer:
        Optional LLM for Section 1 headline generation and (later) Section 2
        query generation.
    query_config:
        QueryBuilderConfig for Section 2.
    diversity_config:
        DiversityFilterConfig for Section 3.
    """

    def __init__(
        self,
        config: TopicMinerConfig | None = None,
        embedder: BaseEmbedder | None = None,
        summarizer: BaseSummarizer | None = None,
        query_config: QueryBuilderConfig | None = None,
        diversity_config: DiversityFilterConfig | None = None,
    ) -> None:
        self._embedder   = embedder or SentenceTransformerEmbedder()
        self._pipeline   = TopicMinerPipeline(
            config=config,
            embedder=self._embedder,
            summarizer=summarizer,
        )
        self._qbuilder   = QueryBuilder(
            config=query_config,
            summarizer=summarizer,
        )
        self._dfilter    = DiversityFilter(
            config=diversity_config,
            embedder=self._embedder,
        )

    # ------------------------------------------------------------------
    # Section 1 — cluster & summarise the user's feed
    # ------------------------------------------------------------------

    def section1(self, posts: list[dict]) -> Section1Response:
        """
        Parameters
        ----------
        posts:
            Raw social-media posts.  Each dict must have at least ``text``
            and ``platform`` keys (full schema in PostNormalizer).

        Returns
        -------
        Section1Response with one TopicOut per detected topic.
        """
        import pandas as pd
        df_raw = pd.DataFrame(posts)
        result: PipelineResult = self._pipeline.run_from_dataframe(df_raw)
        self._last_pipeline_result = result   # stash for run_full()

        topics_out = self._pipeline_result_to_topics(result)
        return Section1Response(
            topics=topics_out,
            total_posts_processed=len(result.df),
        )

    # ------------------------------------------------------------------
    # Section 2 — generate echo-breaking search queries
    # ------------------------------------------------------------------

    def section2(
        self,
        headlines: list[str] | None = None,
        keywords: list[str] | None = None,
        topics: list[dict] | None = None,
    ) -> Section2Response:
        """
        Accepts either:
        - ``topics`` — a list of TopicOut dicts (direct output of section1)
        - ``headlines`` + ``keywords`` — manual overrides

        Returns
        -------
        Section2Response with a flat list of SearchQuery dicts.
        """
        if topics:
            query_map = self._qbuilder.build_batch(topics)
        else:
            # Single-topic shortcut: combine all headlines/keywords into one topic
            query_map = {
                0: self._qbuilder.build(
                    topic_id=0,
                    headline=(headlines or [""])[0],
                    keywords=keywords or [],
                )
            }

        all_queries: list[SearchQuery] = []
        for qs in query_map.values():
            all_queries.extend(qs)

        return Section2Response(
            queries=[asdict(q) for q in all_queries],
            source_topic_ids=list(query_map.keys()),
        )

    # ------------------------------------------------------------------
    # Section 3 — filter diverse posts
    # ------------------------------------------------------------------

    def section3(
        self,
        new_posts: list[dict],
        bubble_keywords: list[str] | None = None,
        bubble_embeddings: np.ndarray | None = None,
    ) -> Section3Response:
        """
        Parameters
        ----------
        new_posts:
            Posts fetched by the backend using the Section 2 queries.
        bubble_keywords:
            Keywords from Section 1 (for relevance scoring).
        bubble_embeddings:
            Embeddings of the original bubble posts (from PipelineResult.embeddings).
            Pass these for richer divergence scoring.
        """
        result: DiversityResult = self._dfilter.filter(
            new_posts=new_posts,
            bubble_keywords=bubble_keywords,
            bubble_embeddings=bubble_embeddings,
        )
        return Section3Response(
            posts=result.posts,
            scores=result.scores,
            dropped=result.dropped,
        )

    # ------------------------------------------------------------------
    # Combined — run all three sections in one call
    # ------------------------------------------------------------------

    def run_full(
        self,
        posts: list[dict],
        new_posts: list[dict] | None = None,
    ) -> FullPipelineResponse:
        """
        Single entry-point for the dynamic mode.

        Parameters
        ----------
        posts:
            The user's existing feed (Section 1 input).
        new_posts:
            Diverse posts already fetched externally (Section 3 input).
            If None, Section 3 is skipped and its response is None.

        Returns
        -------
        FullPipelineResponse with data for all three pages.
        """
        # --- Section 1 ---
        s1 = self.section1(posts)

        # --- Section 2 ---
        topics_as_dicts = [
            {
                "topic_id": t.topic_id,
                "headline": t.headline,
                "keywords": t.keywords,
            }
            for t in s1.topics
        ]
        s2 = self.section2(topics=topics_as_dicts)

        # --- Section 3 (optional) ---
        s3: Section3Response | None = None
        if new_posts:
            all_keywords = [kw for t in s1.topics for kw in t.keywords]
            bubble_embs  = getattr(self, "_last_pipeline_result", None)
            bubble_embs  = bubble_embs.embeddings if bubble_embs else None
            s3 = self.section3(
                new_posts=new_posts,
                bubble_keywords=all_keywords,
                bubble_embeddings=bubble_embs,
            )

        return FullPipelineResponse(section1=s1, section2=s2, section3=s3)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pipeline_result_to_topics(self, result: PipelineResult) -> list[TopicOut]:
        out = []
        for tr in result.topics:
            s = tr.summary
            out.append(TopicOut(
                topic_id=tr.topic_id,
                headline=s.headline   if s else "",
                category=s.category   if s else "Unknown",
                keywords=tr.keywords,
                key_points=s.key_points if s else [],
                n_posts=tr.n_posts,
                n_perspectives=tr.n_perspectives,
                representative_posts=tr.representative_posts,
            ))
        return out
