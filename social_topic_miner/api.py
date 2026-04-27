"""
Public API — the full-stack backend calls these three methods.

  api.section1(posts)            → Section1Response
  api.section2(request)          → Section2Response
  api.section3(request)          → Section3Response
  api.run_full(posts, new_posts) → FullPipelineResponse

Import the TypedDicts from social_topic_miner.types to validate shapes.
"""

from __future__ import annotations

import logging
from dataclasses import asdict

import numpy as np

from .config import TopicMinerConfig
from .diversity.filter import DiversityFilter, DiversityFilterConfig
from .echo_breaker.query_builder import QueryBuilder, QueryBuilderConfig
from .embedders.base import BaseEmbedder
from .embedders.sentence_transformer import SentenceTransformerEmbedder
from .pipeline import PipelineResult, TopicMinerPipeline
from .summarizers.base import BaseSummarizer
from .types import (
    FullPipelineResponse,
    FullPipelineRequest,
    PostIn,
    Section1Response,
    Section2Request,
    Section2Response,
    Section3Request,
    Section3Response,
    TopicOut,
)

logger = logging.getLogger(__name__)


class TopicMinerAPI:
    """
    Stateless façade over the three-section pipeline.

    Instantiate once at app startup and reuse across requests.

    Parameters
    ----------
    config:
        Master TopicMinerConfig.  All defaults are production-ready.
    embedder:
        Embedding model shared across all sections.
        Defaults to SentenceTransformer("all-MiniLM-L6-v2").
    summarizer:
        Optional LLM for Section 1 headline/category/key_points generation.
        When None, those fields are returned as empty strings/lists.
    query_config:
        Knobs for Section 2 query generation.
    diversity_config:
        Knobs for Section 3 diversity scoring and cutoff.

    Example
    -------
    >>> from social_topic_miner import TopicMinerAPI
    >>> api = TopicMinerAPI()
    >>> response = api.section1(posts)          # list[PostIn]
    >>> queries  = api.section2({"topics": response["topics"]})
    >>> filtered = api.section3({"new_posts": fetched_posts,
    ...                          "bubble_keywords": response["topics"][0]["keywords"]})
    """

    def __init__(
        self,
        config: TopicMinerConfig | None = None,
        embedder: BaseEmbedder | None = None,
        summarizer: BaseSummarizer | None = None,
        query_config: QueryBuilderConfig | None = None,
        diversity_config: DiversityFilterConfig | None = None,
    ) -> None:
        self._embedder    = embedder or SentenceTransformerEmbedder()
        self._summarizer  = summarizer
        self._pipeline    = TopicMinerPipeline(
            config=config,
            embedder=self._embedder,
            summarizer=summarizer,
        )
        self._qbuilder    = QueryBuilder(config=query_config, summarizer=summarizer)
        self._dfilter     = DiversityFilter(config=diversity_config, embedder=self._embedder)
        self._last_result: PipelineResult | None = None

    # ------------------------------------------------------------------
    # Section 1
    # ------------------------------------------------------------------

    def section1(self, posts: list[PostIn]) -> Section1Response:
        """
        Cluster a user's feed and return topic summaries.

        Parameters
        ----------
        posts : list[PostIn]
            Raw social-media posts.  Each item must have at minimum:
              - "text"     : str
              - "platform" : "twitter" | "reddit"
            See social_topic_miner.types.PostIn for all supported fields.

        Returns
        -------
        Section1Response
            {
              "topics": [
                {
                  "topic_id": int,
                  "headline": str,
                  "category": str,
                  "keywords": list[str],
                  "key_points": list[str],
                  "n_posts": int,
                  "n_perspectives": int,
                  "representative_posts": [...]
                },
                ...
              ],
              "total_posts_processed": int
            }
        """
        import pandas as pd

        df_raw = pd.DataFrame(posts)
        result = self._pipeline.run_from_dataframe(df_raw)
        self._last_result = result

        topics = self._to_topic_out_list(result)
        digest = self._build_digest(result)

        return Section1Response(
            topics=topics,
            total_posts_processed=len(result.df),
            digest=digest,
        )

    # ------------------------------------------------------------------
    # Section 2
    # ------------------------------------------------------------------

    def section2(self, request: Section2Request) -> Section2Response:
        """
        Generate echo-breaking search queries from Section 1 output.

        Parameters
        ----------
        request : Section2Request
            Pass ONE of:
            - {"topics": <list of TopicOut from section1()>}   ← preferred
            - {"headlines": [...], "keywords": [...]}           ← manual override

        Returns
        -------
        Section2Response
            {
              "queries": [
                {
                  "query_string": str,
                  "platform":     "twitter" | "reddit" | "any",
                  "intent":       "opposing" | "diverse" | "related",
                  "source_topic_id": int,
                  "source_keywords": list[str],
                  "metadata": {}
                },
                ...
              ],
              "source_topic_ids": list[int]
            }

        Note
        ----
        The query_string values are PLACEHOLDERS.  The logic will be
        improved once the product direction for echo-chamber breaking is
        confirmed.  The schema will not change.
        """
        topic  = request.get("topic")   # single-topic path (priority)
        topics = request.get("topics")  # batch path
        if topic:
            query_map = self._qbuilder.build_batch([topic])
        elif topics:
            # Pass full TopicOut dicts so build_batch() can use all fields
            # (keywords, key_points, representative_posts, summaries) for expansion.
            query_map = self._qbuilder.build_batch(list(topics))
        else:
            headlines = request.get("headlines") or [""]
            keywords  = request.get("keywords") or []
            query_map = {
                0: self._qbuilder.build(
                    topic_id=0,
                    headline=headlines[0],
                    keywords=keywords,
                )
            }

        # Flatten and sort by probability descending so the backend sends
        # highest-confidence queries first.
        all_queries = sorted(
            [asdict(q) for qs in query_map.values() for q in qs],
            key=lambda q: -q["probability"],
        )
        return Section2Response(
            queries=all_queries,
            source_topic_ids=list(query_map.keys()),
        )

    # ------------------------------------------------------------------
    # Section 3
    # ------------------------------------------------------------------

    def section3(self, request: Section3Request) -> Section3Response:
        """
        Filter and rank diverse posts fetched via Section 2 queries.

        Parameters
        ----------
        request : Section3Request
            {
              "new_posts":       list[PostIn],   # required
              "bubble_keywords": list[str]       # optional, improves scoring
            }

        Returns
        -------
        Section3Response
            {
              "posts":   list[PostIn],   # filtered, sorted by diversity score desc
              "scores":  list[float],    # 0-1 diversity score per post
              "dropped": int             # posts removed by cutoff threshold
            }

        Note
        ----
        Scoring is a PLACEHOLDER.  The current implementation uses keyword
        overlap (relevance) and uniform 0.5 (divergence).  Both will be
        replaced with embedding-based and/or stance-detection approaches.
        The schema will not change.
        """
        new_posts       = request.get("new_posts", [])
        bubble_keywords = request.get("bubble_keywords", [])

        bubble_embs: np.ndarray | None = None
        if self._last_result is not None:
            bubble_embs = self._last_result.embeddings

        result = self._dfilter.filter(
            new_posts=new_posts,
            bubble_keywords=bubble_keywords,
            bubble_embeddings=bubble_embs,
        )
        return Section3Response(
            balanced=result.balanced,
            balanced_scores=result.balanced_scores,
            other=result.other,
            other_scores=result.other_scores,
            dropped=result.dropped,
        )

    # ------------------------------------------------------------------
    # Combined — all three sections in one call
    # ------------------------------------------------------------------

    def run_full(self, request: FullPipelineRequest) -> FullPipelineResponse:
        """
        Run all three sections in sequence and return data for all pages.

        Parameters
        ----------
        request : FullPipelineRequest
            {
              "posts":     list[PostIn],   # required — user's feed
              "new_posts": list[PostIn]    # optional — pre-fetched diverse posts
            }

        Returns
        -------
        FullPipelineResponse
            {
              "section1": Section1Response,
              "section2": Section2Response,
              "section3": Section3Response | None   # None when new_posts omitted
            }
        """
        posts     = request.get("posts", [])
        new_posts = request.get("new_posts")

        s1 = self.section1(posts)

        s2 = self.section2({"topics": s1["topics"]})

        s3: Section3Response | None = None
        if new_posts:
            all_keywords = [kw for t in s1["topics"] for kw in t["keywords"]]
            s3 = self.section3({
                "new_posts": new_posts,
                "bubble_keywords": all_keywords,
            })

        return FullPipelineResponse(section1=s1, section2=s2, section3=s3)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_topic_out_list(self, result: PipelineResult) -> list[TopicOut]:
        out = []
        for tr in result.topics:
            s = tr.summary
            out.append(TopicOut(
                topic_id=tr.topic_id,
                headline=s.headline       if s else "",
                category=s.category       if s else "Unknown",
                short_summary=s.short_summary if s else "",
                long_summary=s.long_summary   if s else "",
                keywords=tr.keywords,
                key_points=s.key_points   if s else [],
                n_posts=tr.n_posts,
                n_perspectives=tr.n_perspectives,
                representative_posts=tr.representative_posts,
            ))
        return out

    def _build_digest(self, result: PipelineResult) -> str:
        """Generate the overall feed digest across all topics."""
        topic_summaries = [tr.summary for tr in result.topics if tr.summary is not None]
        if self._summarizer is not None:
            try:
                return self._summarizer.summarize_digest(topic_summaries)
            except Exception:
                logger.warning("Digest generation failed — falling back to headline concat")
        # Fallback: join headlines when no summarizer or on error
        headlines = [tr.summary.headline for tr in result.topics if tr.summary and tr.summary.headline]
        return " | ".join(headlines)
