"""
TopicMinerPipeline — end-to-end orchestrator.

Typical usage
-------------
from social_topic_miner import TopicMinerPipeline, TopicMinerConfig
from social_topic_miner.embedders import SentenceTransformerEmbedder
from social_topic_miner.summarizers import OpenAISummarizer

pipeline = TopicMinerPipeline(
    config=TopicMinerConfig(),
    embedder=SentenceTransformerEmbedder(),
    summarizer=OpenAISummarizer(api_key="sk-..."),
)
result = pipeline.run_from_json("timeline.json")
for topic in result.topics:
    print(topic.summary.headline)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from .clustering import SubPartitioner, TopicClusterer
from .config import TopicMinerConfig
from .embedders.base import BaseEmbedder
from .embedders.sentence_transformer import SentenceTransformerEmbedder
from .preprocessing import PostNormalizer
from .selection import EngagementScorer, PostSampler
from .summarizers.base import BaseSummarizer, TopicSummary

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TopicResult:
    topic_id: int
    keywords: list[str]
    n_posts: int
    n_perspectives: int
    representative_posts: list[dict]
    """List of dicts with keys: text, platform, sub_perspective, engagement_norm."""
    summary: TopicSummary | None = None


@dataclass
class PipelineResult:
    topics: list[TopicResult]
    df: pd.DataFrame
    """Full preprocessed DataFrame with topic_id and sub_perspective columns."""
    embeddings: np.ndarray
    run_at: datetime = field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class TopicMinerPipeline:
    """
    Orchestrates: load → preprocess → embed → cluster → score → partition
                  → sample → summarise.

    Every stage is swappable — pass your own embedder, summarizer, or subclass
    any component.

    Parameters
    ----------
    config:
        Master config dataclass; individual sub-configs can be overridden.
    embedder:
        Any object implementing ``BaseEmbedder.embed(texts) -> np.ndarray``.
        Defaults to ``SentenceTransformerEmbedder`` with "all-MiniLM-L6-v2".
    summarizer:
        Optional BaseSummarizer.  If None, summaries are skipped.
    normalizer:
        Optional PostNormalizer (defaults to one built from config).
    """

    def __init__(
        self,
        config: TopicMinerConfig | None = None,
        embedder: BaseEmbedder | None = None,
        summarizer: BaseSummarizer | None = None,
        normalizer: PostNormalizer | None = None,
    ) -> None:
        self.config = config or TopicMinerConfig()
        self.embedder = embedder or SentenceTransformerEmbedder(self.config.embedder)
        self.summarizer = summarizer
        self.normalizer = normalizer or PostNormalizer(self.config.preprocessing)

        self._clusterer = TopicClusterer(self.config.clustering)
        self._partitioner = SubPartitioner(self.config.partition)
        self._scorer = EngagementScorer(self.config.selection)
        self._sampler = PostSampler(self.config.sampling)

    # ------------------------------------------------------------------
    # Public entry-points
    # ------------------------------------------------------------------

    def run_from_json(self, path: str) -> PipelineResult:
        """Load a raw timeline JSON file and run the full pipeline."""
        logger.info("Loading posts from %s", path)
        df = self.normalizer.from_json(path)
        return self._run(df)

    def run_from_dataframe(self, df: pd.DataFrame) -> PipelineResult:
        """Accept an already-loaded raw DataFrame and run the full pipeline."""
        df = self.normalizer.from_dataframe(df)
        return self._run(df)

    # ------------------------------------------------------------------
    # Internal stages
    # ------------------------------------------------------------------

    def _run(self, df: pd.DataFrame) -> PipelineResult:
        logger.info("Posts after preprocessing: %d", len(df))

        # 1. Embed
        docs = df["text"].tolist()
        logger.info("Embedding %d docs …", len(docs))
        embeddings = self.embedder.embed(docs)

        # 2. Cluster
        cluster_result = self._clusterer.fit(docs, embeddings)
        df["topic_id"] = cluster_result.topic_ids

        # 3. Score & select top-N topics
        df = self._scorer.add_engagement_columns(df)
        selected_ids = self._scorer.top_topic_ids(df)
        logger.info("Selected topics: %s", selected_ids)

        # 4. Sub-perspective partitioning
        df = self._partitioner.partition(df, embeddings, selected_ids)

        # 5. Sample representative posts
        sampled_topics = self._sampler.sample(df, embeddings, selected_ids)
        logger.info("%s Sampled Topics : %s", len(sampled_topics),sampled_topics) 

        # 6. Assemble results (+ optional summarisation)
        topic_results: list[TopicResult] = []
        for st in sampled_topics:
            tid = st.topic_id
            keywords = self._clusterer.get_keywords(tid)
            posts_rows = [
                {
                    "text": df.loc[idx, "text"],
                    "platform": df.loc[idx, "platform"],
                    "sub_perspective": int(df.loc[idx, "sub_perspective"]),
                    "engagement_norm": float(df.loc[idx, "engagement_norm"]),
                    "post_id": df.loc[idx, "post_id"],
                    "created_at": str(df.loc[idx, "created_at"]),
                }
                for idx in st.selected_indices
            ]

            logger.info("For Sampled Topic id %s: keyword is %s and posts are %s ", tid, keywords, posts_rows) 
            # logger.info("Use summarize %s", self.summarizer)
            summary: TopicSummary | None = None
            if self.summarizer is not None:
                logger.info("Call summarize %s for topic %s", self.summarizer, tid)
                try:
                    summary = self.summarizer.summarize(
                        topic_id=tid,
                        posts=[r["text"] for r in posts_rows],
                        keywords=keywords,
                    )
                    logger.info("Summary %s ", summary)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Summarisation failed for topic %d: %s", tid, exc)

            topic_results.append(TopicResult(
                topic_id=tid,
                keywords=keywords,
                n_posts=int((df["topic_id"] == tid).sum()),
                n_perspectives=st.n_perspectives,
                representative_posts=posts_rows,
                summary=summary,
            ))

        return PipelineResult(
            topics=topic_results,
            df=df,
            embeddings=embeddings,
        )

    def partition_and_summarize(
    self,
    df: pd.DataFrame,
    embeddings: np.ndarray,
    topic_id: int = 0,
    keywords: list[str] | None = None,
) -> TopicResult:
    """
    Run partition → sample → summarize on a set of posts that already
    belong to one topic.

    Used by:
      - Section 1's _run() for each selected topic (after clustering)
      - Section 3 for the diversity-filtered posts of a single topic

    Parameters
    ----------
    df:
        DataFrame of posts, already cleaned/normalized. Must contain the
        engagement and platform columns the pipeline expects.
    embeddings:
        Embedding matrix aligned with df (one row per post).
    topic_id:
        Topic id to assign to all posts (defaults to 0 for single-topic
        callers like Section 3).
    keywords:
        Optional keywords for this topic. Section 1 passes c-TF-IDF keywords;
        Section 3 can pass [] or the bubble keywords.

    Returns
    -------
    TopicResult
        Same shape as the elements in PipelineResult.topics.
    """
    df = df.copy().reset_index(drop=True)
    df["topic_id"] = topic_id

    # Ensure engagement columns exist (Section 3 input may not have them yet)
    if "engagement_norm" not in df.columns:
        df = self._scorer.add_engagement_columns(df)

    # Partition this single topic into sub-perspectives
    df = self._partitioner.partition(df, embeddings, [topic_id])

    # Sample representative posts
    sampled = self._sampler.sample(df, embeddings, [topic_id])
    if not sampled:
        # No posts selected — return an empty result
        return TopicResult(
            topic_id=topic_id,
            keywords=keywords or [],
            n_posts=len(df),
            n_perspectives=0,
            representative_posts=[],
            summary=None,
        )

    st = sampled[0]
    posts_rows = [
        {
            "text": df.loc[idx, "text"],
            "platform": df.loc[idx, "platform"],
            "sub_perspective": int(df.loc[idx, "sub_perspective"]),
            "engagement_norm": float(df.loc[idx, "engagement_norm"]),
            "post_id": df.loc[idx, "post_id"],
            "created_at": str(df.loc[idx, "created_at"]),
        }
        for idx in st.selected_indices
    ]

    summary: TopicSummary | None = None
    if self.summarizer is not None:
        try:
            summary = self.summarizer.summarize(
                topic_id=topic_id,
                posts=[r["text"] for r in posts_rows],
                keywords=keywords or [],
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Summarisation failed for topic %d: %s", topic_id, exc)

    return TopicResult(
        topic_id=topic_id,
        keywords=keywords or [],
        n_posts=len(df),
        n_perspectives=st.n_perspectives,
        representative_posts=posts_rows,
        summary=summary,
    )
    
    # ------------------------------------------------------------------
    # Convenience: pretty-print results
    # ------------------------------------------------------------------

    def display(self, result: PipelineResult) -> None:
        sep = "─" * 60
        print(f"\n{'=' * 60}")
        print(f"RUN COMPLETE  |  {result.run_at.strftime('%Y-%m-%d %H:%M UTC')}")
        print(f"Total topics selected: {len(result.topics)}")
        print(f"{'=' * 60}")

        for tr in result.topics:
            print(f"\n{sep}")
            print(f"Topic {tr.topic_id}  ({tr.n_posts} posts, {tr.n_perspectives} perspectives)")
            print(f"Keywords : {', '.join(tr.keywords)}")
            if tr.summary:
                print(f"Category : {tr.summary.category}")
                print(f"Headline : {tr.summary.headline}")
                for kp in tr.summary.key_points:
                    print(f"  • {kp}")
            print(f"{sep}")
            for post in tr.representative_posts:
                preview = post["text"][:140]
                print(
                    f"  [sub={post['sub_perspective']}]"
                    f" [{post['platform']:7s}]"
                    f" eng={post['engagement_norm']:.2f}"
                    f"  {preview}…"
                )
        print()
