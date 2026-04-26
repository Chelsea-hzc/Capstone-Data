"""
Section 2 — Echo-chamber query builder.

Takes the headline/keywords produced by Section 1 and generates search
queries designed to surface diverse or opposing perspectives on the same
topic.  The full-stack backend sends these queries to the social-media
APIs (Twitter search, Reddit search, etc.) and streams the results back
for Section 3.

PLACEHOLDER — the strategy stubs below are intentional starting points.
Replace each `_strategy_*` method with your chosen approach once the
product direction is confirmed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class QueryIntent(str, Enum):
    OPPOSING    = "opposing"     # find counter-arguments / opposing views
    DIVERSE     = "diverse"      # find underrepresented angles
    RELATED     = "related"      # broaden the topic without forcing opposition
    FACTUAL     = "factual"      # find fact-check / news sources


@dataclass
class SearchQuery:
    """
    A single search query to be executed against a social-media platform.

    The full-stack backend iterates over these and calls the relevant API.
    """
    query_string: str
    platform: str                  # "twitter" | "reddit" | "any"
    intent: QueryIntent
    source_topic_id: int = -1
    source_keywords: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class QueryBuilderConfig:
    max_queries_per_topic: int = 3
    """Total queries emitted per topic (across all intents)."""

    intents: list[QueryIntent] = field(default_factory=lambda: [
        QueryIntent.OPPOSING,
        QueryIntent.DIVERSE,
        QueryIntent.RELATED,
    ])

    platforms: list[str] = field(default_factory=lambda: ["twitter", "reddit"])

    use_llm: bool = False
    """
    When True, delegate query generation to the attached summarizer/LLM.
    When False, use the keyword-inversion heuristic (no API key needed).
    """


class QueryBuilder:
    """
    Section 2 entry-point.

    Parameters
    ----------
    config:
        QueryBuilderConfig controlling query count, intents, and strategy.
    summarizer:
        Optional BaseSummarizer — used when config.use_llm is True to let
        the LLM craft richer, context-aware queries.

    Example
    -------
    >>> from social_topic_miner.echo_breaker import QueryBuilder
    >>> qb = QueryBuilder()
    >>> queries = qb.build(
    ...     topic_id=0,
    ...     headline="AI companies cut safety teams amid rapid deployment",
    ...     keywords=["safety", "AI", "deployment", "teams"],
    ... )
    >>> for q in queries:
    ...     print(q.intent, "→", q.query_string)
    """

    def __init__(
        self,
        config: QueryBuilderConfig | None = None,
        summarizer=None,          # BaseSummarizer — typed loosely to avoid circular import
    ) -> None:
        self.config = config or QueryBuilderConfig()
        self.summarizer = summarizer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        topic_id: int,
        headline: str,
        keywords: list[str],
    ) -> list[SearchQuery]:
        """
        Generate search queries for a single topic.

        Parameters
        ----------
        topic_id:   numeric ID from Section 1
        headline:   LLM-generated headline from Section 1
        keywords:   c-TF-IDF keywords from Section 1
        """
        if self.config.use_llm and self.summarizer is not None:
            queries = self._strategy_llm(topic_id, headline, keywords)
        else:
            queries = self._strategy_keyword_inversion(topic_id, headline, keywords)

        logger.info("Topic %d → %d queries generated", topic_id, len(queries))
        return queries[: self.config.max_queries_per_topic]

    def build_batch(
        self,
        topics: list[dict],
    ) -> dict[int, list[SearchQuery]]:
        """
        Run build() for every topic in a Section 1 result batch.

        Parameters
        ----------
        topics:
            List of dicts with keys ``topic_id``, ``headline``, ``keywords``.
            This is the shape returned by api.section1().
        """
        return {
            t["topic_id"]: self.build(
                topic_id=t["topic_id"],
                headline=t.get("headline", ""),
                keywords=t.get("keywords", []),
            )
            for t in topics
        }

    # ------------------------------------------------------------------
    # Strategy: keyword inversion (heuristic, no LLM needed)
    # ------------------------------------------------------------------

    def _strategy_keyword_inversion(
        self,
        topic_id: int,
        headline: str,
        keywords: list[str],
    ) -> list[SearchQuery]:
        """
        PLACEHOLDER — simple keyword-based query generation.

        Current logic:
        - OPPOSING  : top keyword + negation / counter phrases
        - DIVERSE   : headline as a plain search string (broad)
        - RELATED   : keyword pairs to surface adjacent discussions

        TODO: replace with a proper semantic approach once direction is set.
        """
        queries: list[SearchQuery] = []
        top_kw = keywords[:3] if keywords else [headline[:50]]

        # --- OPPOSING ------------------------------------------------
        if QueryIntent.OPPOSING in self.config.intents:
            opposing_q = f'"{top_kw[0]}" -"{top_kw[0]}" OR "against {top_kw[0]}" OR "criticism {top_kw[0]}"'
            queries.append(SearchQuery(
                query_string=opposing_q,
                platform="any",
                intent=QueryIntent.OPPOSING,
                source_topic_id=topic_id,
                source_keywords=top_kw,
            ))

        # --- DIVERSE -------------------------------------------------
        if QueryIntent.DIVERSE in self.config.intents:
            diverse_q = " OR ".join(f'"{kw}"' for kw in top_kw)
            queries.append(SearchQuery(
                query_string=diverse_q,
                platform="reddit",        # Reddit surfaces more long-form diverse views
                intent=QueryIntent.DIVERSE,
                source_topic_id=topic_id,
                source_keywords=top_kw,
            ))

        # --- RELATED -------------------------------------------------
        if QueryIntent.RELATED in self.config.intents:
            related_q = " ".join(top_kw[:2]) + " perspective"
            queries.append(SearchQuery(
                query_string=related_q,
                platform="twitter",
                intent=QueryIntent.RELATED,
                source_topic_id=topic_id,
                source_keywords=top_kw,
            ))

        return queries

    # ------------------------------------------------------------------
    # Strategy: LLM-generated queries
    # ------------------------------------------------------------------

    def _strategy_llm(
        self,
        topic_id: int,
        headline: str,
        keywords: list[str],
    ) -> list[SearchQuery]:
        """
        PLACEHOLDER — ask the LLM to craft targeted search queries.

        TODO: implement once prompt design is finalised.
              The summarizer.summarize() interface will need a new method
              (e.g. summarizer.generate_queries()) or we call it directly.
        """
        logger.warning(
            "LLM query strategy is not yet implemented — "
            "falling back to keyword inversion for topic %d",
            topic_id,
        )
        return self._strategy_keyword_inversion(topic_id, headline, keywords)
