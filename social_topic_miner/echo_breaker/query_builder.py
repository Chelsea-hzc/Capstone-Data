"""
Section 2 — Echo-chamber query builder.

Takes the headline/keywords produced by Section 1 and generates search
queries designed to surface diverse or opposing perspectives on the same
topic.  The full-stack backend sends these queries to the social-media
APIs (Twitter search, Reddit search, etc.) and streams the results back
for Section 3.

Strategy (implemented):
  1. Extract anchor terms from the headline using spaCy NER (falls back
     to the top two keywords when spaCy is not installed).
  2. Generate one query per stance bucket (neutral, supportive, critical,
     emotional, industry) using the STANCE_DICT taxonomy.
  3. Hard-cap OR blocks at MAX_OR_TERMS = 5 (longer blocks return 0 results
     on the Twitter v2 recent-search endpoint).

Query shape (Twitter v2 syntax):
    (anchor1 OR anchor2) (bridge1 OR bridge2) (stance1 OR stance2) -is:retweet lang:en

LLM strategy stub:
  When config.use_llm is True and a summarizer is attached, the builder
  calls the LLM to produce anchor_terms + bridge_terms JSON, then wraps
  those into the same per-stance structure.  Falls back to NER if the
  LLM call fails or returns unusable output.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_OR_TERMS = 4
"""Hard cap on terms inside a single OR block.  Twitter v2 returns 0 results
for OR blocks longer than ~6 terms; 4 gives a safe margin."""

MAX_QUERY_LENGTH = 350
"""Twitter v2 query character limit."""

_STOPWORDS = {
    # Articles / determiners
    "a", "an", "the", "this", "that", "these", "those", "some", "any", "each",
    "every", "both", "all", "few", "more", "most", "other", "such", "no",
    # Pronouns
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
    "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
    # Prepositions
    "in", "on", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below", "to",
    "from", "up", "down", "out", "off", "over", "under", "again", "further",
    "then", "once", "upon", "within", "without", "among", "along", "around",
    "onto", "toward", "towards", "beyond", "behind", "beside", "across",
    # Conjunctions
    "and", "but", "or", "nor", "so", "yet", "for", "if", "as", "than",
    "because", "since", "while", "although", "though", "unless", "until",
    "when", "where", "whether", "either", "neither", "both", "not", "also",
    "however", "therefore", "otherwise", "meanwhile", "furthermore",
    # Auxiliary / modal verbs
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may", "might",
    "must", "shall", "can", "need", "dare", "ought", "used",
    # Common adverbs
    "very", "really", "just", "still", "also", "only", "even", "already",
    "always", "never", "often", "sometimes", "usually", "here", "there",
    "now", "then", "too", "quite", "rather", "much", "many", "more", "most",
    "less", "least", "well", "back", "away", "ever", "soon", "later",
    # Common short/noise words from social media / BERTopic
    "vs", "pro", "via", "per", "etc", "new", "one", "two", "yes", "non",
    "get", "got", "say", "said", "says", "make", "made", "take", "took",
    "come", "came", "give", "gave", "know", "knew", "look", "like", "want",
    "going", "come", "think", "seen", "told", "goes", "went", "let", "put",
    "im", "ive", "its", "isnt", "dont", "doesnt", "cant", "wont", "wasnt",
    "weren", "hasnt", "havent", "wouldnt", "shouldnt", "couldnt",
    # Common generic nouns / process words
    "thing", "things", "time", "year", "years", "month", "week", "day",
    "people", "person", "part", "place", "point", "case", "fact", "number",
    "way", "use", "using", "used", "work", "works", "working", "based",
    "including", "according", "following", "regarding", "related", "using",
    "approval", "combined", "compliance", "entity", "filing", "foreign",
    "funding", "merger", "needed", "ownership", "percent", "process",
    "proposed", "regulatory", "required", "requesting", "review", "various",
}

# Stance taxonomy derived from the query-expansion experiments notebook.
# 10 categories, 279 total terms.  The five PRIMARY stances map to QueryIntent.
STANCE_DICT: dict[str, list[str]] = {
    # PRIMARY STANCES (each produces one query)
    "neutral": [
        "discusses", "explains", "examines", "explores", "reviews",
        "analyzes", "reports", "covers", "addresses", "considers",
        "investigates", "studies", "evaluates", "assesses", "describes",
    ],
    "supportive": [
        "supports", "advocates", "promotes", "endorses", "champions",
        "praises", "celebrates", "welcomes", "applauds", "approves",
        "backs", "defends", "justifies", "validates", "affirms",
    ],
    "critical": [
        "criticizes", "opposes", "challenges", "questions", "disputes",
        "rejects", "condemns", "attacks", "blames", "accuses",
        "warns", "concerns", "problems", "issues", "failures",
    ],
    "emotional": [
        "outrage", "anger", "frustration", "disappointment", "shock",
        "fear", "anxiety", "worry", "upset", "alarmed",
        "disgusted", "appalled", "devastated", "heartbroken", "terrified",
    ],
    "industry": [
        "industry", "market", "business", "sector", "economy",
        "investment", "revenue", "profit", "growth", "strategy",
        "company", "corporate", "enterprise", "commercial", "financial",
    ],
    # SECONDARY STANCES (pooled into queries only when max_queries_per_topic > 5)
    "analytical": [
        "data", "research", "study", "statistics", "evidence",
        "findings", "results", "analysis", "metrics", "trends",
        "survey", "report", "paper", "journal", "academic",
    ],
    "broader": [
        "context", "background", "history", "perspective", "overview",
        "global", "international", "national", "regional", "local",
        "society", "community", "culture", "environment", "impact",
    ],
    "humor": [
        "funny", "lol", "meme", "joke", "hilarious",
        "satire", "irony", "sarcasm", "comedy", "parody",
        "laugh", "ridiculous", "absurd", "bizarre", "viral",
    ],
    "speculation": [
        "rumor", "speculation", "theory", "prediction", "forecast",
        "possibility", "potential", "future", "might", "could",
        "would", "should", "expected", "anticipated", "projected",
    ],
    "community": [
        "community", "people", "public", "users", "fans",
        "audience", "followers", "supporters", "members", "group",
        "movement", "activists", "advocates", "protesters", "citizens",
    ],
}

_PRIMARY_STANCES = ["neutral", "supportive", "critical", "emotional", "industry"]

# Twitter-specific suffix appended to every generated query
# _TWITTER_FILTER = "-is:retweet lang:en"


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class QueryIntent(str, Enum):
    OPPOSING    = "opposing"     # counter-arguments / opposing views
    DIVERSE     = "diverse"      # underrepresented angles
    RELATED     = "related"      # broaden without forcing opposition
    FACTUAL     = "factual"      # fact-check / news sources

# Map primary stance → QueryIntent
_STANCE_TO_INTENT: dict[str, QueryIntent] = {
    "neutral":    QueryIntent.RELATED,
    "supportive": QueryIntent.DIVERSE,
    "critical":   QueryIntent.OPPOSING,
    "emotional":  QueryIntent.DIVERSE,
    "industry":   QueryIntent.RELATED,
    "analytical": QueryIntent.FACTUAL,
    "broader":    QueryIntent.DIVERSE,
    "humor":      QueryIntent.DIVERSE,
    "speculation":QueryIntent.DIVERSE,
    "community":  QueryIntent.DIVERSE,
}


@dataclass
class SearchQuery:
    """A single search query to be executed against a social-media platform."""
    query_string: str
    platform: str                  # "twitter" | "reddit" | "any"
    intent: QueryIntent
    probability: float = 0.5
    """Estimated probability (0-1) that this query returns diverse, useful results.
    Higher = send first.  Based on stance type and anchor quality."""
    source_topic_id: int = -1
    source_keywords: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


# Estimated probability of finding diverse perspectives per stance bucket,
# based on query-expansion experiments.
_STANCE_PROBABILITY: dict[str, float] = {
    "critical":   0.88,
    "emotional":  0.78,
    "supportive": 0.65,
    "broader":    0.62,
    "neutral":    0.58,
    "analytical": 0.55,
    "industry":   0.52,
    "community":  0.50,
    "speculation":0.48,
    "humor":      0.40,
}


@dataclass
class QueryBuilderConfig:
    n_twitter_queries: int = 5
    """Exact number of Twitter queries to emit per topic (one per primary stance)."""

    n_reddit_queries: int = 3
    """Exact number of Reddit queries to emit per topic (cycles headline then key_points)."""

    intents: list[QueryIntent] = field(default_factory=lambda: [
        QueryIntent.OPPOSING,
        QueryIntent.DIVERSE,
        QueryIntent.RELATED,
        QueryIntent.FACTUAL,
    ])

    platforms: list[str] = field(default_factory=lambda: ["twitter", "reddit"])

    use_llm: bool = False
    """
    When True, use the attached summarizer to generate anchor + bridge terms.
    When False, extract anchors from the headline via spaCy NER (or fallback
    to top keywords), then pair with STANCE_DICT terms directly.
    """

    add_twitter_filters: bool = False
    """Append ``-is:retweet lang:en`` to Twitter queries."""

    primary_stances: list[str] = field(
        default_factory=lambda: list(_PRIMARY_STANCES)
    )
    """Which STANCE_DICT buckets generate queries.  Order = query order."""


# ---------------------------------------------------------------------------
# Query assembly helpers
# ---------------------------------------------------------------------------

def _or_block(terms: list[str], max_terms: int = MAX_OR_TERMS) -> str:
    """Return ``(t1 OR t2 OR ...)`` capped at *max_terms*."""
    selected = terms[:max_terms]
    if len(selected) == 1:
        return selected[0]
    return "(" + " OR ".join(selected) + ")"


def build_query(
    anchor_terms: list[str],
    bridge_terms: list[str],
    stance_terms: list[str],
    platform: str = "twitter",
    add_filters: bool = True,
) -> str:
    """
    Assemble a platform query from three term groups.

    Pattern: ``(anchors) (bridge) (stance) [platform_filters]``

    Each group is hard-capped at MAX_OR_TERMS before joining.
    Empty groups are omitted so the query stays valid.

    Parameters
    ----------
    anchor_terms:  named entities / specific nouns — keep the query on-topic
    bridge_terms:  topic keywords that link anchors to stances
    stance_terms:  verbs / adjectives indicating the stance angle
    platform:      "twitter" | "reddit" | "any"
    add_filters:   append -is:retweet lang:en (Twitter only)
    """
    blocks: list[str] = []

    if anchor_terms:
        blocks.append(_or_block(anchor_terms))
    if bridge_terms:
        blocks.append(_or_block(bridge_terms))
    if stance_terms:
        blocks.append(_or_block(stance_terms))

    query = " ".join(blocks)

    # if add_filters and platform == "twitter":
    #     query = f"{query} {_TWITTER_FILTER}"

    return query.strip()


# ---------------------------------------------------------------------------
# Anchor extraction
# ---------------------------------------------------------------------------

_ANCHOR_ENTITY_TYPES = {"ORG", "PERSON", "GPE", "PRODUCT", "EVENT", "NORP"}


def _extract_anchors_spacy(text: str) -> list[str]:
    """
    Extract named entities from *text* using spaCy.

    Only keeps entity types useful as search anchors (ORG, PERSON, GPE,
    PRODUCT, EVENT, NORP) and drops entities longer than 2 words.
    Falls back to an empty list if spaCy is not installed or the model
    is not available (caller then uses keyword fallback).
    """
    try:
        import spacy  # optional dependency
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            return []
        doc = nlp(text)
        seen: set[str] = set()
        anchors: list[str] = []
        for ent in doc.ents:
            if ent.label_ not in _ANCHOR_ENTITY_TYPES:
                continue
            raw = ent.text.strip().strip("'\"'‘’“”")
            if not raw or len(raw.split()) > 2:
                continue
            if raw.lower() not in seen:
                seen.add(raw.lower())
                anchors.append(raw)
            logger.info("spaCy anchor ent %s and raw %s", ent, raw)
        logger.info("spaCy anchor entities: %s", anchors)
        return anchors
    except ImportError:
        return []


def _extract_anchors(
    headline: str,
    keywords: list[str],
    long_summary: str = "",
) -> list[str]:
    """Return NER anchors from headline + long_summary, falling back to top keywords."""
    anchors = _extract_anchors_spacy(headline)

    # Also run NER on long_summary to pick up entities missed in the headline
    if long_summary:
        seen = {a.lower() for a in anchors}
        for ent in _extract_anchors_spacy(long_summary):
            if ent.lower() not in seen:
                seen.add(ent.lower())
                anchors.append(ent)

    if not anchors:
        # Fallback: treat the top two title-case words in keywords as anchors
        anchors = [kw for kw in keywords[:4] if kw and kw[0].isupper()]
        if not anchors:
            anchors = keywords[:2]
    return anchors[:MAX_OR_TERMS]


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

class QueryBuilder:
    """
    Section 2 entry-point.

    Generates one search query per primary stance bucket (neutral,
    supportive, critical, emotional, industry), capped at
    ``config.max_queries_per_topic``.

    Parameters
    ----------
    config:
        QueryBuilderConfig controlling query count, stances, and strategy.
    summarizer:
        Optional BaseSummarizer — used when config.use_llm is True to let
        the LLM produce anchor + bridge terms for richer queries.

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
    ...     print(q.intent.value, "→", q.query_string)
    """

    def __init__(
        self,
        config: QueryBuilderConfig | None = None,
        summarizer=None,
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
        key_points: list[str] | None = None,
        representative_posts: list[dict] | None = None,
        short_summary: str = "",
        long_summary: str = "",
    ) -> list[SearchQuery]:
        """
        Generate search queries for a single topic.

        Parameters
        ----------
        topic_id:              numeric ID from Section 1
        headline:              LLM-generated headline
        keywords:              c-TF-IDF keywords from BERTopic
        key_points:            LLM-generated bullet points (used for keyword expansion)
        representative_posts:  sampled posts (reserved for future embedding-based expansion)
        short_summary:         1-2 sentence summary (reserved for future use)
        long_summary:          full paragraph summary (reserved for future use)
        """

        logger.info("Topic %d : keywords is %s keypoints is %s headline is %s ", topic_id, keywords, key_points, headline)

        expanded_keywords = self._expand_keywords(
            keywords=keywords,
            key_points=key_points,
        )

        logger.info("Expanded Keywords %s ", expanded_keywords)

        if self.config.use_llm and self.summarizer is not None:
            anchor_terms, bridge_terms = self._llm_anchor_bridge(
                topic_id, headline, expanded_keywords
            )
            logger.info("With LLM anchor terms are %s and bridge terms are %s", anchor_terms, bridge_terms)
        else:
            anchor_terms = [t for t in _extract_anchors(headline, expanded_keywords, long_summary) if t and t.strip()]
            bridge_terms = [kw for kw in expanded_keywords if kw not in anchor_terms and kw.strip()]
            logger.info("WITHOUT LLM anchor terms are %s and bridge terms are %s", anchor_terms, bridge_terms)

        queries = self._build_stance_queries(
            topic_id=topic_id,
            anchor_terms=anchor_terms,
            bridge_terms=bridge_terms,
            keywords=expanded_keywords,
            headline=headline,
            key_points=key_points,
        )

        # Twitter sorted by probability desc; Reddit appended after
        twitter_queries = sorted(
            [q for q in queries if q.platform == "twitter"],
            key=lambda q: -q.probability,
        )
        reddit_queries = [q for q in queries if q.platform == "reddit"]

        result = twitter_queries + reddit_queries
        logger.info(
            "Topic %d → %d twitter + %d reddit queries",
            topic_id, len(twitter_queries), len(reddit_queries),
        )
        return result

    def build_batch(
        self,
        topics: list[dict],
    ) -> dict[int, list[SearchQuery]]:
        """
        Run build() for every topic in a Section 1 result batch.

        Parameters
        ----------
        topics:
            List of TopicOut dicts (full Section 1 output per topic).
            Passes all available fields to build() so the expansion algorithm
            can use keywords, key_points, representative_posts, or summaries.
        """
        return {
            t["topic_id"]: self.build(
                topic_id=t["topic_id"],
                headline=t.get("headline", ""),
                keywords=t.get("keywords", []),
                key_points=t.get("key_points"),
                representative_posts=t.get("representative_posts"),
                short_summary=t.get("short_summary", ""),
                long_summary=t.get("long_summary", ""),
            )
            for t in topics
        }

    # ------------------------------------------------------------------
    # Keyword expansion
    # ------------------------------------------------------------------

    def _expand_keywords(
        self,
        keywords: list[str],
        key_points: list[str] | None = None,
    ) -> list[str]:
        """
        Expand the base keyword set using additional topic fields.

        Current strategy: extract content words (length > 4) from key_points.
        Swap this method to use embeddings, NER, or LLM paraphrase later.
        """
        expanded = [k for k in keywords if k and k.strip() and k.lower() not in _STOPWORDS]
        seen = {k.lower() for k in expanded}

        if key_points:
            for kp in key_points:
                for word in kp.split():
                    clean = word.strip('.,!?":;()[]').strip("'\"")
                    if (
                        len(clean) > 4
                        and clean.lower() not in seen
                        and clean.lower() not in _STOPWORDS
                    ):
                        seen.add(clean.lower())
                        expanded.append(clean)

        return [k for k in expanded if k and k.strip()][:20]

    # ------------------------------------------------------------------
    # Strategy: per-stance query generation
    # ------------------------------------------------------------------

    def _build_stance_queries(
        self,
        topic_id: int,
        anchor_terms: list[str],
        bridge_terms: list[str],
        keywords: list[str],
        headline: str = "",
        key_points: list[str] | None = None,
    ) -> list[SearchQuery]:
        """
        Generate one query per stance in config.primary_stances.

        Twitter: (anchors) (bridge_terms) (stance_terms) [filters]
        Reddit:  alternates between headline and key_points as plain-text queries.

        The primary platform alternates: twitter for odd stances, reddit
        for even stances, to spread coverage across platforms.
        """
        queries: list[SearchQuery] = []
        cfg = self.config
        add_filters = cfg.add_twitter_filters

        # --- Twitter: one query per primary stance, exactly n_twitter_queries ---
        stances = list(cfg.primary_stances)
        while len(stances) < cfg.n_twitter_queries:
            stances += list(cfg.primary_stances)

        for stance in stances[:cfg.n_twitter_queries]:
            stance_terms = STANCE_DICT.get(stance, [])[:MAX_OR_TERMS]
            if not stance_terms:
                stance_terms = STANCE_DICT["neutral"][:MAX_OR_TERMS]

            q_string = build_query(
                anchor_terms=anchor_terms,
                bridge_terms=bridge_terms,
                stance_terms=stance_terms,
                platform="twitter",
                add_filters=add_filters,
            )
            if len(q_string) > MAX_QUERY_LENGTH:
                for n_bridge in range(len(bridge_terms) - 1, -1, -1):
                    q_string = build_query(
                        anchor_terms=anchor_terms,
                        bridge_terms=bridge_terms[:n_bridge],
                        stance_terms=stance_terms,
                        platform="twitter",
                        add_filters=add_filters,
                    )
                    if len(q_string) <= MAX_QUERY_LENGTH:
                        break
            if len(q_string) > MAX_QUERY_LENGTH:
                for n_stance in range(len(stance_terms) - 1, 0, -1):
                    q_string = build_query(
                        anchor_terms=anchor_terms,
                        bridge_terms=[],
                        stance_terms=stance_terms[:n_stance],
                        platform="twitter",
                        add_filters=add_filters,
                    )
                    if len(q_string) <= MAX_QUERY_LENGTH:
                        break
            if len(q_string) > MAX_QUERY_LENGTH:
                for n_anchor in range(len(anchor_terms) - 1, 0, -1):
                    q_string = build_query(
                        anchor_terms=anchor_terms[:n_anchor],
                        bridge_terms=[],
                        stance_terms=stance_terms[:1],
                        platform="twitter",
                        add_filters=add_filters,
                    )
                    if len(q_string) <= MAX_QUERY_LENGTH:
                        break
            if len(q_string) > MAX_QUERY_LENGTH:
                logger.warning(
                    "Twitter query topic %d stance %s still exceeds %d chars (%d)",
                    topic_id, stance, MAX_QUERY_LENGTH, len(q_string),
                )

            queries.append(SearchQuery(
                query_string=q_string,
                platform="twitter",
                intent=_STANCE_TO_INTENT.get(stance, QueryIntent.DIVERSE),
                probability=_STANCE_PROBABILITY.get(stance, 0.5),
                source_topic_id=topic_id,
                source_keywords=keywords[:MAX_OR_TERMS],
                metadata={"stance": stance},
            ))

        # --- Reddit: cycle headline then key_points, exactly n_reddit_queries ---
        reddit_sources = [headline] + (key_points or [])
        for j in range(cfg.n_reddit_queries):
            q_string = reddit_sources[j % len(reddit_sources)] if reddit_sources else headline
            queries.append(SearchQuery(
                query_string=q_string,
                platform="reddit",
                intent=QueryIntent.DIVERSE,
                probability=0.60,
                source_topic_id=topic_id,
                source_keywords=keywords[:MAX_OR_TERMS],
                metadata={"stance": "reddit_plain"},
            ))

        return queries

    # ------------------------------------------------------------------
    # Strategy: LLM anchor + bridge extraction
    # ------------------------------------------------------------------

    def _llm_anchor_bridge(
        self,
        topic_id: int,
        headline: str,
        keywords: list[str],
    ) -> tuple[list[str], list[str]]:
        """
        Ask the attached LLM to produce anchor_terms and bridge_terms JSON.

        Expected LLM response (JSON):
            {"anchor_terms": ["Entity A", "Entity B"],
             "bridge_terms": ["keyword1", "keyword2"]}

        Falls back to NER + keyword split on any parse failure.
        """
        if self.summarizer is None:
            return _extract_anchors(headline, keywords), keywords

        prompt = (
            f'Given this news headline: "{headline}"\n'
            f'And these keywords: {keywords}\n\n'
            'Return JSON with two keys:\n'
            '  "anchor_terms": 2–4 named entities or specific nouns '
            'that uniquely identify the topic.\n'
            '  "bridge_terms": 3–5 general keywords that link the '
            'topic to broader discussions.\n'
            'Return ONLY valid JSON, no explanation.'
        )
        try:
            import json
            # BaseSummarizer doesn't have a free-form generate method,
            # so we use the OpenAI/Anthropic client directly if accessible.
            raw = self._call_summarizer_raw(prompt)
            data = json.loads(raw)
            anchor_terms = data.get("anchor_terms", [])[:MAX_OR_TERMS]
            bridge_terms = data.get("bridge_terms", [])[:MAX_OR_TERMS]
            if anchor_terms:
                return anchor_terms, bridge_terms
        except Exception as exc:
            logger.warning(
                "LLM anchor extraction failed for topic %d (%s) — "
                "falling back to NER", topic_id, exc
            )

        return _extract_anchors(headline, keywords), keywords

    def _call_summarizer_raw(self, prompt: str) -> str:
        """
        Call the attached summarizer's underlying client with a raw prompt.

        Supports OpenAISummarizer and AnthropicSummarizer by duck-typing
        their internal client attributes.
        """
        s = self.summarizer

        # OpenAISummarizer
        if hasattr(s, "_client") and hasattr(s._client, "chat"):
            resp = s._client.chat.completions.create(
                model=getattr(s, "model", "gpt-4o-mini"),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=256,
                temperature=0.3,
            )
            return resp.choices[0].message.content or ""

        # AnthropicSummarizer
        if hasattr(s, "_client") and hasattr(s._client, "messages"):
            resp = s._client.messages.create(
                model=getattr(s, "model", "claude-haiku-4-5-20251001"),
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text if resp.content else ""

        raise AttributeError("Unsupported summarizer type for raw LLM call")
