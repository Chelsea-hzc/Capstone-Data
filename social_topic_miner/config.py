"""
Centralised configuration dataclasses.

Every tuneable knob lives here so callers can import one object and override
only what they care about, while production defaults stay sane.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

@dataclass
class PreprocessingConfig:
    min_word_count: int = 8
    """Drop posts with fewer than this many words after cleaning."""

    drop_nsfw: bool = True
    """Remove posts flagged as possibly-sensitive / over-18."""

    dedup: bool = True
    """Case-insensitive exact deduplication on cleaned text."""


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

@dataclass
class EmbedderConfig:
    model_name: str = "all-MiniLM-L6-v2"
    """SentenceTransformer model name *or* OpenAI model name."""

    batch_size: int = 32
    """Batch size passed to SentenceTransformer.encode (ignored by OpenAI)."""

    show_progress_bar: bool = True


# ---------------------------------------------------------------------------
# Topic clustering (BERTopic / UMAP / HDBSCAN)
# ---------------------------------------------------------------------------

@dataclass
class UMAPConfig:
    n_neighbors: int = 5
    n_components: int = 5
    min_dist: float = 0.0
    metric: str = "cosine"
    random_state: int = 42


@dataclass
class HDBSCANConfig:
    min_cluster_size: int = 8
    min_samples: int = 3
    metric: str = "euclidean"
    cluster_selection_method: str = "eom"


@dataclass
class ClusteringConfig:
    umap: UMAPConfig = field(default_factory=UMAPConfig)
    hdbscan: HDBSCANConfig = field(default_factory=HDBSCANConfig)
    min_topic_size: int = 8
    ngram_range: tuple[int, int] = (1, 2)
    top_n_words: int = 10
    """Number of representative keywords per topic."""


# ---------------------------------------------------------------------------
# Topic selection
# ---------------------------------------------------------------------------

@dataclass
class SelectionConfig:
    top_n_topics: int = 5
    """How many top-ranked topics to carry forward."""

    weight_size: float = 0.50
    """Composite score weight: number of posts."""

    weight_total_engagement: float = 0
    """Composite score weight: total normalised engagement."""

    weight_avg_engagement: float = 0.55
    """Composite score weight: per-post normalised engagement."""

    twitter_reply_weight: float = 1.0
    twitter_share_weight: float = 1.0
    twitter_like_weight: float = 1.0
    reddit_comment_weight: float = 1.0
    reddit_like_weight: float = 1.0


# ---------------------------------------------------------------------------
# Sub-perspective partitioning (KMeans within each topic)
# ---------------------------------------------------------------------------

@dataclass
class PartitionConfig:
    min_k: int = 2
    max_k: int = 6
    min_posts_per_perspective: int = 3
    """A candidate k is skipped if any cluster would have fewer posts."""


# ---------------------------------------------------------------------------
# Post sampling
# ---------------------------------------------------------------------------

@dataclass
class SamplingConfig:
    recency_window_hours: int = 24
    """Only consider posts created within this window."""

    engagement_floor_percentile: float = 0.20
    """Drop posts below this within-platform engagement percentile."""

    posts_per_perspective: int = 2
    """How many posts to sample from each sub-perspective."""

    target_min: int = 5
    """Minimum total posts selected per topic."""

    target_max: int = 10
    """Maximum total posts selected per topic."""

    representativeness_weight: float = 0.70
    """Post-scoring weight: cosine similarity to perspective centroid."""

    engagement_weight: float = 0.30
    """Post-scoring weight: normalised engagement (tie-breaker)."""

    temperature: float = 0.10
    """Softmax temperature for sampling probabilities (lower = greedier)."""


# ---------------------------------------------------------------------------
# Summarisation
# ---------------------------------------------------------------------------

@dataclass
class SummarisationConfig:
    max_topic_summarize_tokens: int = 1024
    max_digest_tokens: int = 32
    temperature: float = 0.1
    """Generation temperature (used when do_sample=True)."""

    do_sample: bool = False

    output_schema: dict = field(default_factory=lambda: {
        "category": "String (e.g., Entertainment, Sports, Politics, Tech, Lifestyle)",
        "headline": "String (A concise, news-style headline summarising the event)",
        "key_points": ["String (Key point 1)", "String (Key point 2)"],
    })


# ---------------------------------------------------------------------------
# Master config
# ---------------------------------------------------------------------------

@dataclass
class TopicMinerConfig:
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    embedder: EmbedderConfig = field(default_factory=EmbedderConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    partition: PartitionConfig = field(default_factory=PartitionConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    summarisation: SummarisationConfig = field(default_factory=SummarisationConfig)
