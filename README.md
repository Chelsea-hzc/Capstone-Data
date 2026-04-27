# social-topic-miner

A Python package that clusters social media posts into topics, generates echo-chamber-breaking search queries, and filters diverse content — designed to power a three-tab "news feed awareness" product.

---

## Table of Contents

1. [What this repo does](#what-this-repo-does)
2. [Architecture](#architecture)
3. [Package structure](#package-structure)
4. [Installation](#installation)
5. [API reference](#api-reference)
   - [Section 1 — Cluster your feed](#section-1--cluster-your-feed)
   - [Section 2 — Generate echo-breaking queries](#section-2--generate-echo-breaking-queries)
   - [Section 3 — Filter diverse content](#section-3--filter-diverse-content)
   - [Combined — run all three](#combined--run-all-three)
6. [Input / output types](#input--output-types)
7. [Swapping models](#swapping-models)
8. [Configuration reference](#configuration-reference)
9. [Extending the package](#extending-the-package)
10. [Development](#development)

---

## What this repo does

The package sits between a social-media data-collection layer and a full-stack front/back-end product.  It is called as a Python library — not a web service.

```
[Data collection product]
        │  list of posts (JSON)
        ▼
[social-topic-miner]  ◄─── this repo
   Section 1: cluster & summarise the user's feed
   Section 2: generate ranked search queries to find diverse perspectives
   Section 3: score & split fetched posts into two lists
        │  structured JSON results
        ▼
[Full-stack front/back-end product]
   Tab 1: "Feed & For You"    ← Section 1 output
   Tab 2: "Balanced"          ← Section 3 balanced list → re-run Section 1
   Tab 3: "Other"             ← Section 3 other list
```

---

## Architecture

### Full pipeline flow

```
API — Following feed (N posts)
  │
  ▼
┌──────────────────────────────────────────────────────────────────┐
│  SECTION 1 — Cluster + Summarise  (run on following feed)        │
│                                                                  │
│  posts[]                                                         │
│    ├─► PostNormalizer   Twitter / Reddit → unified schema        │
│    │       └─► TextCleaner   URLs, mentions, hashtags            │
│    ├─► Embedder         text → float vectors                     │
│    │   (pluggable: SentenceTransformer | OpenAI | custom)        │
│    ├─► TopicClusterer   BERTopic (UMAP → HDBSCAN → c-TF-IDF)    │
│    ├─► EngagementScorer rank topics by engagement                │
│    ├─► SubPartitioner   KMeans sub-perspectives per topic        │
│    ├─► PostSampler      pick S representative posts per topic    │
│    └─► Summarizer       LLM → headline · short_summary ·        │
│        (OpenAI|Anthropic|Llama|None)   long_summary · key_points │
│                                                                  │
│  → Section1Response                                              │
│      topics[]:  topic_id, headline, short_summary, long_summary  │
│                 category, keywords, key_points, n_posts,         │
│                 n_perspectives, representative_posts[]           │
│      total_posts_processed: int                                  │
│      digest: str   (5-10 sentence feed overview — LLM or concat) │
└──────────────────────────────────────────────────────────────────┘
        │                               │
        │ Tab 1: Feed & For You         │ full TopicOut per topic
        ▼                               ▼
  [Frontend]            ┌──────────────────────────────────────────┐
                        │  SECTION 2 — Query Generation            │
                        │                                          │
                        │  TopicOut[] (full Section 1 output)      │
                        │    └─► QueryBuilder                      │
                        │          keyword expansion from          │
                        │          keywords + key_points           │
                        │          NER anchor extraction (spaCy)   │
                        │          one query per stance bucket:    │
                        │            critical · emotional ·        │
                        │            supportive · neutral ·        │
                        │            industry                      │
                        │          MAX_OR_TERMS = 5 enforced       │
                        │                                          │
                        │  → Section2Response                      │
                        │      queries[] sorted by probability↓:  │
                        │        query_string, platform, intent,  │
                        │        probability, source_topic_id,     │
                        │        source_keywords, metadata         │
                        │      source_topic_ids: list[int]         │
                        └──────────────────────────────────────────┘
                                        │
                              Backend executes queries
                              against Twitter / Reddit API
                                        │
                                        ▼
                          Search results (K queries × Q posts)
                                        │
                                        ▼
┌──────────────────────────────────────────────────────────────────┐
│  SECTION 3 — Diversity Cut-off                                   │
│                                                                  │
│  new_posts[] (from search) + optional following posts            │
│    └─► DiversityFilter                                           │
│          relevance:   keyword overlap | embedding cosine sim     │
│          divergence:  1 − cosine_sim to bubble centroid          │
│          recency:     timestamp decay (TODO)                     │
│          split:                                                  │
│            score ≥ balanced_threshold (0.50) → balanced[]       │
│            score ≥ min_diversity_score (0.30) → other[]         │
│            score <  min_diversity_score       → dropped          │
│                                                                  │
│  → Section3Response                                              │
│      balanced[]:        high-diversity posts                     │
│      balanced_scores[]: diversity score per balanced post        │
│      other[]:           lower-diversity posts still above floor  │
│      other_scores[]:    diversity score per other post           │
│      dropped:           int  (below minimum floor)               │
└──────────────────────────────────────────────────────────────────┘
        │                               │
        │ Tab 3: Other                  │ balanced[] → re-run Section 1
        ▼                               ▼
  [Frontend]              Section 1 on balanced posts
                                        │
                                        ▼ Tab 2: Balanced
                                  [Frontend]
```

### Key design decisions

| Decision | Rationale |
|---|---|
| Pluggable embedder | SentenceTransformer works offline/free; OpenAI gives higher quality; custom models drop in with one class |
| Pluggable summarizer | OpenAI, Anthropic Claude, local Llama all work; `None` skips summarisation entirely |
| `short_summary` + `long_summary` | Frontend needs both a card preview (1-2 sentences) and a detail view (paragraph) without calling the API twice |
| `digest` on Section1Response | Banner text summarising the whole feed; LLM-generated when a summarizer is attached, headline-concat fallback otherwise |
| Probability on queries | Backend can send only the top-K queries; critical/emotional stances score highest based on experiment results |
| Full TopicOut as Section 2 input | Decouples keyword expansion from the API contract — swap to use representative posts or summaries without changing the caller |
| Two-list Section 3 output | `balanced` goes back into Section 1 to cluster the diverse perspective; `other` is shown as a flat ranked list |
| TypedDicts for all I/O | Backend can import the types for type-checking without pulling in heavy ML deps |
| Sections are independent | Backend can call section1 → wait → section2 → fetch → section3 across multiple HTTP requests |

---

## Package structure

```
social_topic_miner/
│
├── api.py                   ← START HERE — the three public methods
├── types.py                 ← All input/output TypedDicts
├── config.py                ← All tunable parameters in one place
├── pipeline.py              ← Section 1 orchestrator (internal)
│
├── preprocessing/
│   ├── normalizer.py        ← Twitter + Reddit → unified schema
│   └── cleaner.py           ← URL / mention / hashtag cleaning
│
├── embedders/
│   ├── base.py              ← BaseEmbedder ABC (subclass to add a model)
│   ├── sentence_transformer.py
│   └── openai.py
│
├── clustering/
│   ├── topic_model.py       ← BERTopic wrapper
│   └── sub_partitioner.py   ← KMeans sub-perspective split
│
├── selection/
│   ├── scorer.py            ← Platform-normalised engagement scoring
│   └── sampler.py           ← Representative post selection
│
├── summarizers/
│   ├── base.py              ← BaseSummarizer ABC + summarize_digest()
│   ├── openai.py
│   ├── anthropic.py
│   └── llama.py
│
├── echo_breaker/
│   └── query_builder.py     ← Section 2: stance-based query builder
│
└── diversity/
    └── filter.py            ← Section 3: two-list diversity cut-off

examples/
├── quickstart.py            ← Minimal usage examples
└── colab_full_flow_test.py  ← Full flow test matching the product diagram
```

---

## Installation

**From GitHub (recommended):**
```bash
pip install "git+https://github.com/Chelsea-hzc/Capstone-Data.git@test"
```

**With optional extras:**
```bash
pip install "git+https://...@test#egg=social-topic-miner[openai]"      # OpenAI embedder + summarizer
pip install "git+https://...@test#egg=social-topic-miner[anthropic]"   # Anthropic Claude summarizer
pip install "git+https://...@test#egg=social-topic-miner[llama]"       # Local Llama (needs GPU)
pip install "git+https://...@test#egg=social-topic-miner[all]"         # Everything
```

**In Google Colab:**
```python
!pip install "git+https://github.com/Chelsea-hzc/Capstone-Data.git@test" -q
!pip install bertopic umap-learn hdbscan sentence-transformers -q
```

---

## API reference

The full-stack backend interacts with one class: `TopicMinerAPI`.

```python
from social_topic_miner import TopicMinerAPI

api = TopicMinerAPI()   # SentenceTransformer embedder, no summarizer by default
```

To add a summarizer (enables `headline`, `short_summary`, `long_summary`, `digest`, `key_points`):
```python
from social_topic_miner.summarizers import OpenAISummarizer, AnthropicSummarizer

api = TopicMinerAPI(summarizer=OpenAISummarizer(api_key="sk-..."))
# or
api = TopicMinerAPI(summarizer=AnthropicSummarizer(api_key="..."))
```

---

### Section 1 — Cluster your feed

```python
response = api.section1(posts)
```

**Input** `posts: list[PostIn]`

Each post must have at minimum:

| Field | Type | Required | Notes |
|---|---|---|---|
| `text` | `str` | ✅ | Cleaned or raw post text |
| `platform` | `str` | ✅ | `"twitter"` or `"reddit"` |
| `created_at` | `str` | recommended | ISO-8601, e.g. `"2026-04-17T18:46:14.000Z"` |
| `like_count` | `int` | recommended | Used for engagement ranking |
| `reply_count` | `int` | recommended | Twitter only |
| `retweet_count` | `int` | recommended | Twitter only |
| `impression_count` | `int` | optional | Twitter only |
| `ups` | `int` | recommended | Reddit only |
| `num_comments` | `int` | recommended | Reddit only |
| `tweet_id` / `reddit_id` | `str` | optional | Used as `post_id` |

Full schema: [`social_topic_miner/types.py → PostIn`](social_topic_miner/types.py)

**Output** `Section1Response`

```python
{
  "topics": [
    {
      "topic_id":             int,          # cluster ID (stable within a run)
      "headline":             str,          # LLM: news-style headline
      "category":             str,          # LLM: e.g. "Tech", "Sports", "Politics"
      "short_summary":        str,          # LLM: 1-2 sentences for card / preview view
      "long_summary":         str,          # LLM: paragraph for detail / expanded view
      "keywords":             list[str],    # top c-TF-IDF terms from BERTopic
      "key_points":           list[str],    # LLM: bullet-point highlights
      "n_posts":              int,          # posts assigned to this cluster
      "n_perspectives":       int,          # sub-clusters (viewpoints) within topic
      "representative_posts": [
        {
          "post_id":          str,
          "text":             str,
          "platform":         str,
          "sub_perspective":  int,
          "engagement_norm":  float,        # 0–1 within-platform percentile
          "created_at":       str
        },
        ...
      ]
    },
    ...
  ],
  "total_posts_processed": int,
  "digest":                str            # 5-10 sentence overview across all topics
}
```

> `headline`, `category`, `short_summary`, `long_summary`, `key_points` are empty / `"Unknown"` when no summarizer is configured.  
> `digest` falls back to a `" | "`-joined headline string when no summarizer is configured.

---

### Section 2 — Generate echo-breaking queries

```python
# Normal flow — pass the full Section 1 topic output
response = api.section2({"topics": s1["topics"]})

# Manual override for testing
response = api.section2({"headlines": ["AI safety cuts"], "keywords": ["AI", "safety"]})
```

**Input** `Section2Request` — pass one of:

| Option | Fields | When to use |
|---|---|---|
| From Section 1 | `{"topics": list[TopicOut]}` | Normal flow — all topic fields are used for keyword expansion |
| Manual | `{"headlines": list[str], "keywords": list[str]}` | Testing / overrides |

The builder uses the full `TopicOut` dict internally: `keywords` and `key_points` feed keyword expansion; `headline` drives NER anchor extraction (spaCy); `representative_posts` is reserved for future embedding-based expansion.

**Output** `Section2Response`

```python
{
  "queries": [          # sorted by probability descending — send top-K first
    {
      "query_string":     str,          # ready-to-send search string
      "platform":         str,          # "twitter" | "reddit" | "any"
      "intent":           str,          # "opposing" | "diverse" | "related" | "factual"
      "probability":      float,        # 0–1 estimated probability of diverse results
      "source_topic_id":  int,
      "source_keywords":  list[str],
      "metadata":         dict          # includes {"stance": "critical"} etc.
    },
    ...
  ],
  "source_topic_ids": list[int]
}
```

**Query shape (Twitter v2 syntax):**
```
(anchor1 OR anchor2) (bridge1 OR bridge2 OR ...) (stance1 OR stance2 OR ...) -is:retweet lang:en
```

**Stance buckets and default probabilities:**

| Stance | Intent | Probability |
|---|---|---|
| `critical` | opposing | 0.88 |
| `emotional` | diverse | 0.78 |
| `supportive` | diverse | 0.65 |
| `neutral` | related | 0.58 |
| `industry` | related | 0.52 |

> OR blocks are hard-capped at 5 terms (`MAX_OR_TERMS = 5`) — longer blocks return 0 results on the Twitter v2 recent-search endpoint.

---

### Section 3 — Filter diverse content

```python
response = api.section3({
    "new_posts":       fetched_posts,          # list[PostIn] from platform API
    "bubble_keywords": all_bubble_keywords,    # flat list of all Section 1 keywords
})
```

Collect `all_bubble_keywords` from Section 1:
```python
all_bubble_keywords = [kw for t in s1["topics"] for kw in t["keywords"]]
```

**Input** `Section3Request`

| Field | Type | Required | Notes |
|---|---|---|---|
| `new_posts` | `list[PostIn]` | ✅ | Posts fetched via Section 2 queries |
| `bubble_keywords` | `list[str]` | recommended | All keywords from Section 1 — improves relevance scoring |

**Output** `Section3Response`

```python
{
  "balanced":        list[PostIn],  # score ≥ balanced_threshold (default 0.50)
                                    # → feed these back into Section 1 for the Balanced tab
  "balanced_scores": list[float],   # diversity score per balanced post (same order)
  "other":           list[PostIn],  # min_score ≤ score < balanced_threshold (default 0.30–0.50)
                                    # → show as flat ranked list in the Other tab
  "other_scores":    list[float],   # diversity score per other post (same order)
  "dropped":         int            # posts below min_diversity_score, discarded
}
```

Both lists are sorted by diversity score descending.  The combined size is capped at `max_posts_out` (default 20).

**Recommended backend flow:**
```python
s3 = api.section3({"new_posts": search_results, "bubble_keywords": all_bubble_keywords})

# Tab 2 — Balanced: re-cluster the diverse posts
s1_balanced = api.section1(s3["balanced"])

# Tab 3 — Other: show as a flat ranked list
other_posts = s3["other"]
```

**Diversity score components:**

| Component | Weight | Current implementation |
|---|---|---|
| Relevance | 0.40 | Keyword overlap with bubble topics |
| Divergence | 0.40 | 1 − cosine similarity to bubble centroid (requires embedder) |
| Recency | 0.20 | Uniform 0.5 placeholder (timestamp decay TODO) |

> Thresholds (`balanced_threshold`, `min_diversity_score`) and weights are configurable via `DiversityFilterConfig`.

---

### Combined — run all three

When the backend wants to do everything in a single call:

```python
response = api.run_full({
    "posts":     user_feed_posts,    # required
    "new_posts": diverse_posts,      # optional — skip Section 3 if omitted
})
```

**Output** `FullPipelineResponse`

```python
{
  "section1": Section1Response,
  "section2": Section2Response,
  "section3": Section3Response | None   # None when new_posts not provided
}
```

---

## Input / output types

All types live in [`social_topic_miner/types.py`](social_topic_miner/types.py) and are re-exported from the top-level package:

```python
from social_topic_miner import (
    PostIn,
    TopicOut,
    Section1Response,
    Section2Request,  Section2Response,
    Section3Request,  Section3Response,
    FullPipelineRequest, FullPipelineResponse,
)
```

Import these in your backend for type hints — they have no heavy ML dependencies.

---

## Swapping models

### Embedding model

```python
from social_topic_miner import TopicMinerAPI
from social_topic_miner.embedders import SentenceTransformerEmbedder, OpenAIEmbedder
from social_topic_miner.config import EmbedderConfig

# Different SentenceTransformer model
api = TopicMinerAPI(
    embedder=SentenceTransformerEmbedder(
        config=EmbedderConfig(model_name="all-mpnet-base-v2")
    )
)

# OpenAI embeddings
api = TopicMinerAPI(
    embedder=OpenAIEmbedder(api_key="sk-...",
                            config=EmbedderConfig(model_name="text-embedding-3-small"))
)
```

**Custom embedder** — subclass `BaseEmbedder` and implement one method:

```python
from social_topic_miner.embedders.base import BaseEmbedder
import numpy as np

class MyEmbedder(BaseEmbedder):
    def embed(self, texts: list[str]) -> np.ndarray:
        # return float32 array of shape (len(texts), dim)
        ...

api = TopicMinerAPI(embedder=MyEmbedder())
```

### Summarizer

```python
from social_topic_miner.summarizers import OpenAISummarizer, AnthropicSummarizer, LlamaSummarizer

api = TopicMinerAPI(summarizer=OpenAISummarizer(api_key="sk-...", model="gpt-4o-mini"))
api = TopicMinerAPI(summarizer=AnthropicSummarizer(api_key="...", model="claude-haiku-4-5-20251001"))
api = TopicMinerAPI(summarizer=LlamaSummarizer(model_id="meta-llama/Meta-Llama-3.1-8B-Instruct"))
```

**Custom summarizer** — subclass `BaseSummarizer` and implement `summarize()`.  Override `summarize_digest()` for a custom feed digest:

```python
from social_topic_miner.summarizers.base import BaseSummarizer, TopicSummary

class MySummarizer(BaseSummarizer):
    def summarize(self, topic_id: int, posts: list[str], keywords: list[str]) -> TopicSummary:
        # call your LLM; return TopicSummary with headline, short_summary,
        # long_summary, category, key_points
        ...

    def summarize_digest(self, topic_summaries: list[TopicSummary]) -> str:
        # optional — return a multi-sentence digest of all topics
        ...
```

---

## Configuration reference

All parameters have sensible defaults. Override only what you need:

```python
from social_topic_miner import TopicMinerAPI, TopicMinerConfig
from social_topic_miner.config import (
    PreprocessingConfig,  # min_word_count, drop_nsfw, dedup
    ClusteringConfig,     # UMAP, HDBSCAN, min_topic_size
    SelectionConfig,      # top_n_topics, engagement weights
    SamplingConfig,       # recency_window_hours, target_min/max
)
from social_topic_miner.diversity.filter import DiversityFilterConfig

config = TopicMinerConfig(
    preprocessing=PreprocessingConfig(min_word_count=5),
    selection=SelectionConfig(top_n_topics=7),
    sampling=SamplingConfig(recency_window_hours=48, target_max=12),
)
diversity_config = DiversityFilterConfig(
    balanced_threshold=0.55,     # raise to make Balanced tab stricter
    min_diversity_score=0.25,    # lower to keep more posts in Other tab
    max_posts_out=30,
)
api = TopicMinerAPI(config=config, diversity_config=diversity_config)
```

Full list of parameters: [`social_topic_miner/config.py`](social_topic_miner/config.py)

---

## Extending the package

| What to change | Where to look | What to implement |
|---|---|---|
| New embedding model | `embedders/` | Subclass `BaseEmbedder`, implement `embed()` |
| New LLM summarizer | `summarizers/` | Subclass `BaseSummarizer`, implement `summarize()` and optionally `summarize_digest()` |
| New social platform | `preprocessing/normalizer.py` | Override `_extra_platform_rows()` |
| Better keyword expansion | `echo_breaker/query_builder.py` | Replace `_expand_keywords()` — receives full TopicOut including representative posts |
| Better query generation | `echo_breaker/query_builder.py` | Fill in `_strategy_llm()` or extend `_build_stance_queries()` |
| Better diversity scoring | `diversity/filter.py` | Replace `_score_relevance()` and `_score_divergence()` |
| Change balanced / other split | `DiversityFilterConfig` | Adjust `balanced_threshold` and `min_diversity_score` |

---

## Development

```bash
git clone https://github.com/Chelsea-hzc/Capstone-Data.git
cd Capstone-Data
pip install -e ".[dev]"

# run tests
pytest

# lint
ruff check social_topic_miner/
```

### Branch strategy

| Branch | Purpose |
|---|---|
| `main` | Stable — merged from test after review |
| `test` | Active development |
