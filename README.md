# social-topic-miner

A Python package that clusters social media posts into topics, generates echo-chamber-breaking search queries, and filters diverse content — designed to power a three-page "news feed awareness" product.

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
   Section 2: generate queries to find diverse perspectives
   Section 3: score & filter the fetched diverse posts
        │  structured JSON results
        ▼
[Full-stack front/back-end product]
   Page 1: "Your bubble"           ← Section 1 output
   Page 2: "Other perspectives"    ← Section 2 queries executed by backend
   Page 3: "Diverse content"       ← Section 3 output
```

---

## Architecture

### Three-section pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│  SECTION 1 — "Your Bubble"                                      │
│                                                                 │
│  posts[]                                                        │
│    │                                                            │
│    ├─► PostNormalizer   normalise Twitter / Reddit schemas      │
│    │       └─► TextCleaner   clean text (URLs, mentions, etc.) │
│    │                                                            │
│    ├─► Embedder         encode text → float vectors            │
│    │   (pluggable: SentenceTransformer | OpenAI | custom)      │
│    │                                                            │
│    ├─► TopicClusterer   BERTopic (UMAP → HDBSCAN → c-TF-IDF)  │
│    │                                                            │
│    ├─► EngagementScorer rank topics by engagement              │
│    │                                                            │
│    ├─► SubPartitioner   KMeans sub-perspectives per topic      │
│    │                                                            │
│    ├─► PostSampler      pick representative posts              │
│    │                                                            │
│    └─► Summarizer       LLM → headline, category, key_points   │
│        (pluggable: OpenAI | Anthropic | Llama | None)          │
│                                                                 │
│  → Section1Response  { topics[], total_posts_processed }       │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  SECTION 2 — "Find Other Perspectives"          [PLACEHOLDER]  │
│                                                                 │
│  topics[] (from Section 1)                                      │
│    │                                                            │
│    └─► QueryBuilder   headline + keywords → search queries     │
│        strategy: keyword inversion (now) | LLM (TODO)          │
│                                                                 │
│  → Section2Response  { queries[], source_topic_ids[] }         │
│                                                                 │
│  Backend executes queries against Twitter / Reddit API          │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  SECTION 3 — "Diverse Content"                  [PLACEHOLDER]  │
│                                                                 │
│  new_posts[] (fetched by backend using Section 2 queries)       │
│    │                                                            │
│    └─► DiversityFilter                                         │
│          relevance score:  keyword overlap (now) | embed (TODO) │
│          divergence score: 1 − cosine_sim to bubble (TODO)     │
│          recency score:    timestamp decay (TODO)              │
│          cutoff:           drop below min_diversity_score       │
│                                                                 │
│  → Section3Response  { posts[], scores[], dropped }            │
└─────────────────────────────────────────────────────────────────┘
```

### Key design decisions

| Decision | Rationale |
|---|---|
| Pluggable embedder | SentenceTransformer works offline/free; OpenAI gives higher quality; custom models drop in with one class |
| Pluggable summarizer | OpenAI, Anthropic Claude, local Llama all work; `None` skips summarisation entirely |
| TypedDicts for all I/O | Backend can import the types for type-checking without pulling in heavy ML deps |
| Sections are independent | Backend can call section1 → wait → section2 → fetch → section3 across multiple HTTP requests |
| Placeholder sections | Sections 2 & 3 have stable schemas now; logic will improve without breaking the API |

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
│   ├── base.py              ← BaseSummarizer ABC (subclass to add a model)
│   ├── openai.py
│   ├── anthropic.py
│   └── llama.py
│
├── echo_breaker/
│   └── query_builder.py     ← Section 2 placeholder
│
└── diversity/
    └── filter.py            ← Section 3 placeholder

examples/
└── quickstart.py            ← Runnable examples for every configuration
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

api = TopicMinerAPI()   # uses SentenceTransformer + no summarizer by default
```

To add a summarizer:
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
      "headline":             str,          # LLM-generated news headline
      "category":             str,          # e.g. "Tech", "Sports", "Politics"
      "keywords":             list[str],    # top c-TF-IDF terms
      "key_points":           list[str],    # LLM-generated bullet points
      "n_posts":              int,          # posts in this cluster
      "n_perspectives":       int,          # sub-clusters within the topic
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
  "total_posts_processed": int
}
```

> `headline`, `category`, `key_points` are empty/`"Unknown"` when no summarizer is configured.

---

### Section 2 — Generate echo-breaking queries

```python
response = api.section2({"topics": s1["topics"]})

# or with manual overrides:
response = api.section2({"headlines": ["AI safety cuts"], "keywords": ["AI", "safety"]})
```

**Input** `Section2Request` — pass one of:

| Option | Fields | When to use |
|---|---|---|
| From Section 1 | `{"topics": list[TopicOut]}` | Normal flow |
| Manual | `{"headlines": list[str], "keywords": list[str]}` | Testing / overrides |

**Output** `Section2Response`

```python
{
  "queries": [
    {
      "query_string":     str,          # search string for the platform API
      "platform":         str,          # "twitter" | "reddit" | "any"
      "intent":           str,          # "opposing" | "diverse" | "related" | "factual"
      "source_topic_id":  int,
      "source_keywords":  list[str],
      "metadata":         dict          # reserved for future use
    },
    ...
  ],
  "source_topic_ids": list[int]
}
```

> ⚠️ **Placeholder** — `query_string` values are generated by a keyword heuristic.
> The schema is stable; the logic will be improved.

---

### Section 3 — Filter diverse content

```python
response = api.section3({
    "new_posts":       fetched_posts,            # list[PostIn] from platform API
    "bubble_keywords": s1["topics"][0]["keywords"],  # improves scoring
})
```

**Input** `Section3Request`

| Field | Type | Required | Notes |
|---|---|---|---|
| `new_posts` | `list[PostIn]` | ✅ | Posts fetched via Section 2 queries |
| `bubble_keywords` | `list[str]` | recommended | All keywords from Section 1 topics |

**Output** `Section3Response`

```python
{
  "posts":   list[PostIn],   # filtered posts, sorted by diversity score descending
  "scores":  list[float],    # 0–1 diversity score per post (same order)
  "dropped": int             # posts removed by the cutoff threshold
}
```

> ⚠️ **Placeholder** — scoring uses keyword overlap + uniform divergence.
> The schema is stable; embedding-based and stance-detection scoring coming later.

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

**Custom summarizer** — subclass `BaseSummarizer`:

```python
from social_topic_miner.summarizers.base import BaseSummarizer, TopicSummary

class MySummarizer(BaseSummarizer):
    def summarize(self, topic_id: int, posts: list[str], keywords: list[str]) -> TopicSummary:
        # call your LLM, return a TopicSummary
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

config = TopicMinerConfig(
    preprocessing=PreprocessingConfig(min_word_count=5),
    selection=SelectionConfig(top_n_topics=7),
    sampling=SamplingConfig(recency_window_hours=48, target_max=12),
)
api = TopicMinerAPI(config=config)
```

Full list of parameters: [`social_topic_miner/config.py`](social_topic_miner/config.py)

---

## Extending the package

| What to change | Where to look | What to implement |
|---|---|---|
| New embedding model | `embedders/` | Subclass `BaseEmbedder`, implement `embed()` |
| New LLM summarizer | `summarizers/` | Subclass `BaseSummarizer`, implement `summarize()` |
| New social platform | `preprocessing/normalizer.py` | Override `_extra_platform_rows()` |
| Better query generation | `echo_breaker/query_builder.py` | Fill in `_strategy_llm()` or add a new `_strategy_*` method |
| Better diversity scoring | `diversity/filter.py` | Replace `_score_relevance()` and `_score_divergence()` |

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
