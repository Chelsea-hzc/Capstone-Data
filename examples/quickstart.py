"""
Quickstart examples showing common configurations.

Run from the repo root:
    python examples/quickstart.py --json path/to/timeline.json
"""

from __future__ import annotations

import argparse
import json

from social_topic_miner import TopicMinerConfig, TopicMinerPipeline
from social_topic_miner.config import (
    ClusteringConfig,
    EmbedderConfig,
    HDBSCANConfig,
    SamplingConfig,
    SelectionConfig,
    UMAPConfig,
)
from social_topic_miner.embedders import SentenceTransformerEmbedder


def example_default(json_path: str) -> None:
    """Simplest usage — everything defaults to MiniLM + no summariser."""
    pipeline = TopicMinerPipeline()
    result = pipeline.run_from_json(json_path)
    pipeline.display(result)


def example_openai_embedder(json_path: str, openai_key: str) -> None:
    """Swap in the OpenAI text-embedding-3-small model."""
    from social_topic_miner.embedders import OpenAIEmbedder
    from social_topic_miner.config import EmbedderConfig

    pipeline = TopicMinerPipeline(
        embedder=OpenAIEmbedder(
            api_key=openai_key,
            config=EmbedderConfig(model_name="text-embedding-3-small"),
        )
    )
    result = pipeline.run_from_json(json_path)
    pipeline.display(result)


def example_with_openai_summary(json_path: str, openai_key: str) -> None:
    """MiniLM embeddings + GPT-4o-mini for topic summaries."""
    from social_topic_miner.summarizers import OpenAISummarizer

    pipeline = TopicMinerPipeline(
        summarizer=OpenAISummarizer(api_key=openai_key, model="gpt-4o-mini"),
    )
    result = pipeline.run_from_json(json_path)
    pipeline.display(result)

    # Access structured results programmatically
    for topic in result.topics:
        if topic.summary:
            print(f"[{topic.summary.category}] {topic.summary.headline}")


def example_with_anthropic_summary(json_path: str, anthropic_key: str) -> None:
    """MiniLM embeddings + Claude Haiku for topic summaries."""
    from social_topic_miner.summarizers import AnthropicSummarizer

    pipeline = TopicMinerPipeline(
        summarizer=AnthropicSummarizer(
            api_key=anthropic_key,
            model="claude-haiku-4-5-20251001",  # fast & cheap
        ),
    )
    result = pipeline.run_from_json(json_path)
    pipeline.display(result)


def example_custom_config(json_path: str) -> None:
    """Fine-tune clustering and selection parameters."""
    config = TopicMinerConfig(
        embedder=EmbedderConfig(model_name="all-mpnet-base-v2", batch_size=64),
        clustering=ClusteringConfig(
            umap=UMAPConfig(n_neighbors=10, n_components=15),
            hdbscan=HDBSCANConfig(min_cluster_size=10, min_samples=5),
            min_topic_size=10,
        ),
        selection=SelectionConfig(top_n_topics=7),
        sampling=SamplingConfig(
            recency_window_hours=48,
            engagement_floor_percentile=0.10,
            target_max=12,
        ),
    )
    pipeline = TopicMinerPipeline(config=config)
    result = pipeline.run_from_json(json_path)
    pipeline.display(result)


def example_custom_embedder(json_path: str) -> None:
    """
    Plug in any embedding model by subclassing BaseEmbedder.
    Here we show a trivial TF-IDF baseline (no neural network needed).
    """
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    from social_topic_miner.embedders.base import BaseEmbedder

    class TfidfEmbedder(BaseEmbedder):
        def __init__(self, max_features: int = 5000) -> None:
            self._vec = TfidfVectorizer(max_features=max_features, sublinear_tf=True)
            self._fitted = False

        def embed(self, texts: list[str]) -> np.ndarray:
            if not self._fitted:
                matrix = self._vec.fit_transform(texts)
                self._fitted = True
            else:
                matrix = self._vec.transform(texts)
            return normalize(matrix.toarray()).astype(np.float32)

    pipeline = TopicMinerPipeline(embedder=TfidfEmbedder())
    result = pipeline.run_from_json(json_path)
    pipeline.display(result)


def example_export_json(json_path: str, output_path: str) -> None:
    """Run the pipeline and save the structured results as JSON."""
    pipeline = TopicMinerPipeline()
    result = pipeline.run_from_json(json_path)

    output = {
        "run_at": result.run_at.isoformat(),
        "topics": [
            {
                "topic_id": t.topic_id,
                "keywords": t.keywords,
                "n_posts": t.n_posts,
                "n_perspectives": t.n_perspectives,
                "summary": {
                    "category": t.summary.category,
                    "headline": t.summary.headline,
                    "key_points": t.summary.key_points,
                } if t.summary else None,
                "representative_posts": t.representative_posts,
            }
            for t in result.topics
        ],
    }
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)
    print(f"Results written to {output_path}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="social-topic-miner quickstart")
    parser.add_argument("--json", required=True, help="Path to timeline JSON")
    parser.add_argument("--openai-key", default=None)
    parser.add_argument("--anthropic-key", default=None)
    parser.add_argument("--output", default=None, help="Save results to JSON file")
    parser.add_argument(
        "--mode",
        choices=["default", "openai-embed", "openai-summary", "anthropic-summary",
                 "custom-config", "custom-embedder"],
        default="default",
    )
    args = parser.parse_args()

    if args.mode == "default":
        example_default(args.json)
    elif args.mode == "openai-embed":
        example_openai_embedder(args.json, args.openai_key)
    elif args.mode == "openai-summary":
        example_with_openai_summary(args.json, args.openai_key)
    elif args.mode == "anthropic-summary":
        example_with_anthropic_summary(args.json, args.anthropic_key)
    elif args.mode == "custom-config":
        example_custom_config(args.json)
    elif args.mode == "custom-embedder":
        example_custom_embedder(args.json)

    if args.output:
        example_export_json(args.json, args.output)
