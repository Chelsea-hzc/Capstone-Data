"""
social-topic-miner
==================
End-to-end topic clustering and summarisation for social media posts.

Quick-start
-----------
>>> from social_topic_miner import TopicMinerPipeline, TopicMinerConfig
>>> from social_topic_miner.embedders import SentenceTransformerEmbedder
>>> from social_topic_miner.summarizers import OpenAISummarizer
>>>
>>> pipeline = TopicMinerPipeline(
...     embedder=SentenceTransformerEmbedder(),
...     summarizer=OpenAISummarizer(api_key="sk-..."),
... )
>>> result = pipeline.run_from_json("timeline.json")
>>> pipeline.display(result)
"""

from .config import (
    ClusteringConfig,
    EmbedderConfig,
    HDBSCANConfig,
    PartitionConfig,
    PreprocessingConfig,
    SamplingConfig,
    SelectionConfig,
    SummarisationConfig,
    TopicMinerConfig,
    UMAPConfig,
)
from .pipeline import PipelineResult, TopicMinerPipeline, TopicResult

__all__ = [
    # Pipeline
    "TopicMinerPipeline",
    "PipelineResult",
    "TopicResult",
    # Config
    "TopicMinerConfig",
    "PreprocessingConfig",
    "EmbedderConfig",
    "ClusteringConfig",
    "UMAPConfig",
    "HDBSCANConfig",
    "SelectionConfig",
    "PartitionConfig",
    "SamplingConfig",
    "SummarisationConfig",
]

__version__ = "0.1.0"
