"""
OpenAI embedding backend.

Supports text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002.
Requires: pip install openai
"""

from __future__ import annotations

import numpy as np

from ..config import EmbedderConfig
from .base import BaseEmbedder

# OpenAI supports up to 2 048 inputs per batch call.
_OPENAI_BATCH_LIMIT = 2048


class OpenAIEmbedder(BaseEmbedder):
    """
    Parameters
    ----------
    api_key:
        OpenAI API key.  Falls back to the ``OPENAI_API_KEY`` environment variable.
    config:
        EmbedderConfig — ``model_name`` is used as the OpenAI model identifier
        (default ``"text-embedding-3-small"``).
    """

    def __init__(
        self,
        api_key: str | None = None,
        config: EmbedderConfig | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai>=1.0 is required for OpenAIEmbedder. "
                "Install it with: pip install openai"
            ) from exc

        cfg = config or EmbedderConfig(model_name="text-embedding-3-small")
        self.config = cfg
        self._client = OpenAI(api_key=api_key)

    def embed(self, texts: list[str]) -> np.ndarray:
        all_vectors: list[list[float]] = []

        # Batch to stay within API limits
        for start in range(0, len(texts), _OPENAI_BATCH_LIMIT):
            batch = texts[start : start + _OPENAI_BATCH_LIMIT]
            response = self._client.embeddings.create(
                model=self.config.model_name,
                input=batch,
            )
            all_vectors.extend(item.embedding for item in response.data)

        return np.array(all_vectors, dtype=np.float32)
