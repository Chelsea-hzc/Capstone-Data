"""
SentenceTransformer embedding backend.

Supports any model on HuggingFace that works with the sentence-transformers
library (e.g. all-MiniLM-L6-v2, all-mpnet-base-v2, paraphrase-multilingual-MiniLM-L12-v2).
"""

from __future__ import annotations

import numpy as np

from ..config import EmbedderConfig
from .base import BaseEmbedder


class SentenceTransformerEmbedder(BaseEmbedder):
    """
    Parameters
    ----------
    config:
        EmbedderConfig with ``model_name``, ``batch_size``, ``show_progress_bar``.
    model_kwargs:
        Extra keyword arguments forwarded to ``SentenceTransformer.__init__``
        (e.g. ``device="cuda"``, ``cache_folder="/tmp"``).
    """

    def __init__(
        self,
        config: EmbedderConfig | None = None,
        **model_kwargs,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for SentenceTransformerEmbedder. "
                "Install it with: pip install sentence-transformers"
            ) from exc

        self.config = config or EmbedderConfig()
        self._model = SentenceTransformer(self.config.model_name, **model_kwargs)

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts,
            batch_size=self.config.batch_size,
            show_progress_bar=self.config.show_progress_bar,
            convert_to_numpy=True,
        )
        return vectors.astype(np.float32)

    @property
    def dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()
