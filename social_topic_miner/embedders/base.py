"""
Abstract base class for all embedding backends.

To plug in a new embedding model, subclass BaseEmbedder and implement
``embed()``.  The pipeline accepts any object satisfying this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseEmbedder(ABC):
    """
    Minimal protocol every embedder must satisfy.

    ``embed`` must return a float32 array of shape (n_docs, dim).
    """

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Embed a list of strings.

        Returns
        -------
        np.ndarray
            Shape (len(texts), embedding_dim), dtype float32.
        """

    # Convenience alias so embedders can be passed directly as
    # BERTopic ``embedding_model`` — BERTopic calls .encode(), not .embed().
    def encode(self, texts: list[str], **_kwargs) -> np.ndarray:
        return self.embed(texts)
