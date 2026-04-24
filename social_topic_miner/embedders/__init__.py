from .base import BaseEmbedder
from .sentence_transformer import SentenceTransformerEmbedder

__all__ = ["BaseEmbedder", "SentenceTransformerEmbedder"]

# Optional imports — only expose when dependencies are installed
try:
    from .openai import OpenAIEmbedder  # noqa: F401
    __all__.append("OpenAIEmbedder")
except ImportError:
    pass
