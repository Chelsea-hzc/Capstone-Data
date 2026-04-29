"""
OpenAI chat-completion summarisation backend.

Requires: pip install openai
Supports gpt-4o, gpt-4-turbo, gpt-3.5-turbo, etc.
"""

from __future__ import annotations

from ..config import SummarisationConfig
from .base import BaseSummarizer, TopicSummary, _DIGEST_SYSTEM_PROMPT


class OpenAISummarizer(BaseSummarizer):
    """
    Parameters
    ----------
    api_key:
        OpenAI API key (falls back to ``OPENAI_API_KEY`` env var).
    model:
        Chat model to use (e.g. ``"gpt-4o-mini"``).
    config:
        SummarisationConfig with temperature and max_new_tokens.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        config: SummarisationConfig | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai>=1.0 is required. Install with: pip install openai"
            ) from exc

        super().__init__(config)
        self._client = OpenAI(api_key=api_key)
        self.model = model

    def summarize(self, topic_id: int, posts: list[str], keywords: list[str]) -> TopicSummary:
        cfg = self.config
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": self._build_user_prompt(posts, keywords)},
            ],
            max_tokens=cfg.max_topic_summarize_tokens,
            temperature=cfg.temperature,
        )
        raw = response.choices[0].message.content or ""
        return self._parse_response(topic_id, raw)

    def summarize_digest(self, topic_summaries: list[TopicSummary]) -> str:
        cfg = self.config
        if not topic_summaries:
            return ""
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _DIGEST_SYSTEM_PROMPT},
                {"role": "user", "content": self._build_digest_user_prompt(topic_summaries)},
            ],
            max_tokens=cfg.max_digest_tokens,
            temperature=0.7,
        )
        return response.choices[0].message.content or super().summarize_digest(topic_summaries)
