"""
Anthropic (Claude) summarisation backend.

Requires: pip install anthropic
Supports claude-sonnet-4-6, claude-haiku-4-5, etc.
"""

from __future__ import annotations

import logging

from ..config import SummarisationConfig
from .base import BaseSummarizer, TopicSummary, _DIGEST_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

class AnthropicSummarizer(BaseSummarizer):
    """
    Parameters
    ----------
    api_key:
        Anthropic API key (falls back to ``ANTHROPIC_API_KEY`` env var).
    model:
        Claude model ID (e.g. ``"claude-haiku-4-5-20251001"`` for fast/cheap,
        ``"claude-sonnet-4-6"`` for higher quality).
    config:
        SummarisationConfig with temperature and max_new_tokens.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-haiku-4-5-20251001",
        config: SummarisationConfig | None = None,
    ) -> None:
        try:
            import anthropic as _anthropic
        except ImportError as exc:
            raise ImportError(
                "anthropic>=0.25 is required. Install with: pip install anthropic"
            ) from exc

        super().__init__(config)
        import anthropic
        self._client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        logger.info("AnthropicSummarizer initialized model=%s", self.model)

    def summarize(self, topic_id: int, posts: list[str], keywords: list[str]) -> TopicSummary:
        logger.info("Anthropic summarizer for Sampled Topic id %s: keyword is %s and posts are %s ", topic_id, keywords, posts) 
        cfg = self.config
        response = self._client.messages.create(
            model=self.model,
            max_tokens=cfg.max_topic_summarize_tokens,
            system=self.SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": self._build_user_prompt(posts, keywords)},
            ],
            temperature=cfg.temperature,
        )
        logger.info("Anthropic summarizer response %s", response)
        raw = response.content[0].text if response.content else ""
        ## raw text have ```json
        logger.info("Anthropic summarizer raw response.content %s", raw)
        return self._parse_response(topic_id, raw)

    def summarize_digest(self, topic_summaries: list[TopicSummary]) -> str:
        if not topic_summaries:
            return ""
        cfg = self.config
        response = self._client.messages.create(
            model=self.model,
            max_tokens=cfg.max_digest_tokens,
            system=_DIGEST_SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": self._build_digest_user_prompt(topic_summaries)},
            ],
            temperature=cfg.temperature,
        )
        raw = response.content[0].text if response.content else ""
        logger.info("Anthropic digest raw response %s", raw)
        return raw or super().summarize_digest(topic_summaries)
