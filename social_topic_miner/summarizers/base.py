"""
Abstract base class for all summarisation backends.

To plug in a new LLM, subclass BaseSummarizer and implement ``summarize()``.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ..config import SummarisationConfig


@dataclass
class TopicSummary:
    topic_id: int
    category: str
    headline: str
    short_summary: str = ""
    """1-2 sentence summary for a card/preview view in the frontend."""
    long_summary: str = ""
    """Full paragraph for the detail/expanded view in the frontend."""
    key_points: list[str] = field(default_factory=list)
    raw_response: str = ""
    """The raw LLM output before JSON parsing (useful for debugging)."""


_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

_DIGEST_SYSTEM_PROMPT = (
    "You are a senior news editor. You will receive a list of trending topics "
    "from a user's social media feed.\n"
    "Write a neutral, informative digest of 5-10 sentences that captures the "
    "overall themes and key highlights across all topics.\n"
    "Plain text only — no bullet points, no headers, no markdown."
)


class BaseSummarizer(ABC):
    """
    Parameters
    ----------
    config:
        SummarisationConfig with generation knobs and output schema.
    """

    SYSTEM_PROMPT = (
        "You are a senior data analyst. Read the following social media posts "
        "about a single event or topic.\n"
        "Output ONLY a valid JSON object with exactly this schema:\n"
        "{\n"
        '  "category": "String (e.g., Entertainment, Sports, Politics, Tech, Lifestyle)",\n'
        '  "headline": "String (A concise, news-style headline summarising the event)",\n'
        '  "short_summary": "String (1-2 sentences for a preview card)",\n'
        '  "long_summary": "String (2-4 sentence paragraph for the detail view)",\n'
        '  "key_points": ["String (Key point 1)", "String (Key point 2)"]\n'
        "}\n"
        "Do not include any explanation, markdown formatting, or introductory text. "
        "Output raw JSON only."
    )

    def __init__(self, config: SummarisationConfig | None = None) -> None:
        self.config = config or SummarisationConfig()

    @abstractmethod
    def summarize(self, topic_id: int, posts: list[str], keywords: list[str]) -> TopicSummary:
        """
        Parameters
        ----------
        topic_id:
            Topic identifier (for tracking).
        posts:
            Representative post texts for this topic.
        keywords:
            Top c-TF-IDF keywords extracted by BERTopic.
        """

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def summarize_digest(self, topic_summaries: list[TopicSummary]) -> str:
        """
        Generate an overall digest string across all topics.

        Default implementation concatenates short_summaries (or headlines)
        without an LLM call.  Override in subclasses for AI-generated quality.
        """
        if not topic_summaries:
            return ""
        sentences: list[str] = []
        for ts in topic_summaries:
            text = ts.short_summary or ts.headline
            if text:
                sentences.append(text.rstrip(".") + ".")
        return " ".join(sentences)

    def _build_digest_user_prompt(self, topic_summaries: list[TopicSummary]) -> str:
        lines: list[str] = []
        for ts in topic_summaries:
            entry = f"({ts.category}) {ts.headline}"
            if ts.key_points:
                entry += ": " + "; ".join(ts.key_points[:3])
            lines.append(entry)
        return "Topics:\n" + "\n".join(lines) + "\n\nDigest:"

    def _build_user_prompt(self, posts: list[str], keywords: list[str]) -> str:
        posts_block = "\n".join(f"- {p}" for p in posts)
        kw_block = ", ".join(keywords)
        return (
            f"Topic keywords: {kw_block}\n\n"
            f"Posts:\n{posts_block}\n\n"
            "JSON Output:"
        )

    def _parse_response(self, topic_id: int, raw: str) -> TopicSummary:
        """Extract JSON from the raw LLM response and build a TopicSummary."""
        text = raw.strip()
        # Strip markdown code fences if present
        match = _JSON_BLOCK_RE.search(text)
        if match:
            text = match.group(1).strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Graceful fallback — return what we have
            return TopicSummary(
                topic_id=topic_id,
                category="Unknown",
                headline=text[:120],
                key_points=[],
                raw_response=raw,
            )

        return TopicSummary(
            topic_id=topic_id,
            category=data.get("category", "Unknown"),
            headline=data.get("headline", ""),
            short_summary=data.get("short_summary", ""),
            long_summary=data.get("long_summary", ""),
            key_points=data.get("key_points", []),
            raw_response=raw,
        )
