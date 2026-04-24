"""
Local Llama (HuggingFace transformers) summarisation backend.

Requires: pip install transformers accelerate bitsandbytes torch
Supports any instruction-tuned causal LM available on HuggingFace Hub,
e.g. meta-llama/Meta-Llama-3.1-8B-Instruct.
"""

from __future__ import annotations

from ..config import SummarisationConfig
from .base import BaseSummarizer, TopicSummary


class LlamaSummarizer(BaseSummarizer):
    """
    Parameters
    ----------
    model_id:
        HuggingFace model ID (e.g. ``"meta-llama/Meta-Llama-3.1-8B-Instruct"``).
    hf_token:
        HuggingFace access token for gated models (falls back to ``HF_TOKEN`` env var).
    load_in_4bit:
        Quantise to 4-bit with bitsandbytes (requires a CUDA GPU).
    config:
        SummarisationConfig with generation knobs.
    device_map:
        Passed to ``from_pretrained`` — use ``"auto"`` for multi-GPU or
        ``"cpu"`` for CPU-only inference.
    """

    def __init__(
        self,
        model_id: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
        hf_token: str | None = None,
        load_in_4bit: bool = True,
        config: SummarisationConfig | None = None,
        device_map: str = "auto",
    ) -> None:
        try:
            import torch  # noqa: F401
            from transformers import (  # noqa: F401
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
                pipeline,
            )
        except ImportError as exc:
            raise ImportError(
                "transformers, accelerate, bitsandbytes and torch are required. "
                "Install with: pip install transformers accelerate bitsandbytes torch"
            ) from exc

        super().__init__(config)
        self._load(model_id, hf_token, load_in_4bit, device_map)

    # ------------------------------------------------------------------

    def summarize(self, topic_id: int, posts: list[str], keywords: list[str]) -> TopicSummary:
        cfg = self.config
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": self._build_user_prompt(posts, keywords)},
        ]
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        outputs = self._pipe(
            prompt,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            do_sample=cfg.do_sample,
        )
        raw = outputs[0]["generated_text"][len(prompt):].strip()
        return self._parse_response(topic_id, raw)

    # ------------------------------------------------------------------

    def _load(
        self,
        model_id: str,
        hf_token: str | None,
        load_in_4bit: bool,
        device_map: str,
    ) -> None:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            pipeline,
        )

        bnb_config = None
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )

        self._tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map=device_map,
            token=hf_token,
        )
        self._pipe = pipeline("text-generation", model=model, tokenizer=self._tokenizer)
