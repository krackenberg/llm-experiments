"""Thin wrapper around the Anthropic Python SDK.

Install the SDK with:  pip install anthropic
The import is deferred so the rest of the framework works without it
(e.g. when using MockClient).
"""

import os
from dataclasses import dataclass


@dataclass
class AnthropicConfig:
    model_id: str = "claude-sonnet-4-20250514"
    max_tokens: int = 1024
    temperature: float = 0.2


class AnthropicClient:
    """Chat client that calls the Anthropic Messages API."""

    def __init__(self, cfg: AnthropicConfig | None = None):
        try:
            import anthropic as _anthropic
        except ImportError:
            raise ImportError(
                "The 'anthropic' package is required for AnthropicClient. "
                "Install it with:  pip install anthropic"
            )
        self.cfg = cfg or AnthropicConfig()
        self._client = _anthropic.Anthropic(
            api_key=os.environ["ANTHROPIC_API_KEY"],
        )

    def chat(
        self,
        messages: list[dict],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        resp = self._client.messages.create(
            model=self.cfg.model_id,
            max_tokens=max_tokens or self.cfg.max_tokens,
            temperature=temperature if temperature is not None else self.cfg.temperature,
            messages=messages,
        )
        return resp.content[0].text
