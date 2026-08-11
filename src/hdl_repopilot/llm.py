"""Provider-neutral language-model interface."""

from __future__ import annotations

import os
from typing import Protocol

from .config import Settings


class LanguageModel(Protocol):
    def generate(self, *, system_prompt: str, user_prompt: str) -> str: ...


class OpenAICompatibleLanguageModel:
    def __init__(self, settings: Settings):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "Install the 'llm' optional dependency to generate answers."
            ) from exc
        api_key = os.getenv(settings.api_key_env)
        if not api_key:
            raise RuntimeError(f"Set {settings.api_key_env} before generating answers.")
        self.client = OpenAI(api_key=api_key, base_url=settings.llm_base_url)
        self.model = settings.llm_model

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
        )
        return response.choices[0].message.content.strip()
