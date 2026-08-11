"""Pluggable dense embedding providers with an offline deterministic default."""

from __future__ import annotations

import hashlib
import math
import os
import re
from collections.abc import Sequence
from typing import Protocol

from .config import Settings

TOKEN_PATTERN = re.compile(r"[A-Za-z_]\w*|\d+")


class EmbeddingProvider(Protocol):
    dimensions: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else vector


class HashEmbeddingProvider:
    """Dependency-free feature hashing for smoke tests and offline demos.

    This provider is not intended to replace a trained embedding model. It makes
    the complete indexing/retrieval framework runnable before model selection.
    """

    def __init__(self, dimensions: int = 512):
        self.dimensions = dimensions

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        output = []
        for text in texts:
            vector = [0.0] * self.dimensions
            for token in TOKEN_PATTERN.findall(text.lower()):
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                bucket = int.from_bytes(digest[:4], "little") % self.dimensions
                sign = 1.0 if digest[4] & 1 else -1.0
                vector[bucket] += sign
            output.append(_normalize(vector))
        return output


class OpenAIEmbeddingProvider:
    def __init__(self, settings: Settings):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the 'llm' optional dependency for API embeddings.") from exc
        api_key = os.getenv(settings.api_key_env)
        if not api_key:
            raise RuntimeError(f"Set {settings.api_key_env} before using API embeddings.")
        self.client = OpenAI(api_key=api_key, base_url=settings.llm_base_url)
        self.model = settings.embedding_model
        self.dimensions = settings.embedding_dimensions

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        response = self.client.embeddings.create(
            model=self.model,
            input=list(texts),
            dimensions=self.dimensions,
            encoding_format="float",
        )
        return [_normalize(list(item.embedding)) for item in response.data]


class SentenceTransformerEmbeddingProvider:
    def __init__(self, settings: Settings):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Install the 'local-embeddings' optional dependency for this provider."
            ) from exc
        self.model = SentenceTransformer(settings.embedding_model)
        self.dimensions = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self.model.encode(list(texts), normalize_embeddings=True)
        return vectors.tolist()


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    if settings.embedding_provider == "hash":
        return HashEmbeddingProvider(settings.embedding_dimensions)
    if settings.embedding_provider == "openai":
        return OpenAIEmbeddingProvider(settings)
    if settings.embedding_provider == "sentence-transformers":
        return SentenceTransformerEmbeddingProvider(settings)
    raise ValueError(f"Unsupported embedding provider: {settings.embedding_provider}")
