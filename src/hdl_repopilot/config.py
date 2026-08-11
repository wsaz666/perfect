"""Application configuration loaded from environment variables or CLI flags."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value is not None else default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value is not None else default


@dataclass(frozen=True)
class Settings:
    """Runtime settings with safe, local-first defaults."""

    index_dir: Path = Path(".repopilot")
    embedding_provider: str = "hash"
    embedding_model: str = "text-embedding-v3"
    embedding_dimensions: int = 512
    llm_base_url: str = "https://api.siliconflow.cn/v1"
    llm_model: str = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
    api_key_env: str = "LLM_API_KEY"
    lexical_weight: float = 0.45
    dense_weight: float = 0.45
    graph_weight: float = 0.10
    top_k: int = 8
    graph_depth: int = 1
    chunk_chars: int = 2_000
    chunk_overlap: int = 200
    context_chars: int = 14_000

    @classmethod
    def from_env(cls, *, index_dir: Path | None = None) -> Settings:
        settings = cls(
            index_dir=index_dir or Path(os.getenv("REPOPILOT_INDEX_DIR", ".repopilot")),
            embedding_provider=os.getenv("EMBEDDING_PROVIDER", "hash"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-v3"),
            embedding_dimensions=_env_int("EMBEDDING_DIMENSIONS", 512),
            llm_base_url=os.getenv("LLM_BASE_URL", "https://api.siliconflow.cn/v1"),
            llm_model=os.getenv("LLM_MODEL", "Qwen/Qwen3-Coder-30B-A3B-Instruct"),
            api_key_env=os.getenv("LLM_API_KEY_ENV", "LLM_API_KEY"),
            lexical_weight=_env_float("LEXICAL_WEIGHT", 0.45),
            dense_weight=_env_float("DENSE_WEIGHT", 0.45),
            graph_weight=_env_float("GRAPH_WEIGHT", 0.10),
            top_k=_env_int("TOP_K", 8),
            graph_depth=_env_int("GRAPH_DEPTH", 1),
            chunk_chars=_env_int("CHUNK_CHARS", 2_000),
            chunk_overlap=_env_int("CHUNK_OVERLAP", 200),
            context_chars=_env_int("CONTEXT_CHARS", 14_000),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.embedding_provider not in {"hash", "openai", "sentence-transformers"}:
            raise ValueError(f"Unsupported embedding provider: {self.embedding_provider}")
        if min(self.lexical_weight, self.dense_weight, self.graph_weight) < 0:
            raise ValueError("Retrieval weights cannot be negative.")
        if self.lexical_weight + self.dense_weight + self.graph_weight <= 0:
            raise ValueError("At least one retrieval weight must be positive.")
        if not 0 <= self.chunk_overlap < self.chunk_chars:
            raise ValueError("chunk_overlap must be non-negative and smaller than chunk_chars.")
