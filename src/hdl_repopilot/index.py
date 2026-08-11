"""Portable on-disk index snapshot."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import DocumentChunk, HdlModule

INDEX_VERSION = 1


@dataclass(frozen=True)
class IndexSnapshot:
    repository: str
    chunks: tuple[DocumentChunk, ...]
    modules: tuple[HdlModule, ...]
    embeddings: tuple[tuple[float, ...], ...]
    embedding_provider: str
    embedding_model: str
    embedding_dimensions: int

    def validate(self) -> None:
        if len(self.chunks) != len(self.embeddings):
            raise ValueError("Chunk and embedding counts do not match.")
        if self.embeddings and len(self.embeddings[0]) != self.embedding_dimensions:
            raise ValueError("Embedding dimensions do not match index metadata.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": INDEX_VERSION,
            "repository": self.repository,
            "embedding": {
                "provider": self.embedding_provider,
                "model": self.embedding_model,
                "dimensions": self.embedding_dimensions,
            },
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "modules": [module.to_dict() for module in self.modules],
            "embeddings": [list(vector) for vector in self.embeddings],
        }

    def save(self, index_dir: Path) -> Path:
        self.validate()
        index_dir.mkdir(parents=True, exist_ok=True)
        destination = index_dir / "index.json"
        temporary = index_dir / "index.json.tmp"
        temporary.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        temporary.replace(destination)
        return destination

    @classmethod
    def load(cls, index_dir: Path) -> IndexSnapshot:
        source = index_dir / "index.json"
        if not source.exists():
            raise FileNotFoundError(f"Index not found: {source}. Run the index command first.")
        data = json.loads(source.read_text(encoding="utf-8"))
        if data.get("version") != INDEX_VERSION:
            raise ValueError(f"Unsupported index version: {data.get('version')}")
        embedding = data["embedding"]
        snapshot = cls(
            repository=data["repository"],
            chunks=tuple(DocumentChunk.from_dict(item) for item in data["chunks"]),
            modules=tuple(HdlModule.from_dict(item) for item in data["modules"]),
            embeddings=tuple(tuple(vector) for vector in data["embeddings"]),
            embedding_provider=embedding["provider"],
            embedding_model=embedding["model"],
            embedding_dimensions=embedding["dimensions"],
        )
        snapshot.validate()
        return snapshot
