"""Shared domain models."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModuleInstance:
    module_name: str
    instance_name: str
    line: int


@dataclass(frozen=True)
class HdlModule:
    name: str
    source: str
    start_line: int
    end_line: int
    ports: tuple[str, ...] = ()
    parameters: tuple[str, ...] = ()
    instances: tuple[ModuleInstance, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HdlModule:
        return cls(
            name=data["name"],
            source=data["source"],
            start_line=data["start_line"],
            end_line=data["end_line"],
            ports=tuple(data.get("ports", ())),
            parameters=tuple(data.get("parameters", ())),
            instances=tuple(ModuleInstance(**item) for item in data.get("instances", ())),
        )


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    source: str
    text: str
    start_line: int
    end_line: int
    kind: str = "text"
    module: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        source: str,
        text: str,
        start_line: int,
        end_line: int,
        kind: str = "text",
        module: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> DocumentChunk:
        identity = f"{source}:{start_line}:{end_line}:{module or ''}:{text}"
        chunk_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        return cls(
            chunk_id=chunk_id,
            source=source,
            text=text,
            start_line=start_line,
            end_line=end_line,
            kind=kind,
            module=module,
            metadata=metadata or {},
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DocumentChunk:
        return cls(**data)


@dataclass(frozen=True)
class SearchResult:
    chunk: DocumentChunk
    score: float
    lexical_score: float = 0.0
    dense_score: float = 0.0
    graph_score: float = 0.0

    @property
    def citation(self) -> str:
        return f"{self.chunk.source}:{self.chunk.start_line}-{self.chunk.end_line}"


@dataclass(frozen=True)
class Answer:
    text: str
    sources: tuple[SearchResult, ...]


@dataclass(frozen=True)
class ValidationResult:
    tool: str
    available: bool
    success: bool | None
    stdout: str = ""
    stderr: str = ""
