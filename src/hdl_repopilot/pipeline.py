"""High-level indexing, retrieval, and grounded-answer pipeline."""

from __future__ import annotations

from pathlib import Path

from .config import Settings
from .context import pack_context
from .embeddings import EmbeddingProvider, build_embedding_provider
from .index import IndexSnapshot
from .ingestion import IngestionResult, ingest_repository
from .llm import LanguageModel, OpenAICompatibleLanguageModel
from .models import Answer, SearchResult
from .retrieval import HybridRetriever

SYSTEM_PROMPT = """You are HDL-RepoPilot, a senior Verilog/SystemVerilog engineer.
Answer only from the supplied repository context. Cite factual statements using
the exact [SOURCE path:start-end] markers. If the context is insufficient, say
what is missing instead of inventing module behavior."""


class RepoPilot:
    def __init__(
        self,
        settings: Settings,
        *,
        embedding_provider: EmbeddingProvider | None = None,
        language_model: LanguageModel | None = None,
    ):
        self.settings = settings
        self.embedding_provider = embedding_provider or build_embedding_provider(settings)
        self.language_model = language_model
        self.snapshot: IndexSnapshot | None = None
        self.retriever: HybridRetriever | None = None

    def index_repository(self, repository: Path) -> IngestionResult:
        ingestion = ingest_repository(
            repository,
            chunk_chars=self.settings.chunk_chars,
            overlap=self.settings.chunk_overlap,
        )
        embeddings = self.embedding_provider.embed([chunk.text for chunk in ingestion.chunks])
        snapshot = IndexSnapshot(
            repository=str(repository.resolve()),
            chunks=ingestion.chunks,
            modules=ingestion.modules,
            embeddings=tuple(tuple(vector) for vector in embeddings),
            embedding_provider=self.settings.embedding_provider,
            embedding_model=self.settings.embedding_model,
            embedding_dimensions=self.embedding_provider.dimensions,
        )
        snapshot.save(self.settings.index_dir)
        self._attach(snapshot)
        return ingestion

    def load_index(self) -> IndexSnapshot:
        snapshot = IndexSnapshot.load(self.settings.index_dir)
        if snapshot.embedding_provider != self.settings.embedding_provider:
            raise ValueError(
                "The configured embedding provider does not match the saved index: "
                f"{self.settings.embedding_provider!r} != {snapshot.embedding_provider!r}."
            )
        self._attach(snapshot)
        return snapshot

    def _attach(self, snapshot: IndexSnapshot) -> None:
        self.snapshot = snapshot
        self.retriever = HybridRetriever(snapshot, self.embedding_provider, self.settings)

    def search(self, query: str, *, top_k: int | None = None) -> list[SearchResult]:
        if self.retriever is None:
            self.load_index()
        return self.retriever.search(query, top_k=top_k)  # type: ignore[union-attr]

    def ask(self, query: str) -> Answer:
        results = self.search(query)
        context = pack_context(results, self.settings.context_chars)
        language_model = self.language_model or OpenAICompatibleLanguageModel(self.settings)
        answer = language_model.generate(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=f"Question:\n{query}\n\nRepository context:\n{context}",
        )
        return Answer(text=answer, sources=tuple(results))
