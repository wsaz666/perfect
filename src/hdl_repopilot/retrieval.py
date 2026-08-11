"""Hybrid lexical, dense, and module-graph retrieval."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

from .config import Settings
from .embeddings import EmbeddingProvider
from .graph import ModuleGraph
from .index import IndexSnapshot
from .models import SearchResult

TOKEN_PATTERN = re.compile(r"[A-Za-z_]\w*|\d+")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def _minmax(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    low, high = min(values), max(values)
    if math.isclose(low, high):
        return [1.0 if high > 0 else 0.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Query and index embedding dimensions do not match.")
    return sum(a * b for a, b in zip(left, right, strict=True))


class BM25:
    def __init__(self, documents: Sequence[str], *, k1: float = 1.5, b: float = 0.75):
        self.documents = [tokenize(document) for document in documents]
        self.k1 = k1
        self.b = b
        self.average_length = (
            sum(len(document) for document in self.documents) / len(self.documents)
            if self.documents
            else 0.0
        )
        self.document_frequency: Counter[str] = Counter()
        for document in self.documents:
            self.document_frequency.update(set(document))

    def score(self, query: str) -> list[float]:
        query_tokens = tokenize(query)
        document_count = len(self.documents)
        scores = []
        for document in self.documents:
            frequencies = Counter(document)
            score = 0.0
            for token in query_tokens:
                df = self.document_frequency[token]
                if not df or not frequencies[token]:
                    continue
                idf = math.log(1 + (document_count - df + 0.5) / (df + 0.5))
                denominator = frequencies[token] + self.k1 * (
                    1 - self.b + self.b * len(document) / max(self.average_length, 1.0)
                )
                score += idf * frequencies[token] * (self.k1 + 1) / denominator
            scores.append(score)
        return scores


class HybridRetriever:
    def __init__(
        self,
        snapshot: IndexSnapshot,
        embedding_provider: EmbeddingProvider,
        settings: Settings,
    ):
        self.snapshot = snapshot
        self.embedding_provider = embedding_provider
        self.settings = settings
        self.bm25 = BM25([chunk.text for chunk in snapshot.chunks])
        self.graph = ModuleGraph(snapshot.modules)

    def search(self, query: str, *, top_k: int | None = None) -> list[SearchResult]:
        if not query.strip() or not self.snapshot.chunks:
            return []
        lexical = _minmax(self.bm25.score(query))
        query_vector = self.embedding_provider.embed([query])[0]
        dense = _minmax([_dot(query_vector, vector) for vector in self.snapshot.embeddings])

        initial = sorted(
            range(len(self.snapshot.chunks)),
            key=lambda index: lexical[index] + dense[index],
            reverse=True,
        )[: max(5, top_k or self.settings.top_k)]
        seed_modules = {
            self.snapshot.chunks[index].module
            for index in initial
            if self.snapshot.chunks[index].module
        }
        graph_distances = self.graph.expand(seed_modules, depth=self.settings.graph_depth)
        graph_scores = []
        for chunk in self.snapshot.chunks:
            distance = graph_distances.get(chunk.module) if chunk.module else None
            graph_scores.append(1.0 / (distance + 1) if distance is not None else 0.0)

        results = []
        for index, chunk in enumerate(self.snapshot.chunks):
            score = (
                self.settings.lexical_weight * lexical[index]
                + self.settings.dense_weight * dense[index]
                + self.settings.graph_weight * graph_scores[index]
            )
            results.append(
                SearchResult(
                    chunk=chunk,
                    score=score,
                    lexical_score=lexical[index],
                    dense_score=dense[index],
                    graph_score=graph_scores[index],
                )
            )
        results.sort(key=lambda item: item.score, reverse=True)
        return results[: top_k or self.settings.top_k]
