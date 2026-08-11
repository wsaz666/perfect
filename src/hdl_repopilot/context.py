"""Context packing with stable source citations."""

from __future__ import annotations

from collections.abc import Sequence

from .models import SearchResult


def pack_context(results: Sequence[SearchResult], max_chars: int) -> str:
    sections = []
    used = 0
    for result in results:
        header = f"[SOURCE {result.citation}]"
        section = f"{header}\n{result.chunk.text.strip()}"
        if used + len(section) > max_chars:
            remaining = max_chars - used
            if remaining > len(header) + 80:
                sections.append(section[:remaining])
            break
        sections.append(section)
        used += len(section)
    return "\n\n".join(sections)
