"""Repository ingestion with module-aware HDL chunking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .hdl_parser import module_source, parse_hdl_modules
from .models import DocumentChunk, HdlModule

HDL_EXTENSIONS = {".v", ".sv", ".vh", ".svh"}
TEXT_EXTENSIONS = HDL_EXTENSIONS | {".txt", ".md", ".rst", ".pdf"}
IGNORED_DIRECTORIES = {".git", ".repopilot", ".venv", "node_modules", "build", "dist"}


@dataclass(frozen=True)
class IngestionResult:
    chunks: tuple[DocumentChunk, ...]
    modules: tuple[HdlModule, ...]
    files_seen: int


def _window_text(
    text: str,
    *,
    source: str,
    start_line: int,
    chunk_chars: int,
    overlap: int,
    kind: str,
    module: str | None = None,
) -> list[DocumentChunk]:
    if not text.strip():
        return []
    chunks = []
    cursor = 0
    step = chunk_chars - overlap
    while cursor < len(text):
        end = min(cursor + chunk_chars, len(text))
        piece = text[cursor:end].strip()
        if piece:
            local_start_line = text.count("\n", 0, cursor)
            local_end_line = text.count("\n", 0, end)
            chunks.append(
                DocumentChunk.create(
                    source=source,
                    text=piece,
                    start_line=start_line + local_start_line,
                    end_line=start_line + local_end_line,
                    kind=kind,
                    module=module,
                )
            )
        if end == len(text):
            break
        cursor += step
    return chunks


def _read_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError(
                "Install the 'pdf' optional dependency to ingest PDF files."
            ) from exc
        return "\n\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    raw = path.read_bytes()
    for encoding in ("utf-8", "utf-8-sig", "gb18030", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def ingest_repository(
    root: Path,
    *,
    chunk_chars: int = 2_000,
    overlap: int = 200,
) -> IngestionResult:
    """Read supported files below ``root`` and create structure-aware chunks."""
    root = root.resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Repository directory not found: {root}")

    all_chunks: list[DocumentChunk] = []
    all_modules: list[HdlModule] = []
    files_seen = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or any(part in IGNORED_DIRECTORIES for part in path.parts):
            continue
        if path.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        files_seen += 1
        source = path.relative_to(root).as_posix()
        text = _read_text(path)

        if path.suffix.lower() in HDL_EXTENSIONS:
            modules = parse_hdl_modules(text, source)
            all_modules.extend(modules)
            if modules:
                for module in modules:
                    all_chunks.extend(
                        _window_text(
                            module_source(text, module),
                            source=source,
                            start_line=module.start_line,
                            chunk_chars=chunk_chars,
                            overlap=overlap,
                            kind="hdl_module",
                            module=module.name,
                        )
                    )
                continue

        all_chunks.extend(
            _window_text(
                text,
                source=source,
                start_line=1,
                chunk_chars=chunk_chars,
                overlap=overlap,
                kind="hdl" if path.suffix.lower() in HDL_EXTENSIONS else "text",
            )
        )

    return IngestionResult(
        chunks=tuple(all_chunks),
        modules=tuple(all_modules),
        files_seen=files_seen,
    )
