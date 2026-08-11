"""Command-line interface for HDL-RepoPilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import Settings
from .graph import ModuleGraph
from .pipeline import RepoPilot
from .validator import HdlValidator


def _settings(args: argparse.Namespace) -> Settings:
    return Settings.from_env(index_dir=args.index_dir)


def _print_results(results) -> None:
    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}. {result.citation} score={result.score:.3f} "
            f"lexical={result.lexical_score:.3f} dense={result.dense_score:.3f} "
            f"graph={result.graph_score:.3f}"
        )
        preview = " ".join(result.chunk.text.split())[:240]
        print(f"   {preview}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hdl-repopilot",
        description="Structure-aware retrieval and grounded Q&A for HDL repositories.",
    )
    parser.add_argument("--index-dir", type=Path, default=Path(".repopilot"))
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Index an HDL repository.")
    index_parser.add_argument("repository", type=Path)

    search_parser = subparsers.add_parser("search", help="Search without calling an LLM.")
    search_parser.add_argument("query")
    search_parser.add_argument("--top-k", type=int)

    ask_parser = subparsers.add_parser("ask", help="Generate a grounded answer with citations.")
    ask_parser.add_argument("query")

    subparsers.add_parser("graph", help="Print the module dependency graph.")

    validate_parser = subparsers.add_parser("validate", help="Lint or compile an HDL file.")
    validate_parser.add_argument("file", type=Path)
    validate_parser.add_argument(
        "--tool",
        choices=["auto", "verilator", "iverilog"],
        default="auto",
    )

    serve_parser = subparsers.add_parser("serve", help="Launch the optional Gradio UI.")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=7860)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = _settings(args)

    if args.command == "index":
        pilot = RepoPilot(settings)
        result = pilot.index_repository(args.repository)
        print(
            f"Indexed {result.files_seen} files, {len(result.modules)} modules, "
            f"and {len(result.chunks)} chunks into {settings.index_dir}."
        )
        return

    if args.command == "validate":
        result = HdlValidator(args.tool).validate_file(args.file)
        print(json.dumps(result.__dict__, ensure_ascii=False, indent=2))
        raise SystemExit(0 if result.success is not False else 1)

    pilot = RepoPilot(settings)
    snapshot = pilot.load_index()
    if args.command == "search":
        _print_results(pilot.search(args.query, top_k=args.top_k))
    elif args.command == "ask":
        answer = pilot.ask(args.query)
        print(answer.text)
        print("\nSources:")
        for source in answer.sources:
            print(f"- {source.citation}")
    elif args.command == "graph":
        lines = ModuleGraph(snapshot.modules).dependency_lines()
        print("\n".join(lines) if lines else "No module dependencies found.")
    elif args.command == "serve":
        from .ui import launch

        launch(settings, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
