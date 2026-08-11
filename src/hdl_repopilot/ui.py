"""Optional Gradio interface."""

from __future__ import annotations

from .config import Settings
from .pipeline import RepoPilot


def _format_results(results) -> str:
    sections = []
    for rank, result in enumerate(results, start=1):
        language = "systemverilog" if result.chunk.kind.startswith("hdl") else "text"
        sections.append(
            f"### {rank}. `{result.citation}` — score {result.score:.3f}\n\n"
            f"```{language}\n{result.chunk.text}\n```"
        )
    return "\n\n".join(sections) or "No matching context found."


def build_app(settings: Settings):
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("Install the 'ui' optional dependency to launch Gradio.") from exc

    pilot = RepoPilot(settings)
    pilot.load_index()

    def search(query: str) -> str:
        return _format_results(pilot.search(query))

    def ask(query: str) -> tuple[str, str]:
        answer = pilot.ask(query)
        return answer.text, _format_results(answer.sources)

    with gr.Blocks(title="HDL-RepoPilot") as app:
        gr.Markdown(
            "# HDL-RepoPilot\n"
            "Structure-aware retrieval and grounded answers for Verilog/SystemVerilog repositories."
        )
        with gr.Tab("Search"):
            search_query = gr.Textbox(label="Repository question", lines=2)
            search_button = gr.Button("Search", variant="primary")
            search_output = gr.Markdown()
            search_button.click(search, inputs=search_query, outputs=search_output)
        with gr.Tab("Ask with LLM"):
            ask_query = gr.Textbox(label="Repository question", lines=2)
            ask_button = gr.Button("Ask", variant="primary")
            answer_output = gr.Markdown(label="Answer")
            sources_output = gr.Markdown(label="Retrieved sources")
            ask_button.click(ask, inputs=ask_query, outputs=[answer_output, sources_output])
    return app


def launch(settings: Settings, *, host: str, port: int) -> None:
    build_app(settings).queue().launch(server_name=host, server_port=port, share=False)
