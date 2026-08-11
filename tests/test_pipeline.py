import tempfile
import unittest
from pathlib import Path

from hdl_repopilot.config import Settings
from hdl_repopilot.embeddings import HashEmbeddingProvider
from hdl_repopilot.pipeline import RepoPilot
from hdl_repopilot.validator import HdlValidator


class FakeLanguageModel:
    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        return "The counter is defined in [SOURCE counter.sv:1-3]."


class PipelineTests(unittest.TestCase):
    def test_index_search_and_grounded_answer_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            index_dir = Path(tmp) / "index"
            root.mkdir()
            (root / "counter.sv").write_text(
                "module counter(input logic clk, output logic q);\n"
                "always_ff @(posedge clk) q <= ~q;\nendmodule\n",
                encoding="utf-8",
            )
            settings = Settings(index_dir=index_dir, embedding_dimensions=64)
            model = FakeLanguageModel()
            pilot = RepoPilot(
                settings,
                embedding_provider=HashEmbeddingProvider(64),
                language_model=model,
            )
            ingestion = pilot.index_repository(root)
            answer = pilot.ask("Where is the counter implemented?")

            self.assertEqual(len(ingestion.modules), 1)
            self.assertTrue((index_dir / "index.json").exists())
            self.assertIn("counter.sv", answer.text)
            self.assertIn("[SOURCE counter.sv:1-3]", model.user_prompt)

    def test_validator_reports_unavailable_unknown_tool(self):
        result = HdlValidator("not-installed").validate_code("module x; endmodule")
        self.assertFalse(result.available)
        self.assertIsNone(result.success)


if __name__ == "__main__":
    unittest.main()
