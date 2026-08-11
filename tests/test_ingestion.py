import tempfile
import unittest
from pathlib import Path

from hdl_repopilot.ingestion import ingest_repository


class IngestionTests(unittest.TestCase):
    def test_module_aware_chunks_keep_source_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "counter.sv").write_text(
                "module counter(input logic clk, output logic q);\n"
                "always_ff @(posedge clk) q <= ~q;\n"
                "endmodule\n",
                encoding="utf-8",
            )
            result = ingest_repository(root, chunk_chars=200, overlap=20)

            self.assertEqual(result.files_seen, 1)
            self.assertEqual(result.modules[0].name, "counter")
            self.assertEqual(result.chunks[0].module, "counter")
            self.assertEqual(result.chunks[0].source, "counter.sv")
            self.assertGreaterEqual(result.chunks[0].end_line, 3)


if __name__ == "__main__":
    unittest.main()
