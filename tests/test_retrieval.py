import tempfile
import unittest
from pathlib import Path

from hdl_repopilot.config import Settings
from hdl_repopilot.embeddings import HashEmbeddingProvider
from hdl_repopilot.index import IndexSnapshot
from hdl_repopilot.models import DocumentChunk, HdlModule, ModuleInstance
from hdl_repopilot.retrieval import HybridRetriever


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.chunks = (
            DocumentChunk.create(
                source="rtl/fifo.sv",
                text="module fifo parameter DEPTH stores queued words",
                start_line=1,
                end_line=20,
                kind="hdl_module",
                module="fifo",
            ),
            DocumentChunk.create(
                source="rtl/top.sv",
                text="module top instantiates fifo as u_fifo",
                start_line=1,
                end_line=15,
                kind="hdl_module",
                module="top",
            ),
            DocumentChunk.create(
                source="rtl/uart.sv",
                text="module uart receives serial data",
                start_line=1,
                end_line=12,
                kind="hdl_module",
                module="uart",
            ),
        )
        self.modules = (
            HdlModule("fifo", "rtl/fifo.sv", 1, 20),
            HdlModule(
                "top",
                "rtl/top.sv",
                1,
                15,
                instances=(ModuleInstance("fifo", "u_fifo", 5),),
            ),
            HdlModule("uart", "rtl/uart.sv", 1, 12),
        )
        self.provider = HashEmbeddingProvider(64)

    def test_hybrid_search_prefers_relevant_hdl(self):
        snapshot = IndexSnapshot(
            repository="demo",
            chunks=self.chunks,
            modules=self.modules,
            embeddings=tuple(
                tuple(item) for item in self.provider.embed([c.text for c in self.chunks])
            ),
            embedding_provider="hash",
            embedding_model="hash",
            embedding_dimensions=64,
        )
        settings = Settings(
            index_dir=Path(tempfile.gettempdir()),
            embedding_dimensions=64,
            top_k=2,
        )
        results = HybridRetriever(snapshot, self.provider, settings).search("FIFO queue depth")
        self.assertEqual(results[0].chunk.module, "fifo")
        self.assertIn("top", {result.chunk.module for result in results})
        self.assertEqual(results[0].citation, "rtl/fifo.sv:1-20")


if __name__ == "__main__":
    unittest.main()
