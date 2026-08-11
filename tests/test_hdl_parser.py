import unittest

from hdl_repopilot.graph import ModuleGraph
from hdl_repopilot.hdl_parser import parse_hdl_modules

HDL = """module child #(parameter WIDTH = 8) (
    input logic clk,
    output logic [WIDTH-1:0] data
);
endmodule

module top(input logic clk, output logic [7:0] data);
    always_ff @(posedge clk) begin
        if (clk) data <= '0;
    end
    child #(.WIDTH(8)) u_child (.clk(clk), .data(data));
endmodule
"""


class HdlParserTests(unittest.TestCase):
    def test_extracts_module_structure_and_instances(self):
        modules = parse_hdl_modules(HDL, "rtl/top.sv")
        self.assertEqual([module.name for module in modules], ["child", "top"])
        self.assertIn("WIDTH", modules[0].parameters)
        self.assertIn("clk", modules[0].ports)
        self.assertEqual(modules[0].instances, ())
        self.assertEqual(len(modules[1].instances), 1)
        self.assertEqual(modules[1].instances[0].module_name, "child")
        self.assertEqual(modules[1].instances[0].instance_name, "u_child")

    def test_builds_bidirectional_dependency_expansion(self):
        graph = ModuleGraph(parse_hdl_modules(HDL, "rtl/top.sv"))
        self.assertEqual(graph.outgoing["top"], {"child"})
        self.assertEqual(graph.expand(["child"], depth=1), {"child": 0, "top": 1})


if __name__ == "__main__":
    unittest.main()
