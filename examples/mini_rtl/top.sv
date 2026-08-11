module top (
    input  logic       clk,
    input  logic       rst_n,
    input  logic       enable,
    output logic [7:0] count
);
    counter #(.WIDTH(8)) u_counter (
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),
        .count(count)
    );
endmodule
