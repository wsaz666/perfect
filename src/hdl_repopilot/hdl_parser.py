"""Lightweight Verilog/SystemVerilog structure extraction.

This regex parser establishes the framework contract. It deliberately keeps the
parser behind a small interface so a tree-sitter or slang backend can replace it
without changing ingestion or retrieval code.
"""

from __future__ import annotations

import re
from pathlib import Path

from .models import HdlModule, ModuleInstance

MODULE_PATTERN = re.compile(
    r"\bmodule\s+(?P<name>[A-Za-z_]\w*)\b(?P<body>.*?)\bendmodule\b",
    flags=re.DOTALL,
)
PORT_PATTERN = re.compile(
    r"\b(?:input|output|inout)\b(?:\s+(?:wire|reg|logic|signed|unsigned))*"
    r"(?:\s*\[[^\]]+\])?\s+(?P<names>[^;\)]+)",
    flags=re.IGNORECASE,
)
PARAMETER_PATTERN = re.compile(
    r"\b(?:parameter|localparam)\b(?:\s+\w+)?\s+(?P<name>[A-Za-z_]\w*)",
    flags=re.IGNORECASE,
)
INSTANCE_PATTERN = re.compile(
    r"(?m)^\s*(?P<module>[A-Za-z_]\w*)\b\s*"
    r"(?:#\s*\((?:[^()]|\([^()]*\))*\)\s*)?"
    r"(?P<instance>[A-Za-z_]\w*)\b\s*\(",
)
INSTANCE_KEYWORDS = {
    "always",
    "always_comb",
    "always_ff",
    "always_latch",
    "assign",
    "case",
    "else",
    "for",
    "function",
    "generate",
    "if",
    "module",
    "task",
    "while",
}


def _line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _split_names(raw_names: str) -> list[str]:
    names = []
    for raw_name in raw_names.split(","):
        name = raw_name.split("=")[0].strip()
        match = re.search(r"([A-Za-z_]\w*)$", name)
        if match:
            names.append(match.group(1))
    return names


def parse_hdl_modules(text: str, source: str | Path) -> list[HdlModule]:
    """Extract modules, ports, parameters, and module instances."""
    source_name = Path(source).as_posix()
    modules: list[HdlModule] = []
    for match in MODULE_PATTERN.finditer(text):
        module_text = match.group(0)
        body = match.group("body")
        body_offset = match.start("body")
        ports = []
        for port_match in PORT_PATTERN.finditer(module_text):
            ports.extend(_split_names(port_match.group("names")))
        parameters = tuple(
            dict.fromkeys(item.group("name") for item in PARAMETER_PATTERN.finditer(module_text))
        )
        instances = []
        for instance_match in INSTANCE_PATTERN.finditer(body):
            module_name = instance_match.group("module")
            if module_name.lower() in INSTANCE_KEYWORDS:
                continue
            instances.append(
                ModuleInstance(
                    module_name=module_name,
                    instance_name=instance_match.group("instance"),
                    line=_line_number(text, body_offset + instance_match.start()),
                )
            )
        modules.append(
            HdlModule(
                name=match.group("name"),
                source=source_name,
                start_line=_line_number(text, match.start()),
                end_line=_line_number(text, match.end()),
                ports=tuple(dict.fromkeys(ports)),
                parameters=parameters,
                instances=tuple(instances),
            )
        )
    return modules


def module_source(text: str, module: HdlModule) -> str:
    lines = text.splitlines()
    return "\n".join(lines[module.start_line - 1 : module.end_line])
