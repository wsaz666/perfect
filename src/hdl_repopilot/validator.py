"""HDL compiler/linter integration."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from .models import ValidationResult


class HdlValidator:
    def __init__(self, preferred_tool: str = "auto", timeout: int = 30):
        self.preferred_tool = preferred_tool
        self.timeout = timeout

    def _resolve_tool(self) -> tuple[str, str] | None:
        candidates = (
            ("verilator", shutil.which("verilator")),
            ("iverilog", shutil.which("iverilog")),
        )
        if self.preferred_tool != "auto":
            return next(
                (
                    item
                    for item in candidates
                    if item[0] == self.preferred_tool and item[1]
                ),
                None,
            )
        return next((item for item in candidates if item[1]), None)

    def validate_code(self, code: str, *, suffix: str = ".sv") -> ValidationResult:
        resolved = self._resolve_tool()
        if resolved is None:
            return ValidationResult(
                tool=self.preferred_tool,
                available=False,
                success=None,
                stderr="Install Verilator or Icarus Verilog to enable validation.",
            )
        tool_name, executable = resolved
        with tempfile.TemporaryDirectory(prefix="hdl-repopilot-") as tmp:
            source = Path(tmp) / f"candidate{suffix}"
            source.write_text(code, encoding="utf-8")
            if tool_name == "verilator":
                command = [executable, "--lint-only", "-Wall", str(source)]
            else:
                output = Path(tmp) / "candidate.out"
                command = [executable, "-g2012", "-o", str(output), str(source)]
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
        return ValidationResult(
            tool=tool_name,
            available=True,
            success=result.returncode == 0,
            stdout=result.stdout.strip(),
            stderr=result.stderr.strip(),
        )

    def validate_file(self, path: Path) -> ValidationResult:
        return self.validate_code(path.read_text(encoding="utf-8"), suffix=path.suffix or ".sv")
