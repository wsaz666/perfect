"""HDL-RepoPilot: structure-aware retrieval for Verilog repositories."""

from .config import Settings
from .pipeline import RepoPilot

__all__ = ["RepoPilot", "Settings"]
__version__ = "0.1.0"
