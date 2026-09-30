"""Offline temporal correspondence from per-frame interaction tokens."""
from .aligner import FilterConfig, OfflineProgressAligner
from .data import TokenSequence

__all__ = ["FilterConfig", "OfflineProgressAligner", "TokenSequence"]
__version__ = "0.1.0"
