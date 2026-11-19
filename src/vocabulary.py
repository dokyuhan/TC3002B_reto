"""Shared AST state vocabulary and language constants across all layers."""

from __future__ import annotations

DEFAULT_AST_STATES: tuple[str, ...] = (
    "IMPORT",
    "FUNCTION",
    "CLASS",
    "ASSIGN",
    "IF",
    "FOR",
    "WHILE",
    "CALL",
    "RETURN",
    "EXCEPTION",
    "OP",
    "OTHER",
)

NUM_STATES: int = len(DEFAULT_AST_STATES)
# Negative sentinel keeps valid state IDs contiguous across 0..K-1 matrix indices.
PAD_TOKEN: int = -1

STATE_TO_ID: dict[str, int] = {name: idx for idx, name in enumerate(DEFAULT_AST_STATES)}
SUPPORTED_LANGUAGES: frozenset[str] = frozenset({"java", "python", "cpp"})
