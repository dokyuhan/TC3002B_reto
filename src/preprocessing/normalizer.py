"""Shared AST state vocabulary for preprocessing and feature extraction."""

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
PAD_TOKEN: int = -1
