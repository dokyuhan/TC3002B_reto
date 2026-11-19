"""Contrato de datos compartido entre Paso 1, Paso 2 y Paso 3."""

from __future__ import annotations

# Propuesta inicial de 12 estados AST (indexación base 0).
# Debe congelarse en conjunto con el Integrante 1 antes de procesar el dataset real.
DEFAULT_AST_STATES: tuple[str, ...] = (
    "IMPORT",       # 0: Import, ImportFrom
    "FUNCTION",     # 1: FunctionDef, AsyncFunctionDef, Lambda
    "CLASS",        # 2: ClassDef
    "ASSIGN",       # 3: Assign, AnnAssign, AugAssign
    "IF",           # 4: If, IfExp
    "FOR",          # 5: For, AsyncFor, Comprehension
    "WHILE",        # 6: While
    "CALL",         # 7: Call
    "RETURN",       # 8: Return, Yield, YieldFrom
    "EXCEPTION",    # 9: Try, ExceptHandler, Raise, With
    "OP",           # 10: BinOp, UnaryOp, BoolOp, Compare
    "OTHER",        # 11: Otros nodos estructurales relevantes
)

STATE_TO_ID: dict[str, int] = {name: idx for idx, name in enumerate(DEFAULT_AST_STATES)}
NUM_STATES: int = len(DEFAULT_AST_STATES)
PAD_TOKEN: int = -1
SPLITS: tuple[str, ...] = ("train", "val", "test")