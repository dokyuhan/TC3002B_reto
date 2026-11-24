"""Input rules shared by the training-data script and the app, so both treat code the same way."""

from __future__ import annotations

import re

# A sample that declares its own top-level type (the training data never kept these as they were).
TOP_LEVEL_TYPE = re.compile(
    r"^\s*(?:@\w+(?:\([^)]*\))?\s*)*(?:(?:public|private|protected|final|abstract|static)\s+)*"
    r"(?:class|interface|enum|record)\s"
)
COMMENTS = re.compile(r"//[^\n]*|/\*.*?\*/", re.DOTALL)


def declares_top_level_type(code: str) -> bool:
    """True when the code starts with its own class, interface, enum or record declaration."""
    return TOP_LEVEL_TYPE.match(code) is not None


def has_empty_body(code: str) -> bool:
    """True when the method body contains nothing but whitespace and comments (a stub)."""
    stripped = COMMENTS.sub("", code)
    start, end = stripped.find("{"), stripped.rfind("}")
    return start != -1 and end > start and not stripped[start + 1 : end].strip()
