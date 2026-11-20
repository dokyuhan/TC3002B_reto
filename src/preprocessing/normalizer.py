"""Shared AST state vocabulary and deterministic source code normalization."""

from __future__ import annotations

import re

from src.preprocessing.vocabulary import (
    DEFAULT_AST_STATES,
    NUM_STATES,
    PAD_TOKEN,
    STATE_TO_ID,
    SUPPORTED_LANGUAGES,
)

__all__ = [
    "DEFAULT_AST_STATES",
    "NUM_STATES",
    "PAD_TOKEN",
    "STATE_TO_ID",
    "SUPPORTED_LANGUAGES",
    "normalize_source",
]

_JAVA_HEADER_RE = re.compile(
    r"^[ \t]*(?:package[ \t]+|import[ \t]+(?:static[ \t]+)?)[\w.*]+[ \t]*;[ \t]*$",
    re.MULTILINE,
)

_JAVA_IO_CLASS_NAMES = (
    "FastScanner",
    "FastReader",
    "InputReader",
    "InputReader2",
    "FastIO",
    "FastInput",
    "FastOutput",
    "FastWriter",
    "OutputWriter",
    "MyScanner",
    "Reader",
    "Scanner",
    "Scan",
    "ScanReader",
    "FastScan",
    "Kattio",
    "StdIn",
    "StdOut",
)

_JAVA_IO_CLASS_HEADER_RE = re.compile(
    r"(?m)^[ \t]*(?:(?:public|private|protected|static|final)[ \t]+)*"
    r"class[ \t]+(?:" + "|".join(_JAVA_IO_CLASS_NAMES) + r")\b[^{;]*\{"
)

_JAVA_IO_DECL_LINE_RE = re.compile(
    r"^[ \t]*(?:(?:public|private|protected|static|final)[ \t]+)*"
    r"(?:BufferedReader|PrintWriter|StringTokenizer|InputStreamReader|"
    r"BufferedWriter|BufferedOutputStream|Scanner)[ \t]+\w+[ \t]*=[ \t]*new[ \t]+[^;\n]+;[ \t]*$",
    re.MULTILINE,
)

_PYTHON_TOP_IMPORT_RE = re.compile(
    r"^(?:from[ \t]+[\w.]+[ \t]+import[ \t]+\([^)]*\)|"
    r"(?:from[ \t]+[\w.]+[ \t]+)?import[ \t]+[^\n]+)[ \t]*$",
    re.MULTILINE,
)

_PYTHON_IO_BOILERPLATE_RE = re.compile(
    r"^(?:sys\.setrecursionlimit\([^)\n]*\)|"
    r"input[ \t]*=[ \t]*(?:lambda[ \t]*:[ \t]*)?sys\.stdin\.readline[^\n]*)[ \t]*$",
    re.MULTILINE,
)

_CPP_INCLUDE_RE = re.compile(
    r'^[ \t]*#[ \t]*include[ \t]*(?:<[^>]+>|"[^"]+")[ \t]*$',
    re.MULTILINE,
)

_CPP_USING_NS_RE = re.compile(
    r"^[ \t]*using[ \t]+namespace[ \t]+[\w:]+[ \t]*;[ \t]*$",
    re.MULTILINE,
)

_CPP_FAST_IO_RE = re.compile(
    r"^[ \t]*(?:(?:std::)?(?:ios_base|ios)::sync_with_stdio\([^;\n]*\)|"
    r"(?:std::)?(?:cin|cout)\.tie\([^;\n]*\))[ \t]*;[ \t]*$",
    re.MULTILINE,
)

_MULTI_BLANK_RE = re.compile(r"\n{3,}")


def _find_matching_brace(source: str, open_brace_idx: int) -> int:
    """Return index of matching '}' for '{' at open_brace_idx, or -1 if unmatched."""
    depth = 0
    i = open_brace_idx
    n = len(source)

    while i < n:
        ch = source[i]
        if ch == "/" and i + 1 < n:
            nxt = source[i + 1]
            if nxt == "/":
                newline = source.find("\n", i + 2)
                i = n if newline == -1 else newline + 1
                continue
            if nxt == "*":
                end = source.find("*/", i + 2)
                i = n if end == -1 else end + 2
                continue
        elif ch in ('"', "'"):
            quote = ch
            i += 1
            while i < n:
                if source[i] == "\\":
                    i += 2
                    continue
                if source[i] == quote:
                    i += 1
                    break
                i += 1
            continue
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1

    return -1


def _strip_java_io_classes(source: str) -> str:
    """Remove common competitive programming I/O helper classes using brace balancing."""
    spans: list[tuple[int, int]] = []
    pos = 0
    while True:
        match = _JAVA_IO_CLASS_HEADER_RE.search(source, pos)
        if match is None:
            break
        open_brace = match.end() - 1
        close_brace = _find_matching_brace(source, open_brace)
        if close_brace == -1:
            pos = match.end()
            continue
        spans.append((match.start(), close_brace + 1))
        pos = close_brace + 1

    if not spans:
        return source

    chunks: list[str] = []
    last = 0
    for start, end in spans:
        chunks.append(source[last:start])
        last = end
    chunks.append(source[last:])
    candidate = "".join(chunks)
    # Guard against wiping a file whose only class uses a generic helper name.
    return candidate if candidate.strip() else source


def normalize_source(
    source_code: str,
    language: str,
    strip_boilerplate: bool = True,
) -> str:
    """Normalize whitespace/line endings and optionally strip language boilerplate."""
    lang = language.strip().lower()
    if lang not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported language: {language!r}. Expected one of {sorted(SUPPORTED_LANGUAGES)}."
        )

    text = source_code.lstrip("\ufeff")
    text = text.replace("\xa0", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    if not strip_boilerplate:
        return text

    if lang == "java":
        text = _JAVA_HEADER_RE.sub("", text)
        text = _strip_java_io_classes(text)
        text = _JAVA_IO_DECL_LINE_RE.sub("", text)
    elif lang == "python":
        text = _PYTHON_TOP_IMPORT_RE.sub("", text)
        text = _PYTHON_IO_BOILERPLATE_RE.sub("", text)
    elif lang == "cpp":
        text = _CPP_INCLUDE_RE.sub("", text)
        text = _CPP_USING_NS_RE.sub("", text)
        text = _CPP_FAST_IO_RE.sub("", text)

    text = _MULTI_BLANK_RE.sub("\n\n", text)
    return text.strip()
