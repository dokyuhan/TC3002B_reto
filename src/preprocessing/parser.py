"""Tree-sitter 0.26+ AST parsing and 12-state sequence extraction."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import numpy as np
from tree_sitter import Language, Parser
import tree_sitter_cpp
import tree_sitter_java
import tree_sitter_python

from src.preprocessing.normalizer import (
    STATE_TO_ID,
    SUPPORTED_LANGUAGES,
    normalize_source,
)

COMMENT_NODES: frozenset[str] = frozenset(
    {
        "comment",
        "line_comment",
        "block_comment",
    }
)

SKIP_WRAPPER_NODES: frozenset[str] = frozenset(
    {
        # Root & body containers
        "program",
        "module",
        "translation_unit",
        "block",
        "compound_statement",
        "class_body",
        "interface_body",
        "enum_body",
        "constructor_body",
        "switch_block",
        "field_declaration_list",
        "declaration_list",
        "static_initializer",
        # Statement / expression / argument / parameter wrappers
        "expression_statement",
        "argument_list",
        "parenthesized_expression",
        "modifiers",
        "dimensions",
        "dimensions_expr",
        "formal_parameters",
        "parameters",
        "parameter_list",
        "condition_clause",
        "type_arguments",
        "type_parameters",
        "template_argument_list",
        "template_parameter_list",
        "subscript_argument_list",
        "access_specifier",
        "type_qualifier",
        "storage_class_specifier",
        "superclass",
        "super_interfaces",
        "type_list",
        "base_class_clause",
        "annotation",
        "marker_annotation",
        "decorator",
        "asterisk",
        "string_fragment",
        "string_content",
        "string_start",
        "string_end",
        "escape_sequence",
        "pattern_list",
        "expression_list",
        "as_pattern",
        "as_pattern_target",
        "type",
        "type_parameter",
        "dotted_name",
        "system_lib_string",
        "empty_statement",
        "labeled_statement",
    }
)

NODE_TYPE_TO_STATE: dict[str, str] = {
    # IMPORT
    "import_declaration": "IMPORT",
    "package_declaration": "IMPORT",
    "import_statement": "IMPORT",
    "import_from_statement": "IMPORT",
    "future_import_statement": "IMPORT",
    "preproc_include": "IMPORT",
    "using_declaration": "IMPORT",
    "namespace_alias_definition": "IMPORT",
    # FUNCTION
    "method_declaration": "FUNCTION",
    "constructor_declaration": "FUNCTION",
    "compact_constructor_declaration": "FUNCTION",
    "lambda_expression": "FUNCTION",
    "method_reference": "FUNCTION",
    "function_definition": "FUNCTION",
    "function_declarator": "FUNCTION",
    "lambda": "FUNCTION",
    "lambda_expression_cpp": "FUNCTION",
    "formal_parameter": "FUNCTION",
    "spread_parameter": "FUNCTION",
    "parameter_declaration": "FUNCTION",
    "optional_parameter_declaration": "FUNCTION",
    "typed_parameter": "FUNCTION",
    "default_parameter": "FUNCTION",
    "typed_default_parameter": "FUNCTION",
    "list_splat_pattern": "FUNCTION",
    "dictionary_splat_pattern": "FUNCTION",
    # CLASS (classes, structs, interfaces, namespaces, and type nodes)
    "class_declaration": "CLASS",
    "interface_declaration": "CLASS",
    "enum_declaration": "CLASS",
    "enum_constant": "CLASS",
    "record_declaration": "CLASS",
    "class_definition": "CLASS",
    "class_specifier": "CLASS",
    "struct_specifier": "CLASS",
    "enum_specifier": "CLASS",
    "namespace_definition": "CLASS",
    "template_declaration": "CLASS",
    "type_identifier": "CLASS",
    "scoped_type_identifier": "CLASS",
    "integral_type": "CLASS",
    "floating_point_type": "CLASS",
    "boolean_type": "CLASS",
    "void_type": "CLASS",
    "array_type": "CLASS",
    "generic_type": "CLASS",
    "wildcard": "CLASS",
    "primitive_type": "CLASS",
    "sized_type_specifier": "CLASS",
    "template_type": "CLASS",
    "type_descriptor": "CLASS",
    "placeholder_type_specifier": "CLASS",
    "qualified_identifier": "CLASS",
    # ASSIGN (assignments, variable/field declarations, mutations)
    "assignment_expression": "ASSIGN",
    "assignment": "ASSIGN",
    "augmented_assignment": "ASSIGN",
    "named_expression": "ASSIGN",
    "local_variable_declaration": "ASSIGN",
    "field_declaration": "ASSIGN",
    "variable_declarator": "ASSIGN",
    "declaration": "ASSIGN",
    "init_declarator": "ASSIGN",
    "pointer_declarator": "ASSIGN",
    "reference_declarator": "ASSIGN",
    "array_declarator": "ASSIGN",
    "update_expression": "ASSIGN",
    "global_statement": "ASSIGN",
    "nonlocal_statement": "ASSIGN",
    "delete_statement": "ASSIGN",
    "delete_expression": "ASSIGN",
    # IF (conditionals and branching)
    "if_statement": "IF",
    "elif_clause": "IF",
    "else_clause": "IF",
    "switch_statement": "IF",
    "switch_expression": "IF",
    "switch_block_statement_group": "IF",
    "switch_rule": "IF",
    "switch_label": "IF",
    "case_statement": "IF",
    "match_statement": "IF",
    "case_clause": "IF",
    "ternary_expression": "IF",
    "conditional_expression": "IF",
    # FOR (counted/iterator loops and comprehensions)
    "for_statement": "FOR",
    "enhanced_for_statement": "FOR",
    "for_range_loop": "FOR",
    "for_in_clause": "FOR",
    "list_comprehension": "FOR",
    "dictionary_comprehension": "FOR",
    "set_comprehension": "FOR",
    "generator_expression": "FOR",
    # WHILE
    "while_statement": "WHILE",
    "do_statement": "WHILE",
    # CALL (calls and object/array allocations)
    "method_invocation": "CALL",
    "call": "CALL",
    "call_expression": "CALL",
    "object_creation_expression": "CALL",
    "array_creation_expression": "CALL",
    "new_expression": "CALL",
    "explicit_constructor_invocation": "CALL",
    "super": "CALL",
    # RETURN (control transfer: return, break, continue, yield)
    "return_statement": "RETURN",
    "break_statement": "RETURN",
    "continue_statement": "RETURN",
    "yield": "RETURN",
    "yield_statement": "RETURN",
    "pass_statement": "RETURN",
    "goto_statement": "RETURN",
    # EXCEPTION (error handling & assertions)
    "try_statement": "EXCEPTION",
    "try_with_resources_statement": "EXCEPTION",
    "catch_clause": "EXCEPTION",
    "catch_formal_parameter": "EXCEPTION",
    "catch_type": "EXCEPTION",
    "except_clause": "EXCEPTION",
    "finally_clause": "EXCEPTION",
    "throw_statement": "EXCEPTION",
    "raise_statement": "EXCEPTION",
    "throws": "EXCEPTION",
    "assert_statement": "EXCEPTION",
    "with_statement": "EXCEPTION",
    "with_clause": "EXCEPTION",
    "with_item": "EXCEPTION",
    "resource_specification": "EXCEPTION",
    "resource": "EXCEPTION",
    # OP (operators, expressions, accesses, identifiers, literals, collections)
    "binary_expression": "OP",
    "unary_expression": "OP",
    "binary_operator": "OP",
    "unary_operator": "OP",
    "boolean_operator": "OP",
    "comparison_operator": "OP",
    "not_operator": "OP",
    "cast_expression": "OP",
    "instanceof_expression": "OP",
    "sizeof_expression": "OP",
    "pointer_expression": "OP",
    "array_access": "OP",
    "subscript": "OP",
    "subscript_expression": "OP",
    "slice": "OP",
    "field_access": "OP",
    "field_expression": "OP",
    "attribute": "OP",
    "identifier": "OP",
    "field_identifier": "OP",
    "namespace_identifier": "OP",
    "scoped_identifier": "OP",
    "this": "OP",
    "decimal_integer_literal": "OP",
    "hex_integer_literal": "OP",
    "octal_integer_literal": "OP",
    "binary_integer_literal": "OP",
    "decimal_floating_point_literal": "OP",
    "hex_floating_point_literal": "OP",
    "integer": "OP",
    "float": "OP",
    "number_literal": "OP",
    "string": "OP",
    "string_literal": "OP",
    "concatenated_string": "OP",
    "character_literal": "OP",
    "char_literal": "OP",
    "true": "OP",
    "false": "OP",
    "null_literal": "OP",
    "none": "OP",
    "nullptr": "OP",
    "list": "OP",
    "tuple": "OP",
    "dictionary": "OP",
    "pair": "OP",
    "set": "OP",
    "array_initializer": "OP",
    "initializer_list": "OP",
    "keyword_argument": "OP",
    "comma_expression": "OP",
}

_OTHER_STATE_ID: int = STATE_TO_ID["OTHER"]


@dataclass(frozen=True)
class ParseResult:
    """Result of parsing source code into a 1D AST state sequence."""

    states: np.ndarray
    has_error: bool
    error_count: int
    token_count: int
    other_ratio: float


@lru_cache(maxsize=4)
def _get_parser(language: str) -> Parser:
    """Build and cache a Tree-sitter 0.26+ Parser for a supported language."""
    if language == "java":
        return Parser(Language(tree_sitter_java.language()))
    if language == "python":
        return Parser(Language(tree_sitter_python.language()))
    if language == "cpp":
        return Parser(Language(tree_sitter_cpp.language()))
    raise ValueError(
        f"Unsupported language: {language!r}. Expected one of {sorted(SUPPORTED_LANGUAGES)}."
    )


def parse_to_state_sequence(
    source_code: str,
    language: str,
    strip_boilerplate: bool = True,
) -> ParseResult:
    """Normalize and parse source code into a 1D int32 AST state ID sequence."""
    lang = language.strip().lower()
    if lang not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported language: {language!r}. Expected one of {sorted(SUPPORTED_LANGUAGES)}."
        )

    normalized = normalize_source(source_code, language=lang, strip_boilerplate=strip_boilerplate)
    parser = _get_parser(lang)
    tree = parser.parse(normalized.encode("utf-8"))
    root = tree.root_node

    state_ids: list[int] = []
    error_count = 0

    stack = [root]
    while stack:
        node = stack.pop()
        node_type = node.type

        if node_type == "ERROR" or node.is_missing:
            error_count += 1
            if node.children:
                stack.extend(reversed(node.children))
            continue

        if not node.is_named:
            continue

        if node_type in COMMENT_NODES:
            continue

        if node_type not in SKIP_WRAPPER_NODES:
            state_name = NODE_TYPE_TO_STATE.get(node_type, "OTHER")
            state_ids.append(STATE_TO_ID[state_name])

        if node.children:
            stack.extend(reversed(node.children))

    states = np.asarray(state_ids, dtype=np.int32)
    token_count = int(states.size)
    other_ratio = float(np.mean(states == _OTHER_STATE_ID)) if token_count > 0 else 0.0
    has_error = bool(root.has_error or error_count > 0)

    return ParseResult(
        states=states,
        has_error=has_error,
        error_count=error_count,
        token_count=token_count,
        other_ratio=other_ratio,
    )
