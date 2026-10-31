"""Safe parsing of mathematical expressions.

SymPy's ``parse_expr`` ultimately calls ``eval``. To make sure that no
LLM-generated or user-supplied text can ever execute arbitrary Python, every
expression is first checked against a strict lexical allowlist:

* only a small set of characters is permitted (no quotes, underscores,
  backslashes, semicolons, ``@``, ``!``-free etc.),
* dunder access, attribute access and keywords are rejected,
* every identifier must be a known math function/constant or a short symbol.

Only after that check is the string handed to SymPy, with an explicit and
minimal ``global_dict`` / ``local_dict`` so no builtins are reachable.
"""

from __future__ import annotations

import re
from functools import lru_cache

import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    rationalize,
    standard_transformations,
)

MAX_EXPRESSION_LENGTH = 500


class ExpressionError(ValueError):
    """Raised when an expression is invalid, unsafe or cannot be parsed."""


# Functions and constants that may appear in an expression.
ALLOWED_FUNCTIONS: dict[str, object] = {
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "cot": sp.cot,
    "sec": sp.sec,
    "csc": sp.csc,
    "asin": sp.asin,
    "acos": sp.acos,
    "atan": sp.atan,
    "arcsin": sp.asin,
    "arccos": sp.acos,
    "arctan": sp.atan,
    "sinh": sp.sinh,
    "cosh": sp.cosh,
    "tanh": sp.tanh,
    "exp": sp.exp,
    "log": sp.log,
    "ln": sp.log,
    "sqrt": sp.sqrt,
    "cbrt": sp.cbrt,
    "Abs": sp.Abs,
    "abs": sp.Abs,
    "factorial": sp.factorial,
    "binomial": sp.binomial,
    "floor": sp.floor,
    "ceiling": sp.ceiling,
    "gamma": sp.gamma,
}

ALLOWED_CONSTANTS: dict[str, object] = {
    "pi": sp.pi,
    "e": sp.E,
    "E": sp.E,
    "I": sp.I,
    "i": sp.I,
    "oo": sp.oo,
    "inf": sp.oo,
    "infinity": sp.oo,
}

GREEK_SYMBOLS = {
    "alpha",
    "beta",
    "delta",
    "theta",
    "mu",
    "sigma",
    "omega",
    "phi",
    "tau",
    "rho",
}

_ALLOWED_CHARS = re.compile(r"^[0-9A-Za-z\s+\-*/^().,=<>]*$")
_IDENTIFIER = re.compile(r"[A-Za-z][A-Za-z0-9]*")
_FORBIDDEN_WORDS = {
    "import",
    "lambda",
    "exec",
    "eval",
    "open",
    "os",
    "sys",
    "class",
    "def",
    "for",
    "while",
    "global",
    "with",
    "yield",
    "subprocess",
    "compile",
    "getattr",
    "setattr",
    "globals",
    "locals",
    "vars",
    "dir",
    "input",
    "print",
    "Symbol",
    "Function",
}


def _symbol_like(name: str) -> bool:
    """Short identifiers (``x``, ``y1``, ``ab``) or greek letter names are symbols."""
    return bool(re.fullmatch(r"[A-Za-z][0-9]?", name)) or name in GREEK_SYMBOLS


def _known(name: str) -> bool:
    return name in ALLOWED_FUNCTIONS or name in ALLOWED_CONSTANTS or _symbol_like(name)


def _check_identifier(match: re.Match[str]) -> str:
    ident = match.group(0)
    if ident in _FORBIDDEN_WORDS:
        raise ExpressionError(f"Forbidden identifier: {ident!r}")
    if _known(ident):
        return ident
    # "sinx" -> "sin x"
    for fn in sorted(ALLOWED_FUNCTIONS, key=len, reverse=True):
        rest = ident[len(fn) :]
        if ident.startswith(fn) and rest and _symbol_like(rest):
            return f"{fn} {rest}"
    # "xy" -> "x*y" (implicit product of single-letter variables)
    if ident.isalpha() and len(ident) <= 3:
        return "*".join(ident)
    raise ExpressionError(f"Unknown identifier {ident!r}. Use standard math functions.")


def validate_expression_text(text: str) -> str:
    """Return a normalised expression string or raise :class:`ExpressionError`."""
    if not isinstance(text, str):
        raise ExpressionError("Expression must be a string.")
    cleaned = text.strip()
    if not cleaned:
        raise ExpressionError("Expression is empty.")
    if len(cleaned) > MAX_EXPRESSION_LENGTH:
        raise ExpressionError(f"Expression exceeds {MAX_EXPRESSION_LENGTH} characters.")
    # Common unicode niceties.
    cleaned = (
        cleaned.replace("×", "*")
        .replace("·", "*")
        .replace("÷", "/")
        .replace("−", "-")
        .replace("²", "^2")
        .replace("³", "^3")
        .replace("π", "pi")
        .replace("√", "sqrt")
    )
    if not _ALLOWED_CHARS.fullmatch(cleaned):
        bad = sorted({c for c in cleaned if not _ALLOWED_CHARS.fullmatch(c)})
        raise ExpressionError(f"Expression contains disallowed characters: {''.join(bad)!r}")
    if re.search(r"[A-Za-z0-9)]\s*\.\s*[A-Za-z]", cleaned):
        raise ExpressionError("Attribute access is not allowed.")
    cleaned = _IDENTIFIER.sub(_check_identifier, cleaned)
    if cleaned.count("(") != cleaned.count(")"):
        raise ExpressionError("Unbalanced parentheses.")
    return cleaned


# ``rationalize`` turns decimal literals into exact rationals (0.1 -> 1/10) so that
# all downstream arithmetic is exact instead of binary floating point.
_TRANSFORMS = standard_transformations + (convert_xor, implicit_multiplication_application, rationalize)


@lru_cache(maxsize=1)
def _global_dict() -> dict[str, object]:
    # Only what SymPy's generated code needs: no Python builtins.
    return {
        "__builtins__": {},
        "Integer": sp.Integer,
        "Float": sp.Float,
        "Rational": sp.Rational,
        "Symbol": sp.Symbol,
        "Function": sp.Function,
    }


def _local_dict(text: str) -> dict[str, object]:
    local: dict[str, object] = {**ALLOWED_FUNCTIONS, **ALLOWED_CONSTANTS}
    for ident in _IDENTIFIER.findall(text):
        if ident not in local:
            local[ident] = sp.Symbol(ident.rstrip("_"), real=None)
    return local


def parse_expression(text: str) -> sp.Expr:
    """Safely parse a single SymPy expression (no ``=`` allowed)."""
    cleaned = validate_expression_text(text)
    if "=" in cleaned or "<" in cleaned or ">" in cleaned:
        raise ExpressionError("Expected an expression, not an equation or inequality.")
    return _parse(cleaned)


def _parse(cleaned: str) -> sp.Expr:
    try:
        result = parse_expr(
            cleaned,
            local_dict=_local_dict(cleaned),
            global_dict=dict(_global_dict()),
            transformations=_TRANSFORMS,
            evaluate=True,
        )
    except Exception as exc:  # SymPy raises many different exception types
        raise ExpressionError(f"Could not parse expression {cleaned!r}.") from exc
    if not isinstance(result, sp.Basic):
        raise ExpressionError("Parsed value is not a mathematical expression.")
    return result


def parse_equation(text: str) -> sp.Eq | sp.Expr:
    """Parse ``lhs = rhs`` into ``Eq``; a bare expression means ``expr = 0``."""
    cleaned = validate_expression_text(text)
    cleaned = cleaned.replace("==", "=")
    if "<" in cleaned or ">" in cleaned:
        raise ExpressionError("Inequalities are not supported yet.")
    parts = cleaned.split("=")
    if len(parts) == 1:
        return sp.Eq(_parse(parts[0]), 0, evaluate=False)
    if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
        raise ExpressionError("An equation must have exactly one '=' with both sides present.")
    return sp.Eq(_parse(parts[0]), _parse(parts[1]), evaluate=False)


def parse_symbol(name: str | None, default: str = "x") -> sp.Symbol:
    name = (name or default).strip()
    if not _symbol_like(name):
        raise ExpressionError(f"Invalid variable name {name!r}.")
    return sp.Symbol(name)


def parse_number_list(values: list[float | int | str]) -> list[float]:
    out: list[float] = []
    for v in values:
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out.append(float(v))
            continue
        expr = parse_expression(str(v))
        if not expr.is_number:
            raise ExpressionError(f"{v!r} is not a number.")
        out.append(float(sp.N(expr)))
    return out


def parse_matrix(rows: list[list[float | int | str]]) -> sp.Matrix:
    if not rows or not all(isinstance(r, list) and r for r in rows):
        raise ExpressionError("Matrix must be a non-empty list of non-empty rows.")
    width = len(rows[0])
    if any(len(r) != width for r in rows):
        raise ExpressionError("All matrix rows must have the same length.")
    if len(rows) > 10 or width > 10:
        raise ExpressionError("Matrices larger than 10x10 are not supported.")
    parsed = []
    for row in rows:
        parsed_row = []
        for v in row:
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                parsed_row.append(sp.nsimplify(v, rational=True))
            else:
                parsed_row.append(parse_expression(str(v)))
        parsed.append(parsed_row)
    return sp.Matrix(parsed)
