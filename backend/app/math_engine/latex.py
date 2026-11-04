"""Turn LaTeX-flavoured problem text (as pasted from many sources) into plain math text.

``Let \\(A\\) be a \\(4 \\times 4\\) matrix with \\(\\text{Tr}(A) = 2\\)`` becomes
``Let A be a 4 × 4 matrix with Tr(A) = 2``. Plain text passes through unchanged.
Nothing is evaluated: this is string rewriting only, and the result still goes
through the safe parser and the intent checks.
"""

from __future__ import annotations

import re

_COMMANDS = {
    r"\times": "×",
    r"\cdot": "*",
    r"\ast": "*",
    r"\div": "/",
    r"\pm": "±",
    r"\le": "<=",
    r"\leq": "<=",
    r"\ge": ">=",
    r"\geq": ">=",
    r"\neq": "!=",
    r"\ne": "!=",
    r"\to": "->",
    r"\rightarrow": "->",
    r"\infty": "oo",
    r"\pi": "pi",
    r"\lambda": "lambda",
    r"\mu": "mu",
    r"\sigma": "sigma",
    r"\alpha": "alpha",
    r"\beta": "beta",
    r"\theta": "theta",
    r"\det": "det",
    r"\tr": "tr",
    r"\ln": "ln",
    r"\log": "log",
    r"\exp": "exp",
    r"\sin": "sin",
    r"\cos": "cos",
    r"\tan": "tan",
    r"\lim": "lim",
    r"\int": "∫",
    r"\sum": "Σ",
    r"\ldots": "...",
    r"\dots": "...",
    r"\cdots": "...",
    r"\quad": " ",
    r"\qquad": " ",
}
_COMMAND_RE = re.compile(
    r"\\(" + "|".join(sorted((c[1:] for c in _COMMANDS), key=len, reverse=True)) + r")(?![A-Za-z])"
)


def _braced(text: str, start: int) -> tuple[str, int] | None:
    """The content of the {...} group starting at ``text[start] == '{'`` and the index after it."""
    if start >= len(text) or text[start] != "{":
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : i], i + 1
    return None


def _replace_two_arg(text: str, command: str, fmt: str) -> str:
    out, i = [], 0
    while True:
        j = text.find(command, i)
        if j < 0:
            out.append(text[i:])
            return "".join(out)
        first = _braced(text, j + len(command))
        second = _braced(text, first[1]) if first else None
        if not (first and second):
            out.append(text[i : j + len(command)])
            i = j + len(command)
            continue
        out.append(text[i:j] + fmt.format(first[0], second[0]))
        i = second[1]


def _replace_one_arg(text: str, command: str, fmt: str) -> str:
    out, i = [], 0
    while True:
        j = text.find(command, i)
        if j < 0:
            out.append(text[i:])
            return "".join(out)
        arg = _braced(text, j + len(command))
        if not arg:
            out.append(text[i : j + len(command)])
            i = j + len(command)
            continue
        out.append(text[i:j] + fmt.format(arg[0]))
        i = arg[1]


def looks_like_latex(text: str) -> bool:
    return "\\" in text or "$" in text


def normalize_latex(text: str) -> str:
    if not looks_like_latex(text):
        return text
    s = text
    # math delimiters
    s = re.sub(r"\\[()\[\]]", " ", s)
    s = s.replace("$$", " ").replace("$", " ")
    s = re.sub(r"\\(?:left|right|big|Big|bigg|Bigg)\b\s*([()\[\]|.])?", lambda m: m.group(1) or "", s)
    s = s.replace(r"\{", "(").replace(r"\}", ")")
    # \text{Tr}, \mathrm{tr}, \operatorname{det} ... -> the word itself
    s = re.sub(r"\\(?:text|textbf|textit|mathrm|mathbf|mathit|operatorname|mathbb|mathcal)\s*\{([^{}]*)\}", r"\1", s)
    for _ in range(3):  # nested fractions / roots
        s = _replace_two_arg(s, r"\frac", "(({0})/({1}))")
        s = _replace_two_arg(s, r"\dfrac", "(({0})/({1}))")
        s = _replace_two_arg(s, r"\tfrac", "(({0})/({1}))")
        s = _replace_one_arg(s, r"\sqrt", "sqrt({0})")
    s = _COMMAND_RE.sub(lambda m: f" {_COMMANDS['\\' + m.group(1)]} ", s)
    s = re.sub(r"\\[,;:! ]", " ", s)  # spacing commands
    # superscripts / subscripts: x^{10} -> x^(10), I_{4} -> I_4, c_{2} -> c_2
    s = re.sub(r"\^\s*\{([^{}]*)\}", r"^(\1)", s)
    s = re.sub(r"_\s*\{\s*(\w+)\s*\}", r"_\1", s)
    s = re.sub(r"\\([A-Za-z]+)", r"\1", s)  # any other command: keep its name
    s = s.replace("{", "(").replace("}", ")")
    s = re.sub(r"\^\(\s*(-?\w+)\s*\)", r"^\1", s)  # x^(5) -> x^5, A^(-1) -> A^-1
    s = re.sub(r"(\w)\s+([\^_])", r"\1\2", s)  # "lambda ^4" -> "lambda^4", "lambda _1" -> "lambda_1"
    s = re.sub(r"\(\s+", "(", s)
    s = re.sub(r"\s+([).,;:?])", r"\1", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r" *\n *", "\n", s)
    return s.strip()
