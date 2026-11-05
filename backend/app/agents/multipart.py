"""Split a question with several parts into self-contained sub-questions.

Each part is then solved by the normal pipeline (its own plan, intent check, computation and
independent verification), so a multi-part question is never answered half-way.

Recognised (offline, deterministic):

* enumerations: "1. … 2. …", "(a) … (b) …", "a) … b) …", "(i) … (ii) …", "Part 1: …";
* clauses joined by "and" / "then" / ";" where each clause starts with a task verb:
  "Solve x^2 - 4 = 0 and find the determinant of [[1,2],[3,4]]";
* two task nouns sharing one object: "the derivative and the integral of x^2 sin(x)";
* several sentences that each give a task: "Solve 2x + 3 = 7. Then compute the mean of 2, 4, 6."

Shared context before the parts ("Let f(x) = x^2 sin(x).") is carried into every part:
a function definition is substituted ("Integrate f(x) from 0 to pi" -> "Integrate (x^2 sin(x))
from 0 to pi"), other context sentences are prepended. "it" in a later part refers to the
expression of the first part and is replaced, so each part reads as its own question.
"""

from __future__ import annotations

import re

from app.math_engine.latex import normalize_latex

MAX_PARTS = 5

_VERB = (
    r"(?:find|compute|calculate|evaluate|determine|solve|differentiate|integrate|simplify|factor(?:ise|ize)?|expand|"
    r"give|state|list|what\s+is|what\s+are|how\s+many|show|work\s+out)"
)
_TASK_NOUN = (
    r"(?:derivative|second\s+derivative|antiderivative|integral|limit|roots?|zeros?|solutions?|determinant|inverse|"
    r"eigenvalues|rank|transpose|rref|factori[sz]ation|expansion|simplification)"
)
_STAT_NOUNS = re.compile(r"\b(mean|median|mode|average|variance|standard deviation|std|range)\b", re.I)


def _labels(kind: str) -> list[str]:
    if kind == "num":
        return [str(i) for i in range(1, 10)]
    if kind == "alpha":
        return list("abcdefgh")
    return ["i", "ii", "iii", "iv", "v", "vi"]


def _enumerated(text: str) -> tuple[str, list[str]] | None:
    """(prefix, parts) for '1. … 2. …', '(a) … (b) …', 'a) … b) …', '(i) … (ii) …'."""
    marker = re.compile(
        r"(?:(?<=\s)|^)(?:part\s+)?(?:\((?P<p>[a-h]|[ivx]{1,4}|\d)\)|(?P<s>[a-h]|[ivx]{1,4}|\d)[.)])(?=\s|:)", re.I
    )
    candidates = [(m.start(), m.end(), (m.group("p") or m.group("s")).lower()) for m in marker.finditer(text)]
    for kind in ("num", "alpha", "roman"):
        labels = _labels(kind)
        chosen, want = [], 0
        for start, end, label in candidates:
            if want < len(labels) and label == labels[want]:
                # a decimal like "x = 2. 3" is not a marker: the label must follow a sentence end or the start
                before = text[:start].rstrip()
                if before and before[-1] not in ".:;?!)" and not before.endswith("\n") and want > 0:
                    if not re.search(r"\s{2,}$|\n$", text[:start]):
                        continue
                chosen.append((start, end))
                want += 1
        if len(chosen) >= 2:
            prefix = text[: chosen[0][0]].strip()
            parts = []
            for i, (_, end) in enumerate(chosen):
                stop = chosen[i + 1][0] if i + 1 < len(chosen) else len(text)
                parts.append(text[end:stop].strip(" :\n"))
            return prefix, [p for p in parts if p]
    return None


def _clauses(text: str) -> list[str] | None:
    """'Solve … and find …' / '…; then compute …' -> clauses that each start with a task verb."""
    pieces = re.split(
        rf"\s*(?:;\s*(?:and\s+|then\s+)?|,?\s+and\s+then\s+|,?\s+then\s+|,?\s+and\s+(?:also\s+)?)(?={_VERB}\b)",
        text,
        flags=re.I,
    )
    if len(pieces) < 2:
        return None
    if not all(re.match(rf"\s*(?:then\s+|also\s+|next\s+|finally\s+)?{_VERB}\b", p, re.I) for p in pieces):
        return None
    return [p.strip(" .") for p in pieces if p.strip(" .")]


def _shared_object(text: str) -> list[str] | None:
    """'Find the derivative and the integral of x^2' -> two questions about the same expression."""
    m = re.match(
        rf"\s*(?P<verb>{_VERB})?\s*(?:the\s+)?(?P<n1>{_TASK_NOUN})\s+and\s+(?:the\s+|its\s+)?(?P<n2>{_TASK_NOUN})"
        rf"\s+(?P<rest>(?:of|for)\s+.+)$",
        text,
        re.I,
    )
    if not m or m.group("n1").lower() == m.group("n2").lower():
        return None
    verb = (m.group("verb") or "Find").strip()
    return [f"{verb} the {m.group('n1')} {m.group('rest')}", f"{verb} the {m.group('n2')} {m.group('rest')}"]


def _sentences(text: str) -> tuple[str, list[str]] | None:
    """Several sentences with a task each; earlier sentences without a task are shared context."""
    sentences = [s.strip() for s in re.split(r"(?<=[.?!])\s+(?=[A-Z])", text) if s.strip()]
    if len(sentences) < 2:
        return None
    tasks = [s for s in sentences if re.match(rf"(?:then\s+|also\s+|next\s+|finally\s+|now\s+)?{_VERB}\b", s, re.I)]
    if len(tasks) < 2:
        return None
    first = sentences.index(tasks[0])
    if any(s not in tasks for s in sentences[first:]):
        return None  # context in between tasks: too ambiguous to split
    prefix = " ".join(sentences[:first])
    return prefix, [re.sub(r"^(?:then|also|next|finally|now)\s+", "", t, flags=re.I).rstrip(".") for t in tasks]


_DEF = re.compile(
    r"\b(?:let\s+)?([a-hk-z])\s*\(\s*([a-z])\s*\)\s*=\s*([^,;]+?)(?=\s*(?:[,;]|\.\s|\.$|$|\bbe\b|\band\b))", re.I
)


def _apply_context(prefix: str, part: str) -> str:
    """Make a part self-contained: substitute a function defined in the prefix, or prepend context."""
    m = _DEF.search(prefix)
    if m:
        name, var, expr = m.group(1), m.group(2), m.group(3).strip()
        rest = (prefix[: m.start()] + prefix[m.end() :]).strip(" .,;")
        rest = re.sub(r"^(?:let|suppose|consider|given)\b\s*", "", rest, flags=re.I).strip(" .,;")
        out = re.sub(rf"\b{name}\s*''\s*\(\s*{var}\s*\)", f"the second derivative of {expr}", part)
        out = re.sub(rf"\b{name}\s*'\s*\(\s*{var}\s*\)", f"the derivative of {expr}", out)
        out = re.sub(
            rf"\b{name}\s*\(\s*([^()]+?)\s*\)",
            lambda mm: f"({expr})" if mm.group(1) == var else f"({expr}) at {var} = {mm.group(1)}",
            out,
        )
        out = re.sub(rf"(?<![\w'])\b{name}\b(?![\w(])", f"({expr})", out)
        return f"{rest}. {out}" if rest and len(rest.split()) > 2 else out
    if prefix:
        return f"{prefix.rstrip('. ')}. {part}"
    return part


def _first_object(part: str) -> str | None:
    """The expression a part is about (for resolving 'it' in later parts)."""
    m = re.search(r"\bof\s+(.+?)(?:\s+(?:from|at|as|with respect|wrt|for|on)\b|$)", part)
    if m:
        return m.group(1).strip(" .")
    m = re.match(rf"\s*{_VERB}\s+(.+?)(?:\s+(?:from|at|as|with respect|wrt|for|on)\b|$)", part, re.I)
    if m and re.search(r"[\dx-z^]", m.group(1)):
        return m.group(1).strip(" .")
    return None


def split_parts(problem: str) -> list[str] | None:
    """Self-contained sub-questions, or None when the question is a single task."""
    text = normalize_latex(problem).strip()
    found = _enumerated(text)
    if found is None:
        sent = _sentences(text)
        if sent is not None:
            found = sent
    if found is None:
        clauses = _clauses(text)
        if clauses is None:
            shared = _shared_object(text)
            if shared is None:
                return None
            clauses = shared
        found = ("", clauses)
    prefix, parts = found
    # a part that is itself "X and Y" with two task verbs is split further
    expanded: list[str] = []
    for p in parts:
        sub = _clauses(p) or _shared_object(p)
        expanded.extend(sub if sub else [p])
    if len(expanded) < 2 or len(expanded) > MAX_PARTS:
        return None
    # statistics nouns asked together ("mean and standard deviation") are one task, not parts
    if len(expanded) == 2 and all(_STAT_NOUNS.search(p) for p in expanded) and not prefix and found[0] == "":
        same_data = re.findall(r"[\d.]+(?:\s*,\s*[\d.]+)+", text)
        if len(same_data) <= 1:
            return None
    out: list[str] = []
    obj: str | None = None
    for p in expanded:
        q = _apply_context(prefix, p)
        if obj and re.search(r"\b(?:it|this|that|the result)\b", q, re.I):
            q = re.sub(r"\b(?:it|this|that|the result)\b", f"({obj})", q, count=1, flags=re.I)
        if obj is None:
            obj = _first_object(q)
        out.append(q[:1].upper() + q[1:])
    return out


def numbers_grounded(parts: list[str], problem: str) -> bool:
    """A split proposed by the LLM may only use numbers that appear in the original question."""
    src = normalize_latex(problem)
    allowed = {float(n) for n in re.findall(r"\d+(?:\.\d+)?", src)}
    for p in parts:
        for n in re.findall(r"\d+(?:\.\d+)?", p):
            if float(n) not in allowed:
                return False
    return True
