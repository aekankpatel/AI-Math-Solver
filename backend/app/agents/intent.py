"""Intent verification: does the planned tool call answer the question that was asked?

The mathematical verifier proves that a tool computed its result correctly. It
cannot tell whether the tool and arguments were the right reading of the user's
question, for example the determinant computed for "determine the rank", a
default standard deviation silently invented, or a "positive solution"
constraint ignored. This module closes that gap with checks that work on the
original text, independently of either planner:

1. **Structured intent extraction** (``extract_intent``). Cue-based, deterministic
   extraction of the task, variable, domain, derivative order, evaluation point,
   integration bounds, limit point/direction, probability relation, distribution,
   counting kind, matrices and every number mentioned in the question.
2. **Parameter validation**. Each argument value must be grounded in the text
   (no invented numbers), and numbers in the text should be used.
3. **Ambiguity detection**. Missing information (no upper limit, no standard
   deviation, several unknowns but one equation, ...) leads to a clarification
   request instead of a guess.
4. **Alignment checks**. Task/operation, slots and expressions in the plan
   are compared with what the text says (expressions numerically, at random points).

Result: an ``IntentVerification`` with status ``confirmed`` / ``assumed`` /
``ambiguous`` / ``mismatch``. Some mismatches carry a safe *fix* (a slot the
question states explicitly but the plan omitted), which the workflow may apply
once before re-checking.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

import sympy as sp

from app.math_engine.parser import _ENGLISH_WORDS, ExpressionError, parse_equation, parse_expression
from app.schemas import IntentCheck, IntentStatus, IntentVerification, ToolCall

# --------------------------------------------------------------------------- extraction


@dataclass
class Intent:
    task: str | None = None
    task_conflict: list[str] = field(default_factory=list)
    variable: str | None = None
    domain: str | None = None
    order: int | None = None
    point: str | None = None
    lower: str | None = None
    upper: str | None = None
    partial_bounds: bool = False
    limit_point: str | None = None
    direction: str | None = None
    relation: tuple[str, str, str | None] | None = None  # (query, threshold, threshold2) with strict marker in query
    distribution: str | None = None
    counting: str | None = None
    matrices: list[list[list[Any]]] = field(default_factory=list)
    vectors: list[list[Any]] = field(default_factory=list)
    numbers: list[float] = field(default_factory=list)
    allowed_numbers: set[float] = field(default_factory=set)
    mentions: set[str] = field(default_factory=set)


_NUM = r"-?\d+(?:\.\d+)?"
_INF_WORDS = r"(?:infinity|∞|oo|inf)"
_WORD_NUMBERS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "half": 0.5,
    "twice": 2,
    "double": 2,
    "squared": 2,
    "cubed": 3,
    "dozen": 12,
    "pair": 2,
    "quarter": 0.25,
    "third": 3,
}

_TASK_CUES: list[tuple[str, str]] = [
    (
        "derivative",
        r"(?<!anti)derivative|differentiat|\bd\s*/\s*d[a-z]\b|rate of change|\bslope\b|\bf\s*'+|how (?:fast|quickly|rapidly)",
    ),
    ("integral", r"integra|antiderivative|area under|∫"),
    ("limit", r"\blim\b|\blim_|\blimit|\bapproach|\btends? to\b|\bgoes to\b"),
    ("factor", r"\bfactor(?:ise|ize|ization|isation)?\b"),
    ("expand", r"\bexpand|\bmultiply out\b"),
    ("simplify", r"\bsimplif"),
]
_MATRIX_OPS: list[tuple[str, str]] = [
    ("eigenvalues", r"eigen"),
    ("inverse", r"\binvers|\binvert"),
    ("rank", r"\brank\b"),
    ("transpose", r"transpos"),
    ("rref", r"\brref\b|row[- ]echelon|row[- ]reduc"),
    ("determinant", r"\bdeterminant\b|\bdet\b|\bdet\s*\("),
    ("multiply", r"\bmultiply|\bproduct\b|\btimes\b"),
    ("solve", r"\bsolve|\bsystem\b|\bax\s*=\s*b\b"),
]
_DISTRIBUTIONS: list[tuple[str, str]] = [
    ("binomial", r"binomial|\btrials?\b|\bcoin|\bflip|\btoss|\bindependent(?:ly)?\b|free throws|\bshots?\b"),
    ("poisson", r"poisson|arrive|per (?:minute|hour|day|second|week)|average rate"),
    ("normal", r"\bnormal|gaussian|bell[- ]curve|\bz-score|\bstandard normal"),
    ("uniform", r"uniform"),
    ("exponential", r"exponential"),
]


def _normalise(text: str) -> str:
    return (
        text.replace("’", "'")
        .replace("≤", "<=")
        .replace("≥", ">=")
        .replace("→", "->")
        .replace("−", "-")
        .replace("×", "*")
    )


def _numbers_in(text: str) -> list[float]:
    out = [abs(float(m)) for m in re.findall(r"(?<![A-Za-z_])\d+(?:\.\d+)?", text)]
    out += [float(m) / 100 for m in re.findall(r"(\d+(?:\.\d+)?)\s*%", text)]
    out += [float(Fraction(int(a), int(b))) for a, b in re.findall(r"(\d+)\s*/\s*(\d+)", text) if int(b)]
    return out


def _value(token: str | None) -> str | None:
    if token is None:
        return None
    token = token.strip().strip(".,;?:)(")
    if re.fullmatch(_INF_WORDS, token):
        return "oo"
    if re.fullmatch(r"-" + _INF_WORDS, token):
        return "-oo"
    return token or None


def _literal_lists(text: str) -> tuple[list, list]:
    import ast

    matrices, vectors = [], []
    for m in re.finditer(r"\[\s*\[.*?\]\s*\]", text):
        try:
            matrices.append(ast.literal_eval(m.group(0)))
        except (ValueError, SyntaxError):
            pass
    rest = re.sub(r"\[\s*\[.*?\]\s*\]", " ", text)
    for m in re.finditer(r"(?<![\w\]])\[([^\[\]]+)\]", rest):
        before = rest[max(0, m.start() - 20) : m.start()].lower()
        if re.search(r"\b(over|on|in)\s+(?:the\s+)?(?:interval\s*)?$|\b(over|on|in)\s*$", before):
            continue  # an interval, not a vector
        try:
            v = ast.literal_eval(m.group(0))
            if isinstance(v, list):
                vectors.append(v)
        except (ValueError, SyntaxError):
            pass
    return matrices, vectors


def extract_intent(problem: str) -> Intent:
    text = _normalise(problem)
    low = text.lower()
    it = Intent()
    it.matrices, it.vectors = _literal_lists(text)

    # ---- task
    if it.matrices:
        it.task = next((op for op, pat in _MATRIX_OPS if re.search(pat, low)), None)
        if it.task and re.search(r"\bdetermin(e|ing)\b", low) and it.task == "determinant":
            it.task = next((op for op, pat in _MATRIX_OPS if op != "determinant" and re.search(pat, low)), None)
        it.task = f"matrix:{it.task}" if it.task else "matrix"
    elif re.search(
        r"\b[pc]\s*\(\s*\d+\s*,\s*\d+\s*\)|permutations?\s*\(|combinations?\s*\(|\d\s*!|\bfactorial\b", low
    ) or re.search(
        r"how many (?:different |distinct |possible )?(?:ways|hands|arrangements|orders|committees|teams|groups|outcomes)"
        r"|in how many ways"
        r"|\bchoose\b|\bcombinations?\b|\bpermutations?\b|\barrange|\bline up\b",
        low,
    ):
        it.task = "counting"
    elif re.search(
        r"probabilit|\bp\s*\(\s*[a-z]|\bp\s*\([^)]*[<>=]|\bchance\b|proportion|what fraction|fraction of|percentage|likelihood",
        low,
    ):
        it.task = "probability"
    else:
        hits = [name for name, pat in _TASK_CUES if re.search(pat, low)]
        calc = [h for h in hits if h in ("derivative", "integral", "limit")]
        if len(calc) > 1:
            it.task_conflict = calc
        if hits:
            it.task = hits[0] if len(calc) <= 1 else None
        elif re.search(r"\bmean\b|\bmedian\b|\baverage\b|standard deviation|\bvariance\b|\bspread\b|statistics", low):
            it.task = "statistics"
        elif re.search(r"\bsolve|\broots?\b|\bzeros?\b|\bsolutions?\b|\bsatisf|=", low):
            it.task = "solve"

    # ---- variable, domain, order
    m = re.search(r"\b(?:solve\s+for|with respect to|wrt|in terms of)\s+([a-z])\b", low) or re.search(
        r"\bd\s*/\s*d([a-z])\b", low
    )
    if not m:  # "... = 10 for y", "for t," (not "for a ..." / "for i ...", which are English)
        m = re.search(r"\bfor\s+([b-hj-z])\b(?!\s*[=(])", low)
    if m:
        it.variable = m.group(1)
    if re.search(r"\bpositive\b", low):
        it.domain = "positive"
    elif re.search(r"\breal\b(?![- ]world)", low):
        it.domain = "real"
    if re.search(r"\bsecond\b|\b2nd\b|\bf\s*''(?!')", low):
        it.order = 2
    elif re.search(r"\bthird\b|\b3rd\b|\bf\s*'''", low):
        it.order = 3

    # ---- calculus slots
    if it.task == "derivative":
        m = re.search(r"\b(?:at|when|where)\s+([a-z])\s*=\s*(-?[\w.]+(?:/\d+)?)", low)
        if m:
            it.point = _value(m.group(2))
    if it.task == "integral":
        m = (
            re.search(r"\bfrom\s+(?:[a-z]\s*=\s*)?(\S+?)\s+(?:to|until)\s+(?:[a-z]\s*=\s*)?(\S+)", low)
            or re.search(r"\bbetween\s+(?:[a-z]\s*=\s*)?(\S+?)\s+and\s+(?:[a-z]\s*=\s*)?(\S+)", low)
            or re.search(r"\b(?:over|on)\s+(?:the\s+)?(?:interval\s+)?\[\s*([^,\]]+?)\s*,\s*([^\]]+?)\s*\]", low)
            or re.search(r"\b(?:over|on)\s*\[\s*([^,\]]+?)\s*,\s*([^\]]+?)\s*\]", low)
        )
        if m:
            it.lower, it.upper = _value(m.group(1)), _value(m.group(2))
        elif re.search(r"\bfrom\s+\S+", low) or re.search(r"\b(?:to|up to)\s+" + _NUM + r"\s*$", low):
            it.partial_bounds = True
    if it.task == "limit":
        m = re.search(
            r"\b([a-z])\s*(?:->|approaches|tends to|goes to)\s*(-?" + _INF_WORDS + r"|-?[\w.]+(?:/\d+)?)(\+|-)?", low
        ) or re.search(r"\bapproach(?:es)?\s+(-?" + _INF_WORDS + r"|-?[\w.]+)(\+|-)?", low)
        if m:
            groups = [g for g in m.groups() if g is not None]
            point_tok = groups[1] if len(groups) >= 2 and len(groups[0]) == 1 and groups[0].isalpha() else groups[0]
            it.limit_point = _value(point_tok)
            if m.group(m.lastindex) in ("+", "-") and m.lastindex and m.group(m.lastindex) != point_tok:
                it.direction = m.group(m.lastindex)
        if re.search(r"from the right|from above", low):
            it.direction = "+"
        elif re.search(r"from the left|from below", low):
            it.direction = "-"

    # ---- probability slots
    if it.task == "probability":
        explicit = re.findall(r"\b(binomial|poisson|normal|gaussian|uniform|exponential)", low)
        if explicit:  # a named distribution beats contextual cues ("per minute", "trials")
            it.distribution = "normal" if explicit[0] == "gaussian" else explicit[0]
        else:
            it.distribution = next((d for d, pat in _DISTRIBUTIONS if re.search(pat, low)), None)
        rel_text = low
        if it.distribution == "uniform":  # the support "between a and b" is not the event
            rel_text = re.sub(rf"(?:between|from)\s+{_NUM}\s+(?:and|to)\s+{_NUM}", " ", low, count=1)
        it.relation = _relation(rel_text)
        for slot, pat in {
            "std": r"standard deviation|\bsd\b|\bstd\b|sigma|σ|variance|standard normal|\bz\b",
            "mean": r"\bmean\b|\baverage\b|\bmu\b|μ|standard normal|\bz\b|expected",
            "n": r"\btrials?\b|\btimes\b|\bflips?\b|\btosses\b|\bshots?\b|\bn\s*=|\bof \d+\b|\d+\s+\w+\s+(?:fail|are|each)|\beach of\b|\btakes \d+",
            "p": r"\bp\s*=|probability (?:of )?\d|probability \d|with probability|\d+\s*%|\bfair\b|chance",
            "lambda": r"lambda|λ|\brate\b|\bmean\b|\baverage\b|per (?:minute|hour|day|second|week)",
            "bounds": r"between|from \d|\[",
        }.items():
            if re.search(pat, low):
                it.mentions.add(slot)

    # ---- counting
    if it.task == "counting":
        if re.search(
            r"\bpermutations?\b|\barrange|\bline up\b|\border\b|\bfirst\b.*\bsecond\b|\b[p]\s*\(\s*\d", low
        ) or re.search(_ROLE_WORDS, low):
            it.counting = "permutations"
        if re.search(
            r"\bchoose\b|\bchos(?:e|en)\b|\bcombinations?\b|\bcommittee|\bhands?\b|\bselect|\bsubsets?\b|\bteams?\b|\b[c]\s*\(\s*\d",
            low,
        ):
            it.counting = "combinations"

    # ---- numbers
    it.numbers = _numbers_in(text)
    allowed = set(it.numbers)
    allowed |= {float(v) for w, v in _WORD_NUMBERS.items() if re.search(rf"\b{w}\b", low)}
    if re.search(r"\bfair\b|\bcoin\b", low):
        allowed.add(0.5)
    if re.search(r"standard normal|\bz\b", low):
        allowed |= {0.0, 1.0}
    if count_roles(low) >= 2:
        allowed.add(float(count_roles(low)))
    it.allowed_numbers = allowed
    return it


# distinct roles make order matter ("a president and a treasurer")
_ROLE_WORDS = r"\b(?:president|vice[- ]president|treasurer|secretary|chair(?:person)?|captain|manager|leader)\b"


def count_roles(low: str) -> int:
    return len(set(re.findall(_ROLE_WORDS, low)))


def _relation(low: str) -> tuple[str, str, str | None] | None:
    m = re.search(rf"p\s*\(\s*({_NUM})\s*(<=|<)\s*[a-z]\s*(<=|<)\s*({_NUM})\s*\)", low)
    if m:
        return ("between", m.group(1), m.group(4))
    m = re.search(rf"p\s*\(\s*[a-z]\s*(<=|>=|<|>|=)\s*({_NUM})\s*\)", low)
    if m:
        return ({"<=": "cdf", "<": "cdf<", ">=": "greater>=", ">": "greater", "=": "pmf"}[m.group(1)], m.group(2), None)
    m = re.search(rf"\bbetween\s+({_NUM})\s+and\s+({_NUM})", low)
    if m:
        return ("between", m.group(1), m.group(2))
    phrases = [
        (r"at most|no more than|not more than|or (?:fewer|less)", "cdf"),
        (r"at least|no fewer than|no less than|or more", "greater>="),
        (r"fewer than|less than|below|under|shorter than|before", "cdf<"),
        (r"more than|greater than|above|over|exceeds?|taller than|longer than|after|lasts more than", "greater"),
        (r"exactly", "pmf"),
        (r"within", "cdf"),
    ]
    for pat, q in phrases:
        m = re.search(rf"(?:{pat})\s+({_NUM})", low)
        if m:
            return (q, m.group(1), None)
    return None


# --------------------------------------------------------------------------- checks


_TOOL_TASK = {
    "solve_equation": "solve",
    "solve_system": "solve",
    "simplify_expression": "simplify",
    "factor_expression": "factor",
    "expand_expression": "expand",
    "differentiate": "derivative",
    "integrate": "integral",
    "limit": "limit",
    "descriptive_statistics": "statistics",
    "probability_distribution": "probability",
    "combinatorics": "counting",
}


def _call_task(call: ToolCall) -> str:
    if call.tool == "matrix_operation":
        return f"matrix:{call.args.get('operation')}"
    return _TOOL_TASK[call.tool]


def describe(call: ToolCall) -> str:
    """Plain-language restatement of what the plan will compute."""
    a = call.args
    var = a.get("variable") or "x"
    t = call.tool
    if t == "solve_equation":
        dom = {"real": " (real solutions only)", "positive": " (positive solutions only)"}.get(a.get("domain", ""), "")
        return f"Solve {a.get('equation')} for {a.get('variable') or 'the unknown'}{dom}."
    if t == "solve_system":
        return "Solve the system " + "; ".join(a.get("equations", [])) + "."
    if t in ("simplify_expression", "factor_expression", "expand_expression"):
        return f"{t.split('_')[0].capitalize()} {a.get('expression')}."
    if t == "differentiate":
        order = int(a.get("order") or 1)
        word = {1: "Derivative", 2: "Second derivative", 3: "Third derivative"}.get(
            order, f"Derivative of order {order}"
        )
        at = f", evaluated at {var} = {a['point']}" if a.get("point") is not None else ""
        return f"{word} of {a.get('expression')} with respect to {var}{at}."
    if t == "integrate":
        if a.get("lower") is not None:
            return f"Definite integral of {a.get('expression')} d{var} from {a['lower']} to {a.get('upper')}."
        return f"Antiderivative of {a.get('expression')} with respect to {var}."
    if t == "limit":
        side = {"+": " from the right", "-": " from the left"}.get(a.get("direction", ""), "")
        return f"Limit of {a.get('expression')} as {var} → {a.get('point')}{side}."
    if t == "matrix_operation":
        extra = f" and B = {a['other']}" if a.get("other") else (f", b = {a['vector']}" if a.get("vector") else "")
        return f"{a.get('operation', '').capitalize()} of A = {a.get('matrix')}{extra}."
    if t == "descriptive_statistics":
        return f"Summary statistics of the data {a.get('data')}."
    if t == "probability_distribution":
        params = ", ".join(f"{k}={v}" for k, v in (a.get("params") or {}).items())
        q, x, x2 = a.get("query"), a.get("x"), a.get("x2")
        event = {
            "pmf": f"P(X = {x})",
            "pdf": f"density at {x}",
            "cdf": f"P(X ≤ {x})",
            "greater": f"P(X ≥ {x})" if a.get("distribution") in ("binomial", "poisson") else f"P(X > {x})",
            "between": f"P({x} ≤ X ≤ {x2})",
        }.get(q, str(q))
        return f"{event} for X ~ {str(a.get('distribution')).capitalize()}({params})."
    if t == "combinatorics":
        op = a.get("operation")
        return {
            "factorial": f"{a.get('n')}!",
            "combinations": f"C({a.get('n')}, {a.get('k')}) (order does not matter)",
            "permutations": f"P({a.get('n')}, {a.get('k')}) (order matters)",
        }.get(op, str(op)) + "."
    return t


def _num_close(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))


def _grounded(value: Any, allowed: set[float]) -> bool:
    try:
        v = abs(float(value))
    except (TypeError, ValueError):
        return True  # symbolic values (pi/2, oo) are checked elsewhere
    return any(_num_close(v, a) for a in allowed)


def _scalar(value: Any) -> float | None:
    try:
        return float(sp.N(parse_expression(str(value)))) if not isinstance(value, (int, float)) else float(value)
    except (ExpressionError, TypeError, ValueError):
        return None


def _same_value(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is b
    sa, sb = str(a).replace(" ", ""), str(b).replace(" ", "")
    if sa == sb:
        return True
    try:
        ea, eb = parse_expression(sa), parse_expression(sb)
    except ExpressionError:
        return False
    if ea.is_infinite or eb.is_infinite:
        return bool(ea == eb)
    try:
        return _num_close(complex(sp.N(ea)).real, complex(sp.N(eb)).real) and abs(complex(sp.N(ea - eb))) < 1e-9
    except (TypeError, ValueError):
        return False


# ---- expression candidates in the text


_MATHY = re.compile(r"[\d=+\-*/^()]|^[a-z]$|^(?:sin|cos|tan|exp|log|ln|sqrt|abs)\b", re.I)


def _math_spans(text: str, strip_definitions: bool) -> list[str]:
    t = _normalise(text)
    if strip_definitions:
        t = re.sub(r"\b[a-z]\s*\(\s*[a-z]\s*\)\s*=\s*", " ", t)
        t = re.sub(r"\by\s*=\s*", " ", t)
    t = re.sub(r"\[\s*\[.*?\]\s*\]|\[[^\]]*\]", " ", t)  # matrices / intervals
    t = re.sub(r"\bd\s*\^?\d?\s*/\s*d[a-z]\^?\d?", " ", t)  # d/dx operator notation
    t = re.sub(r"\bd[a-z]\b", " ", t)  # differentials: dx, dt
    t = re.sub(r"\b(?:at|when|where|as)\s+[a-z]\s*(?:=|->)\s*-?[\w./]+", " ", t)  # slot specs
    t = re.sub(r"\b(?:from|between)\s+-?[\w./]+\s+(?:to|and)\s+-?[\w./]+", " ", t)
    tokens = re.split(r"\s+", t)
    runs, cur = [], []
    for tok in tokens:
        core = tok.strip(".,;:?!")
        short_symbols = re.fullmatch(r"[a-z]{2,3}", core or "") and core.lower() not in _ENGLISH_WORDS
        # prime notation such as y'' or f'(x) belongs to the math run (and makes it unparseable)
        short_symbols = short_symbols or re.fullmatch(r"[a-z]'+(?:\([a-z]\))?", core or "")
        if core and (short_symbols or (_MATHY.search(core) and not re.fullmatch(r"[A-Za-z]{2,}", core))):
            cur.append(core)
            if tok != core and tok[-1] in ".,;:?!":
                runs.append(cur)
                cur = []
        else:
            if cur:
                runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    spans: list[str] = []
    for run in runs:
        whole = " ".join(run)
        if _parses(whole):
            spans.append(whole)
            continue
        # Only operator notation (lim_{x->a}, d/dx, ∫) may be dropped from a run; dropping
        # anything else (y'' in "y'' + y = 0", "> 4" in "x^2 > 4") would leave a fragment.
        kept = [tok for tok in run if not _NOTATION.match(tok)]
        if len(kept) < len(run) and kept and _parses(" ".join(kept)):
            spans.append(" ".join(kept))
    return spans


_NOTATION = re.compile(r"^(?:lim\b|lim_|lim\(|d/d[a-z]|∫|int\b)", re.I)


def has_inequality(text: str) -> bool:
    return bool(re.search(r"(?<![<>=!])[<>](?![<>])|<=|>=|≤|≥", re.sub(r"->|=>", " ", _normalise(text))))


def _parses(piece: str) -> bool:
    try:
        (parse_equation if "=" in piece else parse_expression)(piece)
        return True
    except ExpressionError:
        return False


def _substantial(piece: str) -> bool:
    return bool(re.search(r"[+\-*/^()]|\d\s*[a-z]|[a-z]\s*\d|\b(?:sin|cos|tan|exp|log|ln|sqrt)\b", piece))


def _plain(e: sp.Basic) -> sp.Basic:
    return e.subs({s: sp.Symbol(s.name) for s in e.free_symbols})


def _equivalent(a: sp.Basic, b: sp.Basic, proportional: bool = False) -> bool:
    a, b = _plain(a), _plain(b)
    syms = sorted(a.free_symbols | b.free_symbols, key=lambda s: s.name)
    rng = random.Random(11)
    ratios = []
    for _ in range(5):
        subs = {s: sp.Rational(rng.randint(30, 270), 100) for s in syms}
        try:
            va, vb = complex(sp.N(a.subs(subs))), complex(sp.N(b.subs(subs)))
        except (TypeError, ValueError):
            return False
        if proportional:
            if abs(vb) < 1e-12:
                if abs(va) > 1e-9:
                    return False
                continue
            ratios.append(va / vb)
        elif abs(va - vb) > 1e-9 * max(1.0, abs(vb)):
            return False
    return not proportional or (
        bool(ratios) and all(abs(r - ratios[0]) < 1e-9 * max(1, abs(ratios[0])) for r in ratios)
    )


def _expr_of(text: str) -> sp.Basic | None:
    try:
        if "=" in text:
            eq = parse_equation(text)
            return eq.lhs - eq.rhs
        return parse_expression(text)
    except ExpressionError:
        return None


# ---- main check


@dataclass
class _Ctx:
    checks: list[IntentCheck] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    fixes: dict[str, Any] = field(default_factory=dict)
    # numbers inside written math that a plan expression was proven equivalent to (count as used)
    matched_numbers: set[float] = field(default_factory=set)

    def add(self, name: str, ok: bool, detail: str = "", severity: str = "error", fix: dict | None = None) -> None:
        self.checks.append(IntentCheck(name=name, passed=ok, detail=detail, severity=severity if not ok else "info"))
        if not ok and fix:
            self.fixes.update(fix)


# --------------------------------------------------------------------------- plan validation

_TOOL_CATEGORY = {
    "solve_equation": "algebra",
    "solve_system": "algebra",
    "simplify_expression": "algebra",
    "factor_expression": "algebra",
    "expand_expression": "algebra",
    "differentiate": "calculus",
    "limit": "calculus",
    "integrate": "integrals",
    "matrix_operation": "linear_algebra",
    "descriptive_statistics": "probability_statistics",
    "probability_distribution": "probability_statistics",
    "combinatorics": "probability_statistics",
}
_ENUMS = {
    "operation": {
        "matrix_operation": ("determinant", "inverse", "eigenvalues", "rank", "transpose", "multiply", "solve", "rref"),
        "combinatorics": ("combinations", "permutations", "factorial"),
    },
    "distribution": ("binomial", "poisson", "normal", "uniform", "exponential"),
    "query": ("pmf", "pdf", "cdf", "greater", "between"),
    "direction": ("+", "-", "+-"),
    "domain": ("complex", "real", "positive"),
}
_REQUIRED_EXTRA = {  # arguments a tool needs for a given operation / query
    ("matrix_operation", "multiply"): ("other",),
    ("matrix_operation", "solve"): ("vector",),
    ("probability_distribution", "between"): ("x2",),
}


def _is_number_like(v: Any) -> bool:
    if isinstance(v, bool):
        return False
    if isinstance(v, (int, float)):
        return True
    return isinstance(v, str) and _scalar(v) is not None


def validate_plan(call: ToolCall) -> list[str]:
    """Structural problems with a plan (independent of the question). An LLM plan
    that fails here is rejected, never 'repaired' by guessing."""
    import inspect

    from app.math_engine.tools import TOOLS

    errors: list[str] = []
    a = call.args
    sig = inspect.signature(TOOLS[call.tool])
    for name, param in sig.parameters.items():
        if param.default is inspect.Parameter.empty and a.get(name) in (None, "", [], {}):
            errors.append(f"missing required argument '{name}'")
    unknown = set(a) - set(sig.parameters)
    if unknown:
        errors.append(f"unexpected arguments {sorted(unknown)}")
    for key in ("expression", "equation"):
        if key in a and not isinstance(a[key], str):
            errors.append(f"'{key}' must be text")
    if "equations" in a and not (isinstance(a["equations"], list) and all(isinstance(e, str) for e in a["equations"])):
        errors.append("'equations' must be a list of equations")
    # every expression must pass the same safe parser the tools use (nothing is evaluated here)
    texts = [(k, a[k]) for k in ("expression", "equation") if isinstance(a.get(k), str)]
    texts += [("equations", e) for e in a.get("equations") or [] if isinstance(e, str)]
    for key, text in texts:
        try:
            (parse_equation if "=" in text else parse_expression)(text)
        except ExpressionError as exc:
            errors.append(f"'{key}' is not a valid mathematical expression ({exc})")
    for key in ("point", "lower", "upper", "x", "x2", "n", "k"):
        if a.get(key) is not None and not (_is_number_like(a[key]) or isinstance(a[key], str)):
            errors.append(f"'{key}' must be a number")
    for key in ("x", "x2", "n", "k"):
        if isinstance(a.get(key), str) and not _is_number_like(a[key]):
            errors.append(f"'{key}' = {a[key]!r} is not a number")
    if (
        "order" in a
        and a["order"] is not None
        and not (isinstance(a["order"], int) and not isinstance(a["order"], bool))
    ):
        errors.append("'order' must be an integer")
    if ("lower" in a) != ("upper" in a):
        errors.append("a definite integral needs both 'lower' and 'upper'")
    for key in ("matrix", "other"):
        if key in a and not (isinstance(a[key], list) and a[key] and all(isinstance(r, list) and r for r in a[key])):
            errors.append(f"'{key}' must be a non-empty list of rows")
    if "vector" in a and not isinstance(a["vector"], list):
        errors.append("'vector' must be a list")
    if "data" in a and not (isinstance(a["data"], list) and all(_is_number_like(v) for v in a["data"])):
        errors.append("'data' must be a list of numbers")
    if "params" in a:
        if not isinstance(a["params"], dict):
            errors.append("'params' must be an object")
        else:
            bad = [k for k, v in a["params"].items() if not _is_number_like(v)]
            if bad:
                errors.append(f"distribution parameters must be numbers: {bad}")
    for key, allowed in _ENUMS.items():
        if key in a and a[key] is not None:
            options = allowed.get(call.tool, ()) if isinstance(allowed, dict) else allowed
            if options and a[key] not in options:
                errors.append(f"{key} = {a[key]!r} is not one of {', '.join(options)}")
    for (tool, value), needed in _REQUIRED_EXTRA.items():
        if call.tool == tool and value in (a.get("operation"), a.get("query")):
            errors += [f"'{value}' needs argument '{n}'" for n in needed if a.get(n) in (None, [], "")]
    return errors


def _self_consistency(call: ToolCall) -> list[str]:
    """Contradictions inside an LLM plan: its category vs its tool, and its own
    restatement vs its arguments. The question text is not used here."""
    problems: list[str] = []
    if call.source == "llm" and call.category.value != _TOOL_CATEGORY[call.tool]:
        problems.append(f"the plan's category '{call.category.value}' contradicts its tool '{call.tool}'")
    if call.interpretation:
        told = extract_intent(call.interpretation)
        task = _call_task(call)
        if (
            told.task
            and told.task not in ("solve", "matrix")
            and told.task != task
            and not task.startswith(f"{told.task}")
        ):
            problems.append(f"the plan's own interpretation describes '{told.task}' but its arguments compute '{task}'")
        order = int(call.args.get("order") or 1) if call.tool == "differentiate" else None
        if order is not None and told.order and told.order != order:
            problems.append(f"the interpretation says order {told.order} but the arguments use order {order}")
    return problems


def _rejected(call: ToolCall, name: str, details: list[str]) -> tuple[IntentVerification, dict[str, Any]]:
    detail = "; ".join(details)
    return (
        IntentVerification(
            status=IntentStatus.MISMATCH,
            interpretation=_safe_describe(call),
            checks=[IntentCheck(name=name, passed=False, detail=detail, severity="error")],
            clarification=f"The plan was rejected: {detail}.",
        ),
        {},
    )


def _safe_describe(call: ToolCall) -> str:
    try:
        return describe(call)
    except Exception:
        return call.tool


def check_intent(
    problem: str,
    call: ToolCall,
    declared_ambiguities: list[str] | None = None,
    declared_assumptions: list[str] | None = None,
) -> tuple[IntentVerification, dict[str, Any]]:
    """Check that ``call`` faithfully answers ``problem``.

    Order: (1) structural validation, (2) internal consistency of the plan,
    (3) alignment with the question. The check fails closed: any malformed
    plan or internal error yields MISMATCH, never acceptance.

    Returns the verification and a dict of safe argument fixes (slots that the
    question states explicitly but the plan omitted or changed).
    """
    # LLM self-reports travel with the plan; they can only lower trust, so never drop them
    if declared_ambiguities is None:
        declared_ambiguities = list(call.ambiguities)
    if declared_assumptions is None:
        declared_assumptions = list(call.assumptions)
    errors = validate_plan(call)
    if errors:
        return _rejected(call, "plan is well-formed", errors)
    contradictions = _self_consistency(call)
    if contradictions:
        return _rejected(call, "plan is self-consistent", contradictions)
    try:
        return _check_alignment(problem, call, declared_ambiguities, declared_assumptions)
    except Exception as exc:  # fail closed
        return _rejected(call, "intent check completed", [f"internal error ({type(exc).__name__})"])


def _check_alignment(
    problem: str,
    call: ToolCall,
    declared_ambiguities: list[str] | None,
    declared_assumptions: list[str] | None,
) -> tuple[IntentVerification, dict[str, Any]]:
    it = extract_intent(problem)
    ctx = _Ctx()
    task = _call_task(call)

    # 1. task / operation
    if it.task_conflict:
        ctx.add(
            "single task",
            False,
            "The question asks for several operations (" + ", ".join(it.task_conflict) + "); please ask one at a time.",
            "missing",
        )
    elif it.task is None:
        ctx.add("task stated", True, "No explicit task word; inferred from the plan.")
        ctx.assumptions.append(f"Interpreted the request as: {describe(call)}")
    elif it.task == "matrix" and call.tool == "matrix_operation":
        ctx.add(
            "matrix operation stated", False, "The question does not say which matrix operation to perform.", "missing"
        )
    elif it.task != task and not (it.task == "solve" and task == "matrix:solve"):
        ctx.add("task matches question", False, f"The question asks for '{it.task}' but the plan computes '{task}'.")
    else:
        ctx.add("task matches question", True, task)

    # 2. numbers in the arguments must come from the question
    _check_numbers(problem, it, call, ctx)

    # 3. tool-specific slots
    if call.tool in ("solve_equation", "solve_system"):
        _check_solve(problem, it, call, ctx)
    elif call.tool in (
        "simplify_expression",
        "factor_expression",
        "expand_expression",
        "differentiate",
        "integrate",
        "limit",
    ):
        _check_expression(problem, it, call, ctx)
        _check_calculus_slots(it, call, ctx)
    elif call.tool == "matrix_operation":
        _check_matrices(it, call, ctx)
    elif call.tool == "probability_distribution":
        _check_probability(it, call, ctx)
    elif call.tool == "combinatorics":
        _check_counting(problem, it, call, ctx)
    elif call.tool == "descriptive_statistics":
        _check_data(problem, call, ctx)

    _check_unused(it, call, ctx)

    # 4. what the LLM planner itself declared
    for amb in declared_ambiguities or []:
        ctx.add("planner-declared ambiguity", False, amb, "missing")
    ctx.assumptions += list(declared_assumptions or [])

    failed = [c for c in ctx.checks if not c.passed]
    if any(c.severity == "error" for c in failed):
        status = IntentStatus.MISMATCH
    elif any(c.severity == "missing" for c in failed):
        status = IntentStatus.AMBIGUOUS
    elif failed or ctx.assumptions:
        status = IntentStatus.ASSUMED
        ctx.assumptions += [c.detail for c in failed if c.severity == "warning" and c.detail not in ctx.assumptions]
    else:
        status = IntentStatus.CONFIRMED
    clarification = None
    if status in (IntentStatus.MISMATCH, IntentStatus.AMBIGUOUS):
        first = next(c for c in failed if c.severity in ("error", "missing"))
        clarification = first.detail
    return (
        IntentVerification(
            status=status,
            interpretation=describe(call),
            checks=ctx.checks,
            assumptions=ctx.assumptions,
            clarification=clarification,
        ),
        ctx.fixes,
    )


def _arg_scalars(call: ToolCall) -> list[tuple[str, Any]]:
    a, out = call.args, []
    for key in ("point", "lower", "upper", "x", "x2", "n", "k"):
        if a.get(key) is not None:
            out.append((key, a[key]))
    for key, value in (a.get("params") or {}).items():
        out.append((f"params.{key}", value))
    for v in a.get("data") or []:
        out.append(("data", v))
    return out


def _check_numbers(problem: str, it: Intent, call: ToolCall, ctx: _Ctx) -> None:
    allowed = set(it.allowed_numbers)
    discrete = call.args.get("distribution") in ("binomial", "poisson")
    if discrete:  # strict inequalities on integers shift the threshold by one
        allowed |= {n + 1 for n in it.numbers} | {n - 1 for n in it.numbers if n >= 1}
    if call.args.get("distribution") == "normal" and "variance" in problem.lower():
        allowed |= {n**0.5 for n in it.numbers}
    low = _normalise(problem).lower()
    for key, value in _arg_scalars(call):
        v = _scalar(value)
        if isinstance(value, str) and re.search(r"[a-z]", value.lower()):
            # symbolic value such as pi/3, e or oo: its numbers and constants must be in the question
            consts_ok = all(
                re.search(pat, low)
                for word, pat in (("pi", r"\bpi\b|π"), ("oo", _INF_WORDS + r"|\binfinite"), ("e", r"\be\b|\bexp"))
                if re.search(rf"\b{word}\b", value.lower())
            )
            nums_ok = all(_grounded(n, allowed) for n in _numbers_in(value))
            if consts_ok and nums_ok:
                continue
            ctx.add(f"{key} grounded", False, f"{key} = {value} does not appear in the question.")
            continue
        if v is None or not _grounded(v, allowed):
            if v is None:
                continue
            slot = key.split(".")[-1]
            missing = (
                call.tool == "probability_distribution"
                and key.startswith("params.")
                and (
                    {
                        "std": "std",
                        "sigma": "std",
                        "mean": "mean",
                        "mu": "mean",
                        "n": "n",
                        "p": "p",
                        "lambda": "lambda",
                        "rate": "lambda",
                        "a": "bounds",
                        "b": "bounds",
                    }.get(slot, slot)
                    not in it.mentions
                )
            )
            if missing:
                ctx.add(
                    f"{slot} given",
                    False,
                    f"The question does not state the {_SLOT_NAMES.get(slot, slot)}; "
                    f"the plan assumed {slot} = {value}. Please provide it.",
                    "missing",
                )
            else:
                ctx.add(f"{key} grounded", False, f"{key} = {value} does not appear in the question.")


def _check_unused(it: Intent, call: ToolCall, ctx: _Ctx) -> None:
    """Numbers in the question that the plan ignores (a warning: they may be context)."""
    used = set(ctx.matched_numbers)
    for _, value in _arg_scalars(call):
        v = _scalar(value)
        if v is not None:
            used.add(abs(v))
        used |= set(_numbers_in(str(value)))
    for key in ("expression", "equation"):
        if call.args.get(key):
            used |= set(_numbers_in(str(call.args[key])))
    for eq in call.args.get("equations") or []:
        used |= set(_numbers_in(eq))
    for key in ("matrix", "other", "vector"):
        used |= set(_numbers_in(str(call.args.get(key) or "")))
    if call.tool == "differentiate":
        used.add(float(call.args.get("order") or 1))
    if call.tool == "probability_distribution" and call.args.get("distribution") in ("binomial", "poisson"):
        used |= {u + 1 for u in used} | {u - 1 for u in used}
    unused = [n for n in it.numbers if not any(_num_close(n, u) for u in used)]
    if unused:
        ctx.add(
            "all stated numbers used",
            False,
            "Number(s) in the question not used by the plan: "
            + ", ".join(f"{n:g}" for n in sorted(set(unused)))
            + " (treated as context).",
            "warning",
        )


_SLOT_NAMES = {
    "std": "standard deviation",
    "mean": "mean",
    "n": "number of trials",
    "p": "success probability",
    "lambda": "rate",
    "rate": "rate",
    "a": "interval",
    "b": "interval",
}


def _check_solve(problem: str, it: Intent, call: ToolCall, ctx: _Ctx) -> None:
    a = call.args
    if has_inequality(problem):
        ctx.add(
            "equation, not inequality",
            False,
            "The question contains an inequality; solving inequalities is not supported yet.",
        )
    equations = [a["equation"]] if call.tool == "solve_equation" else list(a.get("equations") or [])
    exprs = [_expr_of(e) for e in equations]
    all_spans = _math_spans(problem, strip_definitions=False)
    spans = [s for s in all_spans if "=" in s]
    if not spans and call.tool == "solve_equation":  # "roots of x^3 - 6x^2 + 11x - 6" means expression = 0
        spans = [s for s in all_spans if _substantial(s)]
    text_eqs = [e for e in (_expr_of(s) for s in spans) if e is not None]
    if not text_eqs:
        if _check_reconstruction(problem, it, equations, ctx):
            ctx.assumptions.append("The equation was reconstructed from words: " + "; ".join(equations))
    else:
        for eq_text, e in zip(equations, exprs, strict=True):
            hits = [
                sp_
                for sp_, t in zip(spans, text_eqs, strict=False)
                if e is not None and _equivalent(e, t, proportional=True)
            ]
            ok = bool(hits)
            for sp_ in hits:
                ctx.matched_numbers |= set(_numbers_in(sp_))
            ctx.add(
                "equation matches question",
                ok,
                f"{eq_text}" if ok else f"'{eq_text}' does not match any equation in the question.",
            )
        if call.tool == "solve_equation" and len(text_eqs) > 1:
            ctx.add(
                "all equations used",
                False,
                f"The question contains {len(text_eqs)} equations but the plan solves only one.",
                "error",
            )
    if call.tool == "solve_equation" and re.fullmatch(r"\s*[a-z]\s*=\s*-?[\w./]+\s*", str(a.get("equation", ""))):
        ctx.add(
            "non-trivial equation",
            False,
            f"'{a.get('equation')}' is already solved; it is probably a value "
            "at which something should be evaluated, not the equation to solve.",
        )
    if call.tool == "solve_equation" and exprs[0] is not None:
        names = sorted({s.name for s in exprs[0].free_symbols})
        var = a.get("variable")
        if var and var not in names:
            ctx.add(
                "variable occurs in equation",
                False,
                f"The plan solves for {var}, which does not occur in the equation.",
            )
        if len(names) > 1 and not it.variable:
            ctx.add(
                "unknown specified",
                False,
                f"The equation has several unknowns ({', '.join(names)}) but only one "
                "equation. Which variable should be solved for, or is there another equation?",
                "missing",
            )
        elif it.variable and var and var != it.variable:
            ctx.add(
                "variable matches question",
                False,
                f"The question solves for {it.variable}, the plan for {var}.",
                fix={"variable": it.variable},
            )
        dom = a.get("domain", "complex")
        if it.domain and dom != it.domain:
            ctx.add(
                "domain constraint applied",
                False,
                f"The question asks for {it.domain} solutions but the plan returns {dom} ones.",
                fix={"domain": it.domain},
            )
        elif not it.domain and dom != "complex":
            ctx.add(
                "domain constraint stated",
                False,
                f"The plan restricts to {dom} solutions, which the question does not ask for.",
            )


def _atomic_numbers(text: str) -> list[float]:
    """Number literals as written (no derived ratios): '2*sqrt(36)' -> [2, 36]."""
    return [abs(float(m)) for m in re.findall(r"(?<![A-Za-z_\d.])\d+(?:\.\d+)?", text)]


def _structural_constants(problem: str, it: Intent) -> set[float]:
    """Constants a faithful translation from words may introduce without computing anything."""
    low = _normalise(problem).lower()
    allowed = set(it.allowed_numbers) | {0.0, 1.0}
    if re.search(r"squar|sqrt|\broot|\bhalf|\btwice|\bdouble|quadratic|√", low):
        allowed.add(2.0)
    if re.search(r"\bcub(?:e|ed|ic)\b", low):
        allowed.add(3.0)
    if re.search(r"percent|%", low):
        allowed.add(100.0)
    return allowed


def _check_reconstruction(problem: str, it: Intent, texts: list[str], ctx: _Ctx) -> bool:
    """Provenance of math the plan wrote although the question contains no matching written math.

    A faithful translation from words may only use numbers stated in the question (as digits or
    words) plus structural constants; it must not reduce the problem to a constant. Values computed
    by the planner itself (e.g. sqrt(36) from a triangle with sides 3, 4, 5 via Heron's formula)
    cannot be verified by the math engine and are rejected. Returns True if the reconstruction is
    acceptable (then it is reported as an assumption, never as confirmed)."""
    allowed = _structural_constants(problem, it)
    ok = True
    for text in texts:
        expr = _expr_of(text)
        if expr is not None and not expr.free_symbols:
            ctx.add(
                "no precomputed values",
                False,
                f"The plan reduces the question to the constant '{text}', which is not written in the question: "
                "the steps that produced it were done outside the math engine and cannot be verified.",
            )
            ok = False
            continue
        if expr is not None:
            low_text = _normalise(problem)
            strangers = sorted(
                s_.name
                for s_ in expr.free_symbols
                if not re.search(rf"(?<![A-Za-z]){re.escape(s_.name)}(?![A-Za-z])", low_text)
            )
            if strangers:
                ctx.add(
                    "symbols traceable to the question",
                    False,
                    f"'{text}' introduces symbol(s) {', '.join(strangers)} that do not appear in the question.",
                )
                ok = False
        foreign = sorted({n for n in _atomic_numbers(text) if not any(_num_close(n, a) for a in allowed)})
        if foreign:
            ctx.add(
                "values traceable to the question",
                False,
                f"'{text}' uses value(s) {', '.join(f'{n:g}' for n in foreign)} that do not come from the question; "
                "intermediate calculations done outside the math engine cannot be verified.",
            )
            ok = False
    return ok


def _check_expression(problem: str, it: Intent, call: ToolCall, ctx: _Ctx) -> None:
    arg = call.args.get("expression")
    e = _expr_of(str(arg)) if arg else None
    if e is None:
        return
    spans = [s for s in _math_spans(problem, strip_definitions=True) if "=" not in s and _substantial(s)]
    cands = [c for c in (_expr_of(s) for s in spans) if c is not None]
    if not cands:
        if _check_reconstruction(problem, it, [str(arg)], ctx):
            ctx.assumptions.append(f"The expression was reconstructed from words: {arg}")
        return
    matches = [sp_ for sp_, c in zip(spans, cands, strict=False) if _equivalent(e, c)]
    ok = bool(matches)
    for sp_ in matches:
        ctx.matched_numbers |= set(_numbers_in(sp_))
    ctx.add(
        "expression matches question",
        ok,
        str(arg) if ok else f"'{arg}' does not match the expression in the question ({spans[0]}).",
    )
    names = sorted({s.name for s in e.free_symbols})
    var = call.args.get("variable")
    if var and names and var not in names:
        ctx.add(
            "variable occurs in expression",
            False,
            f"The plan uses the variable {var}, which does not occur in {arg} (variables: {', '.join(names)}).",
        )
    if call.tool in ("differentiate", "integrate", "limit") and len(names) > 1 and not it.variable:
        if not (call.args.get("variable") and re.search(rf"\b{call.args['variable']}\s*(?:=|->)", _normalise(problem))):
            ctx.add(
                "variable specified",
                False,
                f"The expression has several variables ({', '.join(names)}); with respect to which one?",
                "missing",
            )


def _check_calculus_slots(it: Intent, call: ToolCall, ctx: _Ctx) -> None:
    a = call.args
    if call.tool == "differentiate":
        order = int(a.get("order") or 1)
        want = it.order or 1
        ctx.add(
            "derivative order",
            order == want,
            f"order {order}"
            if order == want
            else f"The question asks for order {want}, the plan computes order {order}.",
            fix={"order": want},
        )
        if it.point is not None:
            ok = a.get("point") is not None and _same_value(a["point"], it.point)
            ctx.add(
                "evaluation point",
                ok,
                f"at {it.point}" if ok else f"The question evaluates at {it.point}; the plan uses {a.get('point')}.",
                fix={"point": it.point},
            )
        elif a.get("point") is not None:
            ctx.add(
                "evaluation point stated",
                False,
                f"The plan evaluates at {a['point']}, which the question does not ask.",
            )
        if it.variable and a.get("variable") and a["variable"] != it.variable:
            ctx.add(
                "variable matches question",
                False,
                f"d/d{it.variable} asked, d/d{a['variable']} planned.",
                fix={"variable": it.variable},
            )
    elif call.tool == "integrate":
        has_bounds = a.get("lower") is not None or a.get("upper") is not None
        if it.lower is not None:
            ok = _same_value(a.get("lower"), it.lower) and _same_value(a.get("upper"), it.upper)
            swapped = _same_value(a.get("lower"), it.upper) and _same_value(a.get("upper"), it.lower)
            ctx.add(
                "integration bounds",
                ok,
                f"[{it.lower}, {it.upper}]"
                if ok
                else (
                    "The plan swaps the integration limits."
                    if swapped
                    else f"The question integrates from {it.lower} to {it.upper}; the plan uses "
                    f"{a.get('lower')} to {a.get('upper')}."
                ),
                fix={"lower": it.lower, "upper": it.upper},
            )
        elif it.partial_bounds:
            ctx.add(
                "both integration limits given",
                False,
                "Only one integration limit is given; what is the other?",
                "missing",
            )
        elif has_bounds:
            ctx.add("integration bounds stated", False, "The plan adds integration limits the question does not give.")
    elif call.tool == "limit":
        if it.limit_point is None:
            ctx.add(
                "limit point given", False, "The question does not say which point the variable approaches.", "missing"
            )
        else:
            ok = _same_value(a.get("point"), it.limit_point)
            ctx.add(
                "limit point",
                ok,
                f"→ {it.limit_point}"
                if ok
                else f"The question takes the limit at {it.limit_point}; the plan at {a.get('point')}.",
                fix={"point": it.limit_point},
            )
        want_dir = it.direction or "+-"
        got_dir = a.get("direction") or "+-"
        if not str(a.get("point")).endswith("oo") and got_dir != want_dir:
            ctx.add(
                "limit direction",
                False,
                f"The question asks for direction '{want_dir}', the plan uses '{got_dir}'.",
                fix={"direction": want_dir},
            )


def _check_matrices(it: Intent, call: ToolCall, ctx: _Ctx) -> None:
    a = call.args

    def same(m1: Any, m2: Any) -> bool:
        try:
            return sp.Matrix(m1).shape == sp.Matrix(m2).shape and all(
                _num_close(float(sp.N(x)), float(sp.N(y))) for x, y in zip(sp.Matrix(m1), sp.Matrix(m2), strict=True)
            )
        except (TypeError, ValueError, sp.SympifyError):
            return False

    if it.matrices:
        ok = same(a.get("matrix"), it.matrices[0])
        ctx.add(
            "matrix matches question", ok, "A as written" if ok else "The plan's matrix differs from the question's."
        )
        if len(it.matrices) > 1 and a.get("operation") == "multiply":
            ctx.add("second matrix matches", same(a.get("other"), it.matrices[1]), "")
    if a.get("operation") == "solve" and it.vectors:
        ok = same([[v] for v in a.get("vector") or []], [[v] for v in it.vectors[0]])
        ctx.add("right-hand side matches", ok, "" if ok else "The plan's vector b differs from the question's.")


def _check_probability(it: Intent, call: ToolCall, ctx: _Ctx) -> None:
    a = call.args
    dist = a.get("distribution")
    if it.distribution and dist != it.distribution:
        ctx.add(
            "distribution", False, f"The question describes a {it.distribution} distribution; the plan uses {dist}."
        )
    elif not it.distribution:
        ctx.assumptions.append(f"Modelled as a {dist} distribution.")
    if it.relation is None:
        ctx.assumptions.append("The probability event was inferred: " + describe(call))
        return
    q, thr, thr2 = it.relation
    discrete = dist in ("binomial", "poisson")
    want: tuple[str, float | None, float | None]
    t = float(thr)
    if q == "between":
        want = ("between", t, float(thr2) if thr2 else None)
    elif q == "pmf":
        want = ("pmf" if discrete else "pdf", t, None)
    elif q == "cdf":
        want = ("cdf", t, None)
    elif q == "cdf<":
        want = ("cdf", t - 1 if discrete else t, None)
    elif q == "greater>=":
        want = ("greater", t, None)
    else:  # strict greater
        want = ("greater", t + 1 if discrete else t, None)
    got_q = a.get("query")
    got_x = _scalar(a.get("x"))
    got_x2 = _scalar(a.get("x2")) if a.get("x2") is not None else None
    ok = (
        got_q == want[0]
        and got_x is not None
        and _num_close(got_x, want[1])
        and (want[2] is None or (got_x2 is not None and _num_close(got_x2, want[2])))
    )
    if not ok and not discrete and q == "greater>=" and got_q == "greater" and got_x is not None:
        ok = _num_close(got_x, t)
    ctx.add(
        "probability event matches question",
        ok,
        describe(call)
        if ok
        else f"The question asks for {_event_text(q, thr, thr2)}; the plan computes {describe(call)}",
    )


def _event_text(q: str, thr: str, thr2: str | None) -> str:
    return {
        "between": f"P({thr} ≤ X ≤ {thr2})",
        "pmf": f"P(X = {thr})",
        "cdf": f"P(X ≤ {thr})",
        "cdf<": f"P(X < {thr})",
        "greater>=": f"P(X ≥ {thr})",
        "greater": f"P(X > {thr})",
    }[q]


def _check_data(problem: str, call: ToolCall, ctx: _Ctx) -> None:
    """The data set must be exactly the list of numbers written in the question."""
    text = _normalise(problem)
    lists = re.findall(r"-?\d+(?:\.\d+)?(?:\s*(?:,|;|\band\b)\s*-?\d+(?:\.\d+)?)+", text)
    if not lists:
        ctx.assumptions.append("The data values were reconstructed from words.")
        return
    longest = max(lists, key=lambda chunk: len(re.findall(r"-?\d+(?:\.\d+)?", chunk)))
    stated = sorted(float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", longest))
    try:
        planned = sorted(float(v) for v in call.args.get("data") or [])
    except (TypeError, ValueError):
        planned = []
    ok = len(stated) == len(planned) and all(_num_close(x, y) for x, y in zip(stated, planned, strict=True))
    ctx.add(
        "data matches question",
        ok,
        f"{len(stated)} values"
        if ok
        else f"The question lists {len(stated)} values ({longest}); the plan uses {len(planned)}: {call.args.get('data')}.",
    )


def _check_counting(problem: str, it: Intent, call: ToolCall, ctx: _Ctx) -> None:
    op = call.args.get("operation")
    if it.counting and op != "factorial" and op != it.counting:
        ctx.add(
            "counting rule",
            False,
            f"The question is about {it.counting} (order "
            f"{'matters' if it.counting == 'permutations' else 'does not matter'}); the plan uses {op}.",
        )
    n, k = _scalar(call.args.get("n")), _scalar(call.args.get("k"))
    m = re.search(r"(\d+)\s*(?:choose|c)\s*(\d+)|\b[pc]\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)", _normalise(problem).lower())
    if m and n is not None and k is not None:
        a_, b_ = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
        ok = _num_close(n, float(a_)) and _num_close(k, float(b_))
        ctx.add(
            "n and k as written",
            ok,
            "" if ok else f"The question gives n = {a_}, k = {b_}; the plan uses n = {n:g}, k = {k:g}.",
        )
    elif n is not None and k is not None and k > n:
        ctx.add("n ≥ k", False, f"k = {k:g} exceeds n = {n:g}; the roles may be swapped.")
