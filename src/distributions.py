"""jemss.distributions

Small, safe parser + sampler for a subset of Julia ``Distributions.jl``
expressions.

The upstream Julia simulator stores distribution *objects* in input files via
string expressions such as ``"Normal(0, 1)"`` or ``"Exponential(0.5)"`` and
constructs them with ``Meta.parse |> eval``.

In Python we **do not** evaluate arbitrary code.  Instead this module parses a
small, explicit subset of distribution expressions and provides sampling using
Python's standard library :mod:`random`.

Supported expressions (case-sensitive, Julia-style):

* ``Normal(μ, σ)``
* ``LogNormal(μ, σ)``
* ``Uniform(a, b)``
* ``Exponential(θ)``  (scale/mean θ)
* ``Gamma(k, θ)``     (shape k, scale θ)
* ``Erlang(k, θ)``    (alias of ``Gamma`` with integer shape)
* ``Weibull(k, θ)``   (shape k, scale θ)
* ``truncated(d, lo, hi)`` where ``d`` is one of the above
* ``Constant(x)`` / ``Deterministic(x)``

If an unsupported expression is encountered, a :class:`ValueError` is raised.
"""

from __future__ import annotations

from dataclasses import dataclass
import ast
import math
import random
import re
from typing import Any, List, Optional, Sequence, Tuple


_CALL_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)\s*$")


def _split_top_level_args(s: str) -> List[str]:
    """Split a Julia-style argument list by commas at top level."""

    args: List[str] = []
    depth = 0
    start = 0
    for i, ch in enumerate(s):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "," and depth == 0:
            args.append(s[start:i].strip())
            start = i + 1
    tail = s[start:].strip()
    if tail:
        args.append(tail)
    return args


def _eval_number_expr(expr: str) -> float:
    """Safely evaluate a small arithmetic expression into a float.

    Allowed:
    * numeric literals
    * +, -, *, /, **
    * parentheses
    * names: Inf, NaN, pi, π
    """

    expr = expr.strip()
    if expr == "":
        raise ValueError("Empty numeric expression")

    # Map Julia identifiers.
    name_map = {
        "Inf": float("inf"),
        "NaN": float("nan"),
        "pi": math.pi,
        "π": math.pi,
    }

    node = ast.parse(expr, mode="eval").body

    def _eval(n: ast.AST) -> float:
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
            v = _eval(n.operand)
            return +v if isinstance(n.op, ast.UAdd) else -v
        if isinstance(n, ast.BinOp) and isinstance(
            n.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
        ):
            a = _eval(n.left)
            b = _eval(n.right)
            if isinstance(n.op, ast.Add):
                return a + b
            if isinstance(n.op, ast.Sub):
                return a - b
            if isinstance(n.op, ast.Mult):
                return a * b
            if isinstance(n.op, ast.Div):
                return a / b
            # Pow
            return a ** b
        if isinstance(n, ast.Name) and n.id in name_map:
            return float(name_map[n.id])
        raise ValueError(f"Unsupported numeric expression: {expr!r}")

    return float(_eval(node))


@dataclass(frozen=True)
class ParsedDistribution:
    name: str
    params: Tuple[float, ...] = ()
    base: Optional["ParsedDistribution"] = None
    lower: Optional[float] = None
    upper: Optional[float] = None

    def sample(self, rng: random.Random) -> float:
        n = self.name
        p = self.params

        if n in {"Constant", "Deterministic"}:
            return float(p[0])

        if n == "Normal":
            mu, sigma = p
            return float(mu) if sigma == 0 else float(rng.normalvariate(mu, sigma))

        if n == "LogNormal":
            mu, sigma = p
            # Python's lognormvariate uses the same parameterisation as
            # Distributions.jl: log(X) ~ Normal(mu, sigma).
            if sigma == 0:
                return float(math.exp(mu))
            return float(rng.lognormvariate(mu, sigma))

        if n == "Uniform":
            a, b = p
            return float(rng.uniform(a, b))

        if n == "Exponential":
            (theta,) = p
            if theta <= 0:
                return 0.0
            # Python uses rate λ; Julia Exponential(θ) uses scale/mean θ.
            return float(rng.expovariate(1.0 / theta))

        if n == "Gamma":
            k, theta = p
            if k <= 0 or theta <= 0:
                return 0.0
            return float(rng.gammavariate(k, theta))

        if n == "Weibull":
            k, theta = p
            if k <= 0 or theta <= 0:
                return 0.0
            # random.weibullvariate(alpha, beta): alpha=scale, beta=shape
            return float(rng.weibullvariate(theta, k))

        if n == "Truncated":
            if self.base is None or self.lower is None or self.upper is None:
                raise ValueError("Truncated distribution missing base/bounds")
            lo, hi = float(self.lower), float(self.upper)
            if lo >= hi:
                raise ValueError("Invalid truncated bounds")
            # Rejection sampling; suitable for positive delay distributions.
            for _ in range(10_000):
                v = float(self.base.sample(rng))
                if lo <= v <= hi:
                    return v
            # Fallback: clamp
            return min(max(float(self.base.sample(rng)), lo), hi)

        raise ValueError(f"Unsupported distribution for sampling: {n}")

    def mean(self) -> Optional[float]:
        n = self.name
        p = self.params
        if n in {"Constant", "Deterministic"}:
            return float(p[0])
        if n == "Normal":
            return float(p[0])
        if n == "LogNormal":
            mu, sigma = p
            return float(math.exp(mu + 0.5 * sigma * sigma))
        if n == "Uniform":
            a, b = p
            return float(0.5 * (a + b))
        if n == "Exponential":
            (theta,) = p
            return float(theta)
        if n == "Gamma":
            k, theta = p
            return float(k * theta)
        if n == "Weibull":
            k, theta = p
            return float(theta * math.gamma(1.0 + 1.0 / k))
        # Truncated mean is non-trivial; we don't rely on it in the simulator.
        return None


def parse_distribution_spec(spec: Any) -> ParsedDistribution:
    """Parse *spec* into a :class:`ParsedDistribution`.

    * If *spec* is None/empty -> Constant(0)
    * If *spec* is numeric -> Constant(spec)
    * If *spec* is a string -> parse supported Julia expressions
    """

    if spec is None:
        return ParsedDistribution("Constant", (0.0,))
    if isinstance(spec, (int, float)):
        return ParsedDistribution("Constant", (float(spec),))
    if not isinstance(spec, str):
        raise ValueError(f"Unsupported distribution spec type: {type(spec).__name__}")

    s = spec.strip()
    if s == "":
        return ParsedDistribution("Constant", (0.0,))

    # Strip module qualifiers in the constructor name (e.g. "Distributions.Normal(...)"),
    # keeping the last identifier before the argument list.
    paren = s.find("(")
    if paren > 0:
        prefix = s[:paren].strip()
        if "." in prefix:
            s = prefix.split(".")[-1] + s[paren:]

    # Support lowercase truncated(...) spelling too.
    if s.startswith("truncated") or s.startswith("Truncated"):
        m = _CALL_RE.match(s)
        if not m:
            raise ValueError(f"Invalid truncated spec: {spec!r}")
        name = m.group(1)
        args = _split_top_level_args(m.group(2))
        if len(args) != 3:
            raise ValueError("truncated(d, lo, hi) requires three arguments")
        base = parse_distribution_spec(args[0])
        lo = _eval_number_expr(args[1])
        hi = _eval_number_expr(args[2])
        return ParsedDistribution("Truncated", (), base=base, lower=lo, upper=hi)

    m = _CALL_RE.match(s)
    if not m:
        # Maybe it's just a number.
        return ParsedDistribution("Constant", (_eval_number_expr(s),))

    name = m.group(1)
    arg_str = m.group(2).strip()
    args = _split_top_level_args(arg_str) if arg_str else []

    # Normalize aliases.
    if name == "Erlang":
        name = "Gamma"

    if name in {"Constant", "Deterministic"}:
        if len(args) != 1:
            raise ValueError(f"{name}(x) requires one argument")
        return ParsedDistribution(name, (_eval_number_expr(args[0]),))

    if name == "Normal":
        if len(args) != 2:
            raise ValueError("Normal(μ, σ) requires two arguments")
        return ParsedDistribution("Normal", (_eval_number_expr(args[0]), _eval_number_expr(args[1])))

    if name == "LogNormal":
        if len(args) != 2:
            raise ValueError("LogNormal(μ, σ) requires two arguments")
        return ParsedDistribution("LogNormal", (_eval_number_expr(args[0]), _eval_number_expr(args[1])))

    if name == "Uniform":
        if len(args) != 2:
            raise ValueError("Uniform(a, b) requires two arguments")
        return ParsedDistribution("Uniform", (_eval_number_expr(args[0]), _eval_number_expr(args[1])))

    if name == "Exponential":
        if len(args) != 1:
            raise ValueError("Exponential(θ) requires one argument")
        return ParsedDistribution("Exponential", (_eval_number_expr(args[0]),))

    if name == "Gamma":
        if len(args) != 2:
            raise ValueError("Gamma(k, θ) requires two arguments")
        return ParsedDistribution("Gamma", (_eval_number_expr(args[0]), _eval_number_expr(args[1])))

    if name == "Weibull":
        if len(args) != 2:
            raise ValueError("Weibull(k, θ) requires two arguments")
        return ParsedDistribution("Weibull", (_eval_number_expr(args[0]), _eval_number_expr(args[1])))

    raise ValueError(f"Unsupported distribution spec: {spec!r}")
