"""jemss.xml_utils

Small XML helper functions mirroring the convenience helpers in
``file/file_io.jl``.

The Julia implementation uses LightXML / EzXML and evaluates element
contents as Julia expressions for some fields. In Python we keep things
explicit and safe:

* ``elt_text`` returns stripped text.
* ``elt_value`` parses primitive values (bool, int, float, None) and
  simple list syntax.
* ``elt_interp_text`` supports $var substitution using
  :class:`string.Template`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import ast
import os
import re
import string
import xml.etree.ElementTree as ET


def xml_file_root(filename: str) -> ET.Element:
    """Parse an XML file and return its root element."""
    tree = ET.parse(filename)
    return tree.getroot()


def find_elt(parent: ET.Element, tag: str) -> Optional[ET.Element]:
    """Find the first direct child element with a given tag."""
    for child in list(parent):
        if child.tag == tag:
            return child
    return None


def contains_elt(parent: ET.Element, tag: str) -> bool:
    return find_elt(parent, tag) is not None


def children_node_names(parent: ET.Element) -> List[str]:
    """Return tag names of direct child *elements*."""
    return [child.tag for child in list(parent) if isinstance(child.tag, str)]


def elt_text(elt: ET.Element) -> str:
    """Return stripped element text ("" if missing)."""
    return (elt.text or "").strip()


def elt_text_required(parent: ET.Element, tag: str) -> str:
    child = find_elt(parent, tag)
    if child is None:
        raise KeyError(f"Element not found: {tag}")
    return elt_text(child)


_BOOL_MAP = {"true": True, "false": False}


def _normalise_value_expr(text: str) -> str:
    """Convert a small subset of Julia-ish literals to Python literals."""
    s = text.strip()
    # booleans
    s = re.sub(r"\btrue\b", "True", s, flags=re.IGNORECASE)
    s = re.sub(r"\bfalse\b", "False", s, flags=re.IGNORECASE)
    # nothing
    s = re.sub(r"\bnothing\b", "None", s, flags=re.IGNORECASE)
    return s


def parse_value(text: str) -> Any:
    """Parse a simple scalar/list value from element text.

    This is intentionally conservative; it is *not* a general expression
    evaluator.
    """

    s = (text or "").strip()
    if s == "":
        return ""

    sl = s.lower()
    if sl in _BOOL_MAP:
        return _BOOL_MAP[sl]
    if sl in ("none", "nothing", "null"):
        return None

    # simple numbers
    try:
        if re.fullmatch(r"[+-]?\d+", s):
            return int(s)
        if re.fullmatch(r"[+-]?(?:\d+\.\d*|\d*\.\d+)(?:[eE][+-]?\d+)?", s) or re.fullmatch(
            r"[+-]?\d+(?:[eE][+-]?\d+)", s
        ):
            return float(s)
    except ValueError:
        pass

    # lists / tuples / dicts (limited) via literal_eval
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        expr = _normalise_value_expr(s)
        try:
            return ast.literal_eval(expr)
        except Exception:
            # fallback: treat as comma-separated tokens
            inner = s[1:-1]
            toks = [t.strip() for t in inner.split(",") if t.strip()]
            return toks

    return s


def elt_value_required(parent: ET.Element, tag: str) -> Any:
    return parse_value(elt_text_required(parent, tag))


def interpolate_string(text: str, variables: Optional[Dict[str, str]] = None) -> str:
    """Perform $var interpolation using a restricted variable mapping."""

    mapping: Dict[str, str] = {}
    mapping.update({k: str(v) for k, v in os.environ.items()})
    if variables:
        mapping.update({k: str(v) for k, v in variables.items()})

    # Strip wrapping quotes if present (common in configs copied from Julia)
    s = (text or "").strip().strip('"')
    return string.Template(s).safe_substitute(mapping)


def elt_interp_text_required(parent: ET.Element, tag: str, variables: Optional[Dict[str, str]] = None) -> str:
    return interpolate_string(elt_text_required(parent, tag), variables)


def elt_attr(elt: ET.Element, name: str) -> Optional[str]:
    return elt.attrib.get(name)


def elt_attr_value(elt: ET.Element, name: str) -> Any:
    val = elt_attr(elt, name)
    if val is None:
        raise KeyError(f"Attribute not found: {name}")
    return parse_value(val)
