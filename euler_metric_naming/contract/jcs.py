"""RFC 8785 JSON Canonicalization Scheme (JCS), ``docs/contract.md`` §9.1.

The output is byte-identical to the TS artifact's ``canonicalize``: object keys
sorted by UTF-16 code units, no whitespace, strings escaped as ECMAScript
``JSON.stringify`` does, numbers serialized as ECMAScript
``Number.prototype.toString``. Non-finite numbers, lone surrogates and
integers outside ±(2**53 − 1) are errors.
"""
from __future__ import annotations

import math
from decimal import Decimal
from typing import Any

MAX_SAFE_INTEGER = 2**53 - 1

_ESCAPES = {
    '"': '\\"',
    "\\": "\\\\",
    "\b": "\\b",
    "\f": "\\f",
    "\n": "\\n",
    "\r": "\\r",
    "\t": "\\t",
}


class CanonicalizationError(ValueError):
    """Raised for values that have no canonical JSON form."""


def canonicalize(value: Any) -> str:
    """Return the JCS text of a JSON value (dict, list, str, number, bool, None)."""
    out: list[str] = []
    _write(value, out)
    return "".join(out)


def format_number(value: float | int) -> str:
    """Serialize a number like ECMAScript ``Number.prototype.toString``."""
    if isinstance(value, bool):
        raise CanonicalizationError("booleans are not numbers")
    if isinstance(value, int):
        if abs(value) > MAX_SAFE_INTEGER:
            raise CanonicalizationError(
                f"integer {value} is outside the safe range ±(2**53 - 1)"
            )
        return str(value)
    if not isinstance(value, float):
        raise CanonicalizationError(f"not a number: {value!r}")
    if math.isnan(value) or math.isinf(value):
        raise CanonicalizationError(f"non-finite number {value!r}")
    if value == 0:
        return "0"  # also -0
    sign = "-" if value < 0 else ""
    # repr() is the shortest round-trip form, as ECMAScript requires.
    _, digit_tuple, exponent = Decimal(repr(abs(value))).normalize().as_tuple()
    digits = "".join(str(d) for d in digit_tuple)
    k = len(digits)
    n = int(exponent) + k
    if k <= n <= 21:
        return sign + digits + "0" * (n - k)
    if 0 < n <= 21:
        return sign + digits[:n] + "." + digits[n:]
    if -6 < n <= 0:
        return sign + "0." + "0" * (-n) + digits
    e = n - 1
    exp = ("+" if e >= 0 else "-") + str(abs(e))
    if k == 1:
        return sign + digits + "e" + exp
    return sign + digits[0] + "." + digits[1:] + "e" + exp


def quote(text: str) -> str:
    """Return the JSON string literal of *text* as ``JSON.stringify`` writes it."""
    out = ['"']
    for ch in text:
        escape = _ESCAPES.get(ch)
        if escape is not None:
            out.append(escape)
            continue
        code = ord(ch)
        if code < 0x20:
            out.append("\\u%04x" % code)
        elif 0xD800 <= code <= 0xDFFF:
            raise CanonicalizationError("lone surrogate in string")
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def utf16_key(text: str) -> bytes:
    """Sort key ordering strings by UTF-16 code units."""
    return text.encode("utf-16-be", "surrogatepass")


def _write(value: Any, out: list[str]) -> None:
    if value is None:
        out.append("null")
    elif value is True:
        out.append("true")
    elif value is False:
        out.append("false")
    elif isinstance(value, (int, float)):
        out.append(format_number(value))
    elif isinstance(value, str):
        out.append(quote(value))
    elif isinstance(value, (list, tuple)):
        out.append("[")
        for index, item in enumerate(value):
            if index:
                out.append(",")
            _write(item, out)
        out.append("]")
    elif isinstance(value, dict):
        keys = list(value)
        for key in keys:
            if not isinstance(key, str):
                raise CanonicalizationError(f"object key {key!r} is not a string")
        quoted = {key: quote(key) for key in keys}
        out.append("{")
        for index, key in enumerate(sorted(keys, key=utf16_key)):
            if index:
                out.append(",")
            out.append(quoted[key])
            out.append(":")
            _write(value[key], out)
        out.append("}")
    else:
        raise CanonicalizationError(f"not a JSON value: {type(value).__name__}")
