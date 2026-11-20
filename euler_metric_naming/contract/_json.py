"""Small JSON helpers shared by the contract modules (§8.0 JSON equality)."""
from __future__ import annotations

from typing import Any


class _Missing:
    """Sentinel for "not recorded", distinct from JSON ``null``."""

    _instance = None

    def __new__(cls) -> _Missing:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "MISSING"

    def __bool__(self) -> bool:
        return False


MISSING: Any = _Missing()


def is_number(value: Any) -> bool:
    """True for JSON numbers (``bool`` is not a number)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def is_record(value: Any) -> bool:
    return isinstance(value, dict)


def json_equal(a: Any, b: Any) -> bool:
    """JSON equality of §8.0.

    Numbers are equal when numerically equal (``1 == 1.0``); ``true`` is not a
    number. Strings compare code point by code point, lists element-wise and
    objects by key set and values. ``null`` equals only ``null``.
    """
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if is_number(a) and is_number(b):
        try:
            return float(a) == float(b)  # IEEE doubles, as in ECMAScript
        except OverflowError:
            return a == b
    if isinstance(a, str) and isinstance(b, str):
        return a == b
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(json_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return False
        return all(json_equal(a[key], b[key]) for key in a)
    return False

