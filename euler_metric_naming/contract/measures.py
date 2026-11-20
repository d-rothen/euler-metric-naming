"""The five observation measures and their phrases, ``docs/contract.md`` §3.3.

Every computation runs in IEEE double arithmetic in the same order as the TS
artifact, so ``d`` and the phrases are identical in both languages.
"""
from __future__ import annotations

import math
from decimal import ROUND_HALF_UP, Decimal, localcontext
from typing import Any, Optional

from ._json import is_number


def measure(kind: Any, a: Any, b: Any) -> Optional[float]:
    """Return ``d >= 0`` for two observed values, or ``None`` when unmeasurable."""
    if kind == "region":
        return _region(a, b)
    if kind == "count":
        return _count(a, b)
    if kind == "fraction":
        return _fraction(a, b)
    if kind == "factor":
        return _factor(a, b)
    if kind == "size":
        return _size(a, b)
    return None


def _nonneg(value: Any) -> Optional[float]:
    if not is_number(value):
        return None
    number = float(value)
    if not math.isfinite(number) or number < 0:
        return None
    return number


def _region_value(value: Any) -> Optional[tuple]:
    if not isinstance(value, dict) or not isinstance(value.get("frame"), str):
        return None
    box = value.get("box")
    if not isinstance(box, list) or len(box) != 4:
        return None
    numbers = [_nonneg(item) for item in box]
    if any(number is None for number in numbers):
        return None
    return (value["frame"], *numbers)


def _region(a: Any, b: Any) -> Optional[float]:
    ra, rb = _region_value(a), _region_value(b)
    if ra is None or rb is None or ra[0] != rb[0]:
        return None
    _, top1, left1, height1, width1 = ra
    _, top2, left2, height2, width2 = rb
    inter_h = max(0.0, min(top1 + height1, top2 + height2) - max(top1, top2))
    inter_w = max(0.0, min(left1 + width1, left2 + width2) - max(left1, left2))
    inter = inter_h * inter_w
    union = height1 * width1 + height2 * width2 - inter
    if not union > 0:
        return None
    return 1.0 - inter / union


def _relative(a: float, b: float) -> float:
    if a == 0 and b == 0:
        return 0.0
    return abs(a - b) / max(abs(a), abs(b))


def _count(a: Any, b: Any) -> Optional[float]:
    na, nb = _nonneg(a), _nonneg(b)
    if na is None or nb is None:
        return None
    return _relative(na, nb)


def _fraction(a: Any, b: Any) -> Optional[float]:
    na, nb = _nonneg(a), _nonneg(b)
    if na is None or nb is None or na > 1 or nb > 1:
        return None
    return abs(na - nb)


def _factor_values(value: Any) -> Optional[list]:
    items = value if isinstance(value, list) else [value]
    if not items:
        return None
    numbers = [_nonneg(item) for item in items]
    if any(number is None or number <= 0 for number in numbers):
        return None
    return numbers


def _factor(a: Any, b: Any) -> Optional[float]:
    if isinstance(a, list) != isinstance(b, list):
        return None
    fa, fb = _factor_values(a), _factor_values(b)
    if fa is None or fb is None or len(fa) != len(fb):
        return None
    worst = 1.0
    for x, y in zip(fa, fb):
        worst = max(worst, x / y, y / x)
    return worst - 1.0


def _size_product(value: Any) -> Optional[float]:
    if not isinstance(value, list) or not value:
        return None
    product = 1.0
    for item in value:
        number = _nonneg(item)
        if number is None:
            return None
        product = product * number
    return product


def _size(a: Any, b: Any) -> Optional[float]:
    if not isinstance(a, list) or not isinstance(b, list) or len(a) != len(b):
        return None
    pa, pb = _size_product(a), _size_product(b)
    if pa is None or pb is None:
        return None
    return _relative(pa, pb)


# -- phrases -------------------------------------------------------------------


def format_percent(d: float) -> str:
    """``d`` in percent with one decimal: ``floor(d*1000 + 0.5) / 10``."""
    t = math.floor(d * 1000 + 0.5)
    if t == 0 and d != 0:
        return "<0.1"
    return f"{t // 10}.{t % 10}"


def format_sig3(value: float) -> str:
    """ECMAScript ``toPrecision(3)`` with trailing fraction zeros removed."""
    number = float(value)
    if not math.isfinite(number):
        return str(number)
    if number == 0:
        return "0"
    with localcontext() as context:
        context.prec = 1200
        exact = Decimal(number)
        sign = "-" if exact < 0 else ""
        exact = abs(exact)
        exponent = exact.adjusted()
        scaled = exact.scaleb(-exponent).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        if scaled >= 10:
            exponent += 1
            scaled = exact.scaleb(-exponent).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
        digits = str(scaled).replace(".", "")
    if exponent < -6 or exponent >= 3:
        mantissa = _strip_fraction(digits[0] + "." + digits[1:])
        return sign + mantissa + "e" + ("+" if exponent >= 0 else "-") + str(abs(exponent))
    if exponent >= 0:
        text = digits[: exponent + 1]
        fraction = digits[exponent + 1 :]
        if fraction:
            text += "." + fraction
    else:
        text = "0." + "0" * (-exponent - 1) + digits
    return sign + _strip_fraction(text)


def _strip_fraction(text: str) -> str:
    if "." not in text:
        return text
    return text.rstrip("0").rstrip(".")


def format_factor(value: Any) -> str:
    """Factor value(s) for phrases: 3 significant digits, lists joined by ``×``."""
    items = value if isinstance(value, list) else [value]
    return "×".join(format_sig3(item) if is_number(item) else "?" for item in items)


def phrase(kind: str, label: str, a: Any, b: Any, d: float) -> str:
    """The verdict phrase of a measured difference (§3.3)."""
    if kind == "factor":
        return f"{label} ({format_factor(a)} vs {format_factor(b)})"
    unit = "pp" if kind == "fraction" else "%"
    return f"{label} differs by {format_percent(d)} {unit}"
