"""Stratifications (``docs/contract.md`` §4.6): identity, bins and edge schemes."""
from __future__ import annotations

import math
from typing import Any, Optional, Sequence

from ._json import is_number
from .closed import STRATIFICATION_DISPLAY_FIELDS


def identity_projection(stratification: Any) -> dict:
    """The stratification without its display fields and null-valued fields.

    Two stratifications are the same exactly when their identity projections
    are JSON-equal (§4.6).
    """
    if not isinstance(stratification, dict):
        return {}
    return {
        key: value
        for key, value in stratification.items()
        if key not in STRATIFICATION_DISPLAY_FIELDS and value is not None
    }


def bin_range(stratification: Any, label: str) -> Optional[tuple]:
    """``(lo, hi)`` of a bin label, the total spanning first to last edge."""
    if not isinstance(stratification, dict):
        return None
    edges = stratification.get("edges")
    if not isinstance(edges, list) or len(edges) < 2:
        return None
    if not all(is_number(edge) for edge in edges):
        return None
    if label == stratification.get("total"):
        return (edges[0], edges[-1])
    labels = stratification.get("labels")
    if not isinstance(labels, list) or len(labels) != len(edges) - 1:
        return None
    for index, candidate in enumerate(labels):
        if candidate == label:
            return (edges[index], edges[index + 1])
    return None


def sqrt_equal_width_edges(lo: float, hi: float, bins: int = 3) -> list[float]:
    """Edges of the ``sqrt_equal_width`` scheme (euler-eval's benchmark bins).

    ``bins`` bins of equal width in ``sqrt(depth)`` between ``lo`` and ``hi``:
    ``e_i = (sqrt(lo) + i * (sqrt(hi) - sqrt(lo)) / bins) ** 2``, squared as
    ``x * x``; the first and last edges are ``lo`` and ``hi`` themselves.
    """
    lo, hi = float(lo), float(hi)
    if not (math.isfinite(lo) and math.isfinite(hi)) or lo < 0 or hi <= lo:
        raise ValueError(f"invalid range [{lo}, {hi}]")
    if bins < 1:
        raise ValueError("bins must be >= 1")
    root_lo = math.sqrt(lo)
    root_hi = math.sqrt(hi)
    step = (root_hi - root_lo) / bins
    edges = [lo]
    for index in range(1, bins):
        root = root_lo + index * step
        edges.append(root * root)
    edges.append(hi)
    return edges


def sqrt_stratification(
    lo: float,
    hi: float,
    *,
    labels: Sequence[str] = ("near", "mid", "far"),
    total: str = "all",
    label: str = "GT depth",
    variable: str = "gt_depth_m",
    unit: str = "m",
    after: Optional[Sequence[str]] = ("depth.cap.max_m",),
    edges: Optional[Sequence[float]] = None,
) -> dict:
    """The GT-depth stratification euler-eval's benchmark bins describe."""
    computed = list(edges) if edges is not None else sqrt_equal_width_edges(
        lo, hi, len(labels)
    )
    stratification = {
        "label": label,
        "variable": variable,
        "unit": unit,
        "edges": computed,
        "labels": list(labels),
        "closed": "left",
        "lastClosed": True,
        "total": total,
        "scheme": "sqrt_equal_width",
    }
    if after:
        stratification["after"] = list(after)
    return stratification
