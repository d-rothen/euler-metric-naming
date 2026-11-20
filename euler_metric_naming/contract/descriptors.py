"""Descriptor semantics: effective compare, merging and relevance (§3.1, §4.2, §8.4)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from ._json import MISSING, json_equal
from .closed import COMPARE_RULES, ROLE_DEFAULT_COMPARE
from .conditions import evaluate
from .globs import glob_any

#: Strictness of compare rules: equal > measure > pair > ignore.
_STRICTNESS = {"equal": 3, "measure": 2, "pair": 1, "ignore": 0}


def effective_compare(descriptor: Optional[Mapping[str, Any]]) -> str:
    """The compare rule a descriptor stands for (§3.1, §3.2).

    No descriptor, an unknown or missing role, or an unknown compare value all
    mean ``equal``. An unknown role makes the descriptor ``equal`` whatever
    its ``compare`` says.
    """
    if not isinstance(descriptor, Mapping):
        return "equal"
    role = descriptor.get("role")
    if role not in ROLE_DEFAULT_COMPARE:
        return "equal"
    if "compare" not in descriptor:
        return ROLE_DEFAULT_COMPARE[role]
    compare = descriptor.get("compare")
    return compare if compare in COMPARE_RULES else "equal"


@dataclass
class Cell:
    """A metric cell: base name and axis values by role (§8.1)."""

    name: str
    base: str
    roles: dict = field(default_factory=dict)  # role -> [values], position order

    def values(self, role: str) -> list:
        return list(self.roles.get(role, ()))

    def value(self, role: str) -> Optional[str]:
        values = self.roles.get(role)
        return values[0] if values else None


def affects_cell(descriptor: Optional[Mapping[str, Any]], cell: Cell) -> bool:
    """Whether a descriptor's ``affects`` covers the cell (§4.2).

    A missing or malformed ``affects`` affects every cell.
    """
    if not isinstance(descriptor, Mapping):
        return True
    affects = descriptor.get("affects")
    if not isinstance(affects, Mapping):
        return True
    metrics = affects.get("metrics", ["*"])
    if isinstance(metrics, list) and not glob_any(metrics, cell.base):
        return False
    axes = affects.get("axes")
    if isinstance(axes, Mapping):
        for role, allowed in axes.items():
            if not isinstance(allowed, list):
                continue
            if not any(value in allowed for value in cell.values(role)):
                return False
    return True


@dataclass
class MergedDescriptor:
    """The descriptor of a key as both sides describe it (§8.4)."""

    key: str
    compare: str
    label: str
    observe: Any = None
    default: Any = MISSING
    hint: Optional[str] = None
    sides: tuple = ()  # the descriptors present (A first)
    candidates: tuple = ()  # the descriptors whose compare and conditions apply

    def affects(self, cell: Cell) -> bool:
        if not self.sides:
            return True
        return any(affects_cell(descriptor, cell) for descriptor in self.sides)

    def skipped(self, context_a: tuple, context_b: tuple) -> bool:
        """The appliesWhen / inertIf rule of §7, under every candidate."""
        if not self.candidates:
            return False
        return all(
            _skipped_by(descriptor, context_a, context_b) for descriptor in self.candidates
        )


def _skipped_by(descriptor: Mapping[str, Any], context_a: tuple, context_b: tuple) -> bool:
    applies = descriptor.get("appliesWhen")
    if applies is not None:
        if evaluate(applies, *context_a) is False and evaluate(applies, *context_b) is False:
            return True
    inert = descriptor.get("inertIf")
    if inert is not None:
        if evaluate(inert, *context_a) is True and evaluate(inert, *context_b) is True:
            return True
    return False


def merge_descriptors(key: str, a: Any, b: Any) -> MergedDescriptor:
    """Merge the descriptors both sides give *key*; the stricter wins (§8.4)."""
    da = a if isinstance(a, Mapping) else None
    db = b if isinstance(b, Mapping) else None
    sides = tuple(d for d in (da, db) if d is not None)
    if not sides:
        return MergedDescriptor(key=key, compare="equal", label=key)
    if len(sides) == 2:
        ca, cb = effective_compare(da), effective_compare(db)
        if _STRICTNESS[ca] > _STRICTNESS[cb]:
            candidates: tuple = (da,)
        elif _STRICTNESS[cb] > _STRICTNESS[ca]:
            candidates = (db,)
        else:
            candidates = sides
    else:
        candidates = sides
    compare = effective_compare(candidates[0])

    observe = candidates[0].get("observe")
    if any(not json_equal(d.get("observe"), observe) for d in candidates[1:]):
        observe = None

    default = MISSING
    defaults = [d["default"] for d in candidates if "default" in d]
    if defaults and all(json_equal(value, defaults[0]) for value in defaults[1:]):
        default = defaults[0]

    label = next(
        (d["label"] for d in candidates if isinstance(d.get("label"), str) and d["label"]),
        key,
    )
    hint = next(
        (d["hint"] for d in candidates if isinstance(d.get("hint"), str) and d["hint"]),
        None,
    )
    return MergedDescriptor(
        key=key,
        compare=compare,
        label=label,
        observe=observe,
        default=default,
        hint=hint,
        sides=sides,
        candidates=candidates,
    )
