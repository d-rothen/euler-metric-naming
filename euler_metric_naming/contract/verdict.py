"""The verdict engine of ``docs/contract.md`` §8.

:func:`compare` judges one metric cell of two normalized contracts and returns
a :class:`Verdict`. It is modality-agnostic: everything it needs travels in
the contracts. The TS artifact's ``compare`` is a line-by-line port.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Collection, Mapping, Optional

from ..axes import decompose
from ._json import MISSING, is_number, json_equal
from .closed import AXIS_ROLES, SUPPORTED_MAJORS, TIERS
from .descriptors import Cell, MergedDescriptor, merge_descriptors
from .evalfile import EvalFile, file_evidence
from .globs import glob_any
from .hashing import valid_rev
from .jcs import format_number
from .legacy import axis_declarations
from .measures import format_sig3, measure, phrase
from .normalize import schema_major
from .strata import bin_range, identity_projection

SEPARATOR = " · "

# Steps, in the order their reasons are listed (§8).
STEP_SCHEMA = 1
STEP_REFERENCES = 2
STEP_REVISION = 3
STEP_STRATIFICATION = 4
STEP_PARAMETERS = 5
STEP_POPULATION = 6

_SEVERITY = {tier: index for index, tier in enumerate(TIERS)}


@dataclass
class Reason:
    """One difference found between the two sides of a cell."""

    tier: str
    code: str
    label: str
    text: str
    key: Optional[str] = None
    measure: Optional[dict] = None
    hint: Optional[str] = None
    step: int = 0

    def to_dict(self) -> dict:
        out: dict = {"tier": self.tier, "code": self.code}
        if self.key is not None:
            out["key"] = self.key
        out["label"] = self.label
        out["text"] = self.text
        if self.measure is not None:
            out["measure"] = dict(self.measure)
        if self.hint is not None:
            out["hint"] = self.hint
        return out


@dataclass
class Verdict:
    """The judgement of one metric cell (§3.11)."""

    tier: str
    label: str
    reasons: list = field(default_factory=list)
    shared: Optional[int] = None

    @property
    def codes(self) -> list:
        return [reason.code for reason in self.reasons]

    def to_dict(self) -> dict:
        out: dict = {"tier": self.tier, "label": self.label}
        if self.shared is not None:
            out["shared"] = self.shared
        out["reasons"] = [reason.to_dict() for reason in self.reasons]
        return out


def _record(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _finish(reasons: list, shared: Optional[int]) -> Verdict:
    reasons = sorted(
        reasons,
        key=lambda r: (-_SEVERITY[r.tier], r.step, r.key if r.key is not None else ""),
    )
    tier = "same"
    for reason in reasons:
        if _SEVERITY[reason.tier] > _SEVERITY[tier]:
            tier = reason.tier
    if tier == "same":
        label = "same protocol"
    else:
        first = next(reason for reason in reasons if reason.tier == tier)
        if tier == "paired":
            label = first.text
        elif tier == "comparable":
            label = "comparable" + SEPARATOR + first.text
        else:
            label = "not comparable" + SEPARATOR + first.text
    return Verdict(tier=tier, label=label, reasons=reasons, shared=shared)


# -- 8.1 cells ---------------------------------------------------------------------


@dataclass
class _Side:
    contract: dict
    cell: Cell
    axes_by_role: dict  # role -> [axis name], position order (axes with a value)


def _decompose_cell(contract: Mapping[str, Any], name: str) -> Optional[_Side]:
    info = _record(contract.get("metricSet"))
    namespace = info.get("metricNamespace")
    if not isinstance(namespace, str):
        return None
    declarations = axis_declarations(info.get("axes"))
    if declarations is None:
        return None
    try:
        parsed = decompose(name, namespace, declarations)
    except (ValueError, TypeError):
        return None
    roles: dict = {}
    axes_by_role: dict = {}
    axis_roles = _record(contract.get("axes"))
    ordered = sorted(declarations.items(), key=lambda kv: kv[1].position)
    for axis_name, _ in ordered:
        value = parsed.axes.get(axis_name)
        if value is None:
            continue
        role = _record(axis_roles.get(axis_name)).get("role")
        if role not in AXIS_ROLES:
            role = "family"
        roles.setdefault(role, []).append(value)
        axes_by_role.setdefault(role, []).append(axis_name)
    return _Side(dict(contract), Cell(name=name, base=parsed.metric, roles=roles), axes_by_role)


def per_item_name(contract: Mapping[str, Any], name: str) -> Optional[str]:
    """The per-item name of an aggregate metric name (§4.11)."""
    side = _decompose_cell(contract, name)
    return _per_item(side, side.cell.base) if side is not None else None


def _per_item(side: _Side, base: str) -> str:
    info = _record(side.contract.get("metricSet"))
    declarations = axis_declarations(info.get("axes")) or {}
    parsed = decompose(side.cell.name, info["metricNamespace"], declarations)
    reduction_axes = set(side.axes_by_role.get("reduction", ())[:1])
    parts = [info["metricNamespace"]]
    for axis_name, _ in sorted(declarations.items(), key=lambda kv: kv[1].position):
        value = parsed.axes.get(axis_name)
        if value is not None and axis_name not in reduction_axes:
            parts.append(value)
    parts.append(base)
    return ".".join(parts)


# -- merged metric definition (8.1, 8.2) ---------------------------------------------


def _definition(contract: Mapping[str, Any], base: str) -> Optional[dict]:
    definition = _record(contract.get("metrics")).get(base)
    return definition if isinstance(definition, dict) else None


def _merged_definition(a: Optional[dict], b: Optional[dict]) -> dict:
    if a is None or b is None:
        return dict(a or b or {})
    merged = dict(a)
    req_a, req_b = a.get("requires"), b.get("requires")
    if isinstance(req_a, list) and isinstance(req_b, list):
        merged["requires"] = list(dict.fromkeys(req_a + req_b))
    else:
        merged.pop("requires", None)
    dep_a = a.get("dependsOn") if isinstance(a.get("dependsOn"), list) else ["*"]
    dep_b = b.get("dependsOn") if isinstance(b.get("dependsOn"), list) else ["*"]
    merged["dependsOn"] = list(dict.fromkeys(dep_a + dep_b))
    pool_a = a.get("pool") if isinstance(a.get("pool"), dict) else {"form": "none"}
    pool_b = b.get("pool") if isinstance(b.get("pool"), dict) else {"form": "none"}
    merged["pool"] = pool_a if json_equal(pool_a, pool_b) else {"form": "none"}
    if not json_equal(a.get("reducer"), b.get("reducer")):
        merged.pop("reducer", None)
    return merged


# -- relevance (8.4) -------------------------------------------------------------------


def _context(contract: Mapping[str, Any]) -> tuple:
    return (_record(contract.get("parameters")), _record(contract.get("observations")))


def _relevant(
    key: str,
    descriptor: MergedDescriptor,
    cell: Cell,
    definition: dict,
    context_a: tuple,
    context_b: tuple,
    reference: bool = False,
) -> bool:
    depends = definition.get("dependsOn")
    depends = depends if isinstance(depends, list) else ["*"]
    if not glob_any(depends, key):
        return False
    if not descriptor.affects(cell):
        return False
    if reference:
        requires = definition.get("requires")
        if isinstance(requires, list) and key[len("reference."):] not in requires:
            return False
    return not descriptor.skipped(context_a, context_b)


# -- the engine ------------------------------------------------------------------------


def compare(
    a: Mapping[str, Any],
    b: Mapping[str, Any],
    metric_name: str,
    items_a: Optional[Collection[str]] = None,
    items_b: Optional[Collection[str]] = None,
    per_item_a: Optional[Collection[str]] = None,
    per_item_b: Optional[Collection[str]] = None,
) -> Verdict:
    """Judge metric cell *metric_name* of normalized contracts *a* and *b*.

    ``items_*`` are the evaluated item id sets (``None`` = unknown);
    ``per_item_*`` the per-item names with values on each side (per-file
    metric names and ``items.series`` names; ``None`` = none known).
    """
    if not (
        schema_major(a.get("schemaVersion")) in SUPPORTED_MAJORS
        and schema_major(b.get("schemaVersion")) in SUPPORTED_MAJORS
    ):
        return _unsupported(a, b)

    # 8.1 cell
    side_a = _decompose_cell(a, metric_name)
    side_b = _decompose_cell(b, metric_name)
    if side_a is None or side_b is None:
        return _finish(
            [Reason("incomparable", "schema", metric_name, "metric name does not decompose", step=STEP_SCHEMA)],
            None,
        )
    if side_a.cell.base != side_b.cell.base or not json_equal(side_a.cell.roles, side_b.cell.roles):
        return _finish(
            [Reason("incomparable", "schema", metric_name, "metric name decomposes differently", step=STEP_SCHEMA)],
            None,
        )
    cell = side_a.cell
    base = cell.base
    def_a, def_b = _definition(a, base), _definition(b, base)
    definition = _merged_definition(def_a, def_b)
    context_a, context_b = _context(a), _context(b)
    descriptors_a = _record(a.get("descriptors"))
    descriptors_b = _record(b.get("descriptors"))
    reasons: list = []

    # 8.5 references
    refs_a, refs_b = _record(a.get("references")), _record(b.get("references"))
    for role in sorted(set(refs_a) | set(refs_b)):
        key = f"reference.{role}"
        descriptor = merge_descriptors(key, descriptors_a.get(key), descriptors_b.get(key))
        if not _relevant(key, descriptor, cell, definition, context_a, context_b, True):
            continue
        reasons.extend(_reference_reasons(key, descriptor, refs_a.get(role), refs_b.get(role)))

    # 8.2 revision
    rev_a = def_a.get("rev") if def_a is not None else None
    rev_b = def_b.get("rev") if def_b is not None else None
    has_a, has_b = valid_rev(rev_a), valid_rev(rev_b)
    if has_a and has_b and float(rev_a) != float(rev_b):
        reasons.append(
            Reason(
                "incomparable",
                "revision",
                base,
                f"{base} definition revised (rev {format_number(rev_a)} vs {format_number(rev_b)})",
                step=STEP_REVISION,
            )
        )
    elif has_a != has_b:
        reasons.append(
            Reason("incomparable", "not_recorded", base, f"{base} revision not recorded", step=STEP_REVISION)
        )

    # 8.3 stratification
    strata_a = side_a.axes_by_role.get("stratum", [])
    strata_b = side_b.axes_by_role.get("stratum", [])
    for index, value in enumerate(cell.values("stratum")):
        reasons.extend(
            _stratum_reasons(a, b, strata_a[index], strata_b[index], value)
        )

    # 8.6 parameters and observations
    params_a, obs_a = context_a
    params_b, obs_b = context_b
    keys = sorted(set(params_a) | set(obs_a) | set(params_b) | set(obs_b))
    for key in keys:
        descriptor = merge_descriptors(key, descriptors_a.get(key), descriptors_b.get(key))
        if descriptor.compare in ("ignore", "pair"):
            continue
        if not _relevant(key, descriptor, cell, definition, context_a, context_b):
            continue
        value_a = params_a[key] if key in params_a else obs_a.get(key, MISSING)
        value_b = params_b[key] if key in params_b else obs_b.get(key, MISSING)
        reason = _parameter_reason(key, descriptor, value_a, value_b)
        if reason is not None:
            reasons.append(reason)

    # 8.7 population
    shared, reason = _population(a, b, side_a, side_b, definition, items_a, items_b, per_item_a, per_item_b)
    if reason is not None:
        reasons.append(reason)
    return _finish(reasons, shared)


def _reference_reasons(key: str, descriptor: MergedDescriptor, ref_a: Any, ref_b: Any) -> list:
    label = descriptor.label
    identity_a = ref_a.get("identity") if isinstance(ref_a, dict) else None
    identity_b = ref_b.get("identity") if isinstance(ref_b, dict) else None
    if not isinstance(identity_a, str) or not isinstance(identity_b, str):
        return [Reason("incomparable", "not_recorded", label, f"{label} not recorded", key=key,
                       hint=descriptor.hint, step=STEP_REFERENCES)]
    if identity_a == identity_b:
        return []
    view_a, view_b = ref_a.get("view"), ref_b.get("view")
    if (
        isinstance(view_a, dict)
        and isinstance(view_b, dict)
        and isinstance(view_a.get("source"), str)
        and view_a.get("source") == view_b.get("source")
    ):
        frame = view_a["source"]
        d = measure("region", {"frame": frame, "box": view_a.get("region")},
                    {"frame": frame, "box": view_b.get("region")})
        if d is not None and d > 0:
            return [Reason("comparable", "domain", label,
                           phrase("region", f"{label} region", None, None, d), key=key,
                           measure={"kind": "region", "d": d}, step=STEP_REFERENCES)]
    return [Reason("incomparable", "reference", label, f"{label} differs", key=key,
                   hint=descriptor.hint, step=STEP_REFERENCES)]


def _stratum_reasons(a: Mapping, b: Mapping, axis_a: str, axis_b: str, value: str) -> list:
    sid_a = _record(_record(a.get("axes")).get(axis_a)).get("stratification")
    sid_b = _record(_record(b.get("axes")).get(axis_b)).get("stratification")
    strat_a = _record(a.get("stratifications")).get(sid_a) if isinstance(sid_a, str) else None
    strat_b = _record(b.get("stratifications")).get(sid_b) if isinstance(sid_b, str) else None
    label = None
    for strat in (strat_a, strat_b):
        if isinstance(strat, dict) and isinstance(strat.get("label"), str):
            label = strat["label"]
            break
    label = label or axis_a
    if not isinstance(strat_a, dict) or not isinstance(strat_b, dict):
        return [Reason("incomparable", "not_recorded", label, "bins not recorded", step=STEP_STRATIFICATION)]
    if json_equal(identity_projection(strat_a), identity_projection(strat_b)):
        return []
    return [Reason("incomparable", "stratification", label, _bins_text(strat_a, strat_b, value),
                   step=STEP_STRATIFICATION)]


def _bins_text(strat_a: dict, strat_b: dict, value: str) -> str:
    """The ``bins differ`` phrase of §8.3."""
    range_a, range_b = bin_range(strat_a, value), bin_range(strat_b, value)
    if range_a is None or range_b is None:
        return f"bins differ ({value})"
    text_a = _with_unit(f"{_edge(range_a[0], True)}–{_edge(range_a[1], False)}", strat_a)
    text_b = _with_unit(f"{_edge(range_b[0], True)}–{_edge(range_b[1], False)}", strat_b)
    if text_a != text_b:
        return f"bins differ ({value} {text_a} vs {text_b})"
    edges_a = _with_unit(_edges_text(strat_a), strat_a)
    edges_b = _with_unit(_edges_text(strat_b), strat_b)
    if edges_a != edges_b:
        return f"bins differ ({value}: edges {edges_a} vs {edges_b})"
    return f"bins differ ({value})"


def _edge(value: Any, low: bool) -> str:
    """An edge in phrases: 3 significant digits; an open end is −∞ / ∞."""
    if value is None:
        return "−∞" if low else "∞"
    return format_sig3(value)


def _edges_text(stratification: dict) -> str:
    edges = stratification["edges"]
    return "[" + ", ".join(_edge(edge, index == 0) for index, edge in enumerate(edges)) + "]"


def _with_unit(text: str, stratification: dict) -> str:
    unit = stratification.get("unit")
    if isinstance(unit, str) and unit:
        return f"{text} {unit}"
    return text


def _parameter_reason(key: str, descriptor: MergedDescriptor, value_a: Any, value_b: Any) -> Optional[Reason]:
    label = descriptor.label
    if value_a is MISSING:
        value_a = descriptor.default
    if value_b is MISSING:
        value_b = descriptor.default
    if value_a is MISSING or value_b is MISSING:
        return Reason("incomparable", "not_recorded", label, f"{label} not recorded", key=key,
                      hint=descriptor.hint, step=STEP_PARAMETERS)
    if json_equal(value_a, value_b):
        return None
    if descriptor.compare == "measure":
        d = measure(descriptor.observe, value_a, value_b)
        if d is not None:
            if d <= 0:
                return None
            return Reason("comparable", "domain", label,
                          phrase(descriptor.observe, label, value_a, value_b, d), key=key,
                          measure={"kind": descriptor.observe, "d": d}, step=STEP_PARAMETERS)
    return Reason("incomparable", "parameter", label, f"{label} differs", key=key,
                  hint=descriptor.hint, step=STEP_PARAMETERS)


def _population(
    a: Mapping, b: Mapping, side_a: _Side, side_b: _Side, definition: dict,
    items_a: Optional[Collection[str]], items_b: Optional[Collection[str]],
    per_item_a: Optional[Collection[str]], per_item_b: Optional[Collection[str]],
) -> tuple:
    manifest_a = _record(a.get("population")).get("manifest")
    manifest_b = _record(b.get("population")).get("manifest")
    if isinstance(manifest_a, str) and manifest_a and manifest_a == manifest_b:
        return None, None
    if items_a is None or items_b is None:
        return None, Reason("incomparable", "population_unknown", "files", "evaluated files not recorded",
                            step=STEP_POPULATION)
    set_a, set_b = set(items_a), set(items_b)
    if set_a == set_b:
        return None, None
    shared = len(set_a & set_b)
    if shared == 0:
        return 0, Reason("incomparable", "population_disjoint", "files", "no shared files", step=STEP_POPULATION)
    if _pairable(a, b, side_a, side_b, definition, per_item_a, per_item_b):
        return shared, Reason("paired", "population_paired", "files", f"compared on {shared} shared files",
                              step=STEP_POPULATION)
    total = max(len(set_a), len(set_b))
    return shared, Reason("comparable", "population_partial", "files",
                          f"computed on different files: {shared} of {total} shared", step=STEP_POPULATION)


def _recompute_op(contract: Mapping, side: _Side, definition: Mapping) -> Optional[str]:
    """The recompute op of the cell's reducer: its reduction-axis value's, else
    the metric definition's implicit ``reducer`` (§4.9)."""
    value = side.cell.value("reduction")
    if value is None:
        value = definition.get("reducer")
        if not isinstance(value, str):
            return None
    reducer = _record(contract.get("reducers")).get(value)
    op = reducer.get("recompute") if isinstance(reducer, dict) else None
    return op if op in ("mean", "median", "pool") else None


def _pairable(
    a: Mapping, b: Mapping, side_a: _Side, side_b: _Side, definition: dict,
    per_item_a: Optional[Collection[str]], per_item_b: Optional[Collection[str]],
) -> bool:
    op = _recompute_op(a, side_a, definition)
    if op is None or op != _recompute_op(b, side_b, definition):
        return False
    if per_item_a is None or per_item_b is None:
        return False
    name_a, name_b = _per_item(side_a, side_a.cell.base), _per_item(side_b, side_b.cell.base)
    if name_a not in per_item_a or name_b not in per_item_b:
        return False
    if op == "pool":
        pool = definition.get("pool")
        if not isinstance(pool, dict) or pool.get("form") not in ("mean", "rms"):
            return False
        weight = pool.get("weight")
        if not isinstance(weight, str) or not weight:
            return False
        if _per_item(side_a, weight) not in per_item_a or _per_item(side_b, weight) not in per_item_b:
            return False
    return True


def _unsupported(a: Mapping, b: Mapping) -> Verdict:
    params_a, params_b = _record(a.get("parameters")), _record(b.get("parameters"))
    versions = [v for v in (a.get("schemaVersion"), b.get("schemaVersion"))
                if schema_major(v) not in SUPPORTED_MAJORS]
    shown = versions[0] if versions and isinstance(versions[0], str) else "?"
    reasons = [Reason("same", "schema", "schema",
                      f"compared by parameters only (schema {shown} not supported)", step=STEP_SCHEMA)]
    for key in sorted(set(params_a) | set(params_b)):
        if key not in params_a or key not in params_b:
            reasons.append(Reason("incomparable", "schema", key, f"{key} not recorded", key=key, step=STEP_SCHEMA))
        elif not json_equal(params_a[key], params_b[key]):
            reasons.append(Reason("incomparable", "schema", key, f"{key} differs", key=key, step=STEP_SCHEMA))
    return _finish(reasons, None)


# -- convenience over loaded files -----------------------------------------------------


def compare_files(file_a: EvalFile, file_b: EvalFile, metric_name: str) -> Verdict:
    """:func:`compare` with item sets and per-item names taken from the files."""
    items_a, names_a = file_evidence(file_a)
    items_b, names_b = file_evidence(file_b)
    return compare(file_a.contract, file_b.contract, metric_name, items_a, items_b, names_a, names_b)


# -- recompute on shared items (3.7, 3.8) ------------------------------------------------


def recompute(
    op: str,
    values: Mapping[str, Any],
    items: Optional[Collection[str]] = None,
    weights: Optional[Mapping[str, Any]] = None,
    form: Optional[str] = None,
) -> Optional[float]:
    """Recompute an aggregate from per-item values on *items* (all when None).

    ``op`` is a reducer's ``recompute``; for ``pool``, ``form`` is the metric's
    ``pool.form`` and ``weights`` its weight series. Nulls (and, for pools,
    null or zero weights) are skipped; returns ``None`` when nothing is left.
    """
    keys = [key for key in values if items is None or key in items]
    keys.sort()
    pairs = []
    for key in keys:
        value = values[key]
        if not is_number(value) or not math.isfinite(float(value)):
            continue
        if op == "pool":
            weight = (weights or {}).get(key)
            if not is_number(weight) or not math.isfinite(float(weight)) or float(weight) == 0:
                continue
            pairs.append((float(value), float(weight)))
        else:
            pairs.append((float(value), 1.0))
    if not pairs:
        return None
    if op == "mean":
        total = 0.0
        for value, _ in pairs:
            total += value
        return total / len(pairs)
    if op == "median":
        ordered = sorted(value for value, _ in pairs)
        middle = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[middle]
        return (ordered[middle - 1] + ordered[middle]) / 2
    if op == "pool" and form in ("mean", "rms"):
        numerator = 0.0
        denominator = 0.0
        for value, weight in pairs:
            numerator += weight * (value * value if form == "rms" else value)
            denominator += weight
        result = numerator / denominator
        return math.sqrt(result) if form == "rms" else result
    return None


def recompute_cell(
    contract: Mapping[str, Any],
    eval_file: EvalFile,
    metric_name: str,
    items: Optional[Collection[str]] = None,
) -> Optional[float]:
    """Recompute aggregate *metric_name* of a loaded file on *items*."""
    side = _decompose_cell(contract, metric_name)
    if side is None:
        return None
    definition = _definition(contract, side.cell.base) or {}
    op = _recompute_op(contract, side, definition)
    if op is None:
        return None
    values = eval_file.per_item_values(_per_item(side, side.cell.base))
    weights = None
    form = None
    if op == "pool":
        pool = definition.get("pool") if isinstance(definition.get("pool"), dict) else {}
        form = pool.get("form")
        weight = pool.get("weight")
        if not isinstance(weight, str):
            return None
        weights = eval_file.per_item_values(_per_item(side, weight))
    return recompute(op, values, items, weights, form)
