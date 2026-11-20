"""The legacy adapter: protocol schema 1.x to a v2 contract (``docs/contract.md`` §10).

The rules are data in the frozen table ``contract/legacy/protocol-1.json``;
this module only interprets it. The TS artifact ports it line by line.
"""
from __future__ import annotations

import copy
import json
from functools import lru_cache
from importlib import resources
from typing import Any, Mapping, Optional

from ._json import MISSING, is_number, json_equal
from .hashing import hash_value
from .strata import sqrt_equal_width_edges


@lru_cache(maxsize=None)
def _table() -> dict:
    node = resources.files("euler_metric_naming.contract") / "legacy" / "protocol-1.json"
    return json.loads(node.read_text(encoding="utf-8"))


def legacy_table() -> dict:
    """The frozen v1 adapter table (a deep copy)."""
    return copy.deepcopy(_table())


def metric_set_info(metric_set: Any) -> dict:
    """``{metricNamespace, axes}`` of a metricSet envelope (camel or snake case)."""
    if not isinstance(metric_set, Mapping):
        return {"metricNamespace": None, "axes": {}}
    namespace = None
    for key in ("metricNamespace", "metric_namespace", "namespace"):
        value = metric_set.get(key)
        if isinstance(value, str) and value.strip():
            namespace = value.strip()
            break
    axes = metric_set.get("axes")
    return {
        "metricNamespace": namespace,
        "axes": copy.deepcopy(axes) if isinstance(axes, Mapping) else {},
    }


def axis_declarations(raw: Any) -> Optional[dict]:
    """Parse metricSet axis declarations from untrusted JSON; None when malformed."""
    from ..axes import AxisDeclaration

    if not isinstance(raw, Mapping):
        return None
    out = {}
    for name, declaration in raw.items():
        if not isinstance(declaration, Mapping):
            return None
        position = declaration.get("position")
        values = declaration.get("values")
        if not (is_number(position) and float(position).is_integer()):
            return None
        if not (isinstance(values, list) and all(isinstance(v, str) for v in values)):
            return None
        description = declaration.get("description")
        out[name] = AxisDeclaration(
            position=int(position),
            values=tuple(values),
            optional=declaration.get("optional") is True,
            description=description if isinstance(description, str) else None,
        )
    return out


def empty_contract(schema_version: Any) -> dict:
    """A contract with every block present and empty."""
    return {
        "schemaVersion": schema_version,
        "task": None,
        "vocabularies": {},
        "parameters": {},
        "descriptors": {},
        "references": {},
        "observations": {},
        "population": {},
        "stratifications": {},
        "reducers": {},
        "axes": {},
        "metrics": {},
        "items": {},
        "metricSet": {"metricNamespace": None, "axes": {}},
    }


def _dims(meta: Any, side: str) -> Optional[tuple]:
    block = meta.get(side) if isinstance(meta, Mapping) else None
    dims = block.get("dimensions") if isinstance(block, Mapping) else None
    if not isinstance(dims, Mapping):
        return None
    height, width = dims.get("height"), dims.get("width")
    if not (is_number(height) and is_number(width) and height > 0 and width > 0):
        return None
    return (float(height), float(width))


def _ratio(target: Optional[tuple], source: Optional[tuple]) -> Optional[list]:
    if target is None or source is None:
        return None
    return [target[0] / source[0], target[1] / source[1]]


def _benchmark_edges(value: Any, labels: list, boundaries: Any) -> Optional[list]:
    if not (
        isinstance(value, list)
        and len(value) == 2
        and all(is_number(item) for item in value)
        and 0 <= value[0] < value[1]
    ):
        return None
    if isinstance(boundaries, Mapping) and json_equal(boundaries.get("range"), value):
        bins = [boundaries.get(label) for label in labels]
        if all(
            isinstance(b, list) and len(b) == 2 and all(is_number(x) for x in b) for b in bins
        ):
            return [bins[0][0]] + [b[1] for b in bins]
    return sqrt_equal_width_edges(value[0], value[1], len(labels))


def normalize_legacy(
    protocol: Mapping[str, Any],
    metric_set: Optional[Mapping[str, Any]] = None,
    meta: Optional[Mapping[str, Any]] = None,
) -> dict:
    """Normalize a schema-1.x ``protocol`` block into a v2 contract (§10)."""
    table = _table()
    v1 = protocol.get("parameters") if isinstance(protocol, Mapping) else None
    v1 = v1 if isinstance(v1, Mapping) else {}
    contract = empty_contract(protocol.get("schemaVersion") if isinstance(protocol, Mapping) else None)
    contract["adapter"] = f"{table['id']}@{table['version']}"
    contract["metricSet"] = metric_set_info(metric_set)
    metadata = metric_set.get("metadata") if isinstance(metric_set, Mapping) else None
    metadata = metadata if isinstance(metadata, Mapping) else {}
    descriptors = table["descriptors"]
    consumed: set = set()

    def describe(key: str) -> None:
        if key in descriptors:
            contract["descriptors"][key] = copy.deepcopy(descriptors[key])

    def set_param(key: str, value: Any) -> None:
        contract["parameters"][key] = copy.deepcopy(value)
        describe(key)

    # references: reference.<role>.<field>
    spec = table["references"]
    fields = spec["fields"]
    accepted = set(fields) | {spec["modalityField"]}
    roles: dict = {}
    for key, value in v1.items():
        if not key.startswith(spec["prefix"]):
            continue
        rest = key[len(spec["prefix"]):]
        role, dot, field = rest.partition(".")
        if not dot or not role or field not in accepted:
            continue
        roles.setdefault(role, {})[field] = value
        consumed.add(key)
    for role, values in roles.items():
        if all(values.get(field) is None for field in fields):
            continue
        name = spec["roleRenames"].get(role, role)
        modality = values.get(spec["modalityField"])
        identity_source = {spec["modalityField"]: modality}
        for field in fields:
            identity_source[field] = values.get(field)
        reference: dict = {}
        if modality is not None:
            reference["modality"] = modality
        reference["dataset"] = {
            "id": values.get("dataset"),
            "scope": values.get("scope"),
            "split": values.get("split"),
        }
        reference["identity"] = hash_value(identity_source)
        contract["references"][name] = reference
        describe(f"reference.{name}")

    # population
    count_key = table["population"]["count"]
    if count_key in v1:
        consumed.add(count_key)
        if is_number(v1[count_key]):
            contract["population"]["count"] = v1[count_key]

    # plain renames
    for v1_key, key in table["parameters"].items():
        if v1_key in v1:
            consumed.add(v1_key)
            set_param(key, v1[v1_key])
    for v1_key, key in table["observations"].items():
        if v1_key in v1:
            consumed.add(v1_key)
            contract["observations"][key] = copy.deepcopy(v1[v1_key])
            describe(key)

    # whether an alignment was applied, from metricSet.metadata
    applied = metadata.get(table["appliedFrom"], MISSING)
    for v1_key, key in table["applied"].items():
        if v1_key in v1 and isinstance(applied, bool):
            set_param(key, applied)

    # benchmark range: a derived parameter and the GT-depth stratification
    bench = table["benchmark"]
    if bench["key"] in v1:
        consumed.add(bench["key"])
        value = v1[bench["key"]]
        set_param(bench["parameter"], value)
        template = bench["template"]
        boundaries: Any = metadata
        for part in bench["boundariesFrom"].split("."):
            boundaries = boundaries.get(part) if isinstance(boundaries, Mapping) else None
        edges = _benchmark_edges(value, template["labels"], boundaries)
        if edges is not None:
            stratification: dict = {}
            for key, item in template.items():
                stratification[key] = copy.deepcopy(item)
                if key == "unit":
                    stratification["edges"] = edges
            if "edges" not in stratification:
                stratification["edges"] = edges
            contract["stratifications"][bench["stratification"]] = stratification

    # metrics revision: the policies it implies
    revision = table["revision"]
    if revision["key"] in v1:
        consumed.add(revision["key"])
        r = v1[revision["key"]]
        if is_number(r) and float(r).is_integer() and r >= 1:
            for key, steps in revision["policies"].items():
                chosen = steps[0][1]
                for start, value in steps:
                    if r >= start:
                        chosen = value
                set_param(key, chosen)
            if r > revision["lastKnown"]:
                set_param(revision["beyondKey"], r)
        else:
            set_param(revision["beyondKey"], r)

    # resampling factors from meta dimensions
    resample = table["resample"]
    method = v1.get(resample["key"])
    gt_dims, pred_dims = _dims(meta, "gt"), _dims(meta, "pred")
    if method in resample["prediction"]["methods"]:
        key = resample["prediction"]["observation"]
        contract["observations"][key] = _ratio(gt_dims, pred_dims)
        describe(key)
    elif method in resample["gt"]["methods"]:
        set_param(resample["gt"]["parameter"], _ratio(pred_dims, gt_dims))

    # everything else passes through, undescribed (compared strictly)
    for key, value in v1.items():
        if key not in consumed:
            contract["parameters"][key] = copy.deepcopy(value)

    # axis roles and reducers
    for name in contract["metricSet"]["axes"]:
        if name in table["axisRoles"]:
            contract["axes"][name] = copy.deepcopy(table["axisRoles"][name])
    contract["reducers"] = copy.deepcopy(table["reducers"])
    return contract
