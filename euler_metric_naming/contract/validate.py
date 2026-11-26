"""Dependency-free structural validation of a v2 ``protocol`` block.

Producers validate what they write; consumers stay lenient (unknown fields are
ignored, unknown closed-set values compare strictly). :func:`protocol_problems`
returns every problem found; :func:`check_protocol` raises on the first batch.
"""
from __future__ import annotations

import re
from typing import Any

from ._json import is_number
from .closed import (
    AXIS_ROLES,
    COMPARE_RULES,
    DIRECTIONS,
    DISPLAY_SCALES,
    MAX_KEY_LENGTH,
    MAX_LIST_LENGTH,
    MAX_PARAMETERS,
    MAX_STRING_LENGTH,
    METRIC_KINDS,
    OBSERVE_KINDS,
    POOL_FORMS,
    RECOMPUTE_OPS,
    ROLES,
)
from .conditions import condition_problems
from .hashing import valid_rev
from .strata import valid_edges

_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_VERSION_RE = re.compile(r"^(\d+)\.(\d+)$")
_SCHEME_RE = re.compile(r"^[^/\s]+/[0-9]+$")


class ProtocolValidationError(ValueError):
    """A protocol block that violates the contract; ``problems`` lists why."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = list(problems)
        super().__init__("invalid protocol: " + "; ".join(self.problems))


def parameter_value_problem(value: Any) -> str | None:
    """The v1-compatible shape of a parameter value: a scalar or ≤64 scalars."""
    if isinstance(value, (list, tuple)):
        if len(value) > MAX_LIST_LENGTH:
            return f"must have at most {MAX_LIST_LENGTH} items"
        items = list(value)
    else:
        items = [value]
    for item in items:
        if item is None or isinstance(item, bool):
            continue
        if is_number(item):
            if item != item or item in (float("inf"), float("-inf")):
                return "numbers must be finite"
            continue
        if isinstance(item, str):
            if len(item) > MAX_STRING_LENGTH:
                return f"strings must have at most {MAX_STRING_LENGTH} characters"
            continue
        return "must be a JSON scalar or a list of scalars"
    return None


def parameters_problems(parameters: Any, path: str = "parameters") -> list[str]:
    if not isinstance(parameters, dict):
        return [f"{path} must be an object"]
    problems = []
    if len(parameters) > MAX_PARAMETERS:
        problems.append(f"{path} must have at most {MAX_PARAMETERS} keys")
    for key, value in parameters.items():
        if not isinstance(key, str) or not key.strip() or len(key) > MAX_KEY_LENGTH:
            problems.append(f"{path} keys must have 1 to {MAX_KEY_LENGTH} characters")
            continue
        problem = parameter_value_problem(value)
        if problem:
            problems.append(f"{path}.{key} {problem}")
    return problems


def descriptor_problems(descriptor: Any, path: str) -> list[str]:
    if not isinstance(descriptor, dict):
        return [f"{path} must be an object"]
    problems = []
    if descriptor.get("role") not in ROLES:
        problems.append(f"{path}.role must be one of {', '.join(ROLES)}")
    if "compare" in descriptor and descriptor["compare"] not in COMPARE_RULES:
        problems.append(f"{path}.compare must be one of {', '.join(COMPARE_RULES)}")
    if "observe" in descriptor and descriptor["observe"] not in OBSERVE_KINDS:
        problems.append(f"{path}.observe must be one of {', '.join(OBSERVE_KINDS)}")
    compare = descriptor.get("compare")
    if compare is None and descriptor.get("role") == "domain":
        compare = "measure"
    if compare == "measure" and descriptor.get("observe") not in OBSERVE_KINDS:
        problems.append(f"{path}.observe is required for compare 'measure'")
    for field in ("label", "hint", "description", "vocabulary"):
        if field in descriptor and not isinstance(descriptor[field], str):
            problems.append(f"{path}.{field} must be a string")
    if "unit" in descriptor and not (descriptor["unit"] is None or isinstance(descriptor["unit"], str)):
        problems.append(f"{path}.unit must be a string or null")
    for field in ("appliesWhen", "inertIf"):
        if field in descriptor:
            problems.extend(condition_problems(descriptor[field], f"{path}.{field}"))
    if "affects" in descriptor:
        problems.extend(_affects_problems(descriptor["affects"], f"{path}.affects"))
    if "derivedFrom" in descriptor and not _string_list(descriptor["derivedFrom"]):
        problems.append(f"{path}.derivedFrom must be a list of strings")
    return problems


def _affects_problems(affects: Any, path: str) -> list[str]:
    if not isinstance(affects, dict):
        return [f"{path} must be an object"]
    problems = []
    if "metrics" in affects and not _string_list(affects["metrics"]):
        problems.append(f"{path}.metrics must be a list of globs")
    axes = affects.get("axes")
    if axes is not None:
        if not isinstance(axes, dict):
            problems.append(f"{path}.axes must be an object")
        else:
            for role, values in axes.items():
                if role not in AXIS_ROLES:
                    problems.append(f"{path}.axes.{role} is not an axis role")
                if not _string_list(values):
                    problems.append(f"{path}.axes.{role} must be a list of strings")
    return problems


def metric_problems(definition: Any, path: str) -> list[str]:
    if not isinstance(definition, dict):
        return [f"{path} must be an object"]
    problems = []
    if "kind" in definition and definition["kind"] not in METRIC_KINDS:
        problems.append(f"{path}.kind must be one of {', '.join(METRIC_KINDS)}")
    if "rev" in definition and not valid_rev(definition["rev"]):
        problems.append(f"{path}.rev must be an integer >= 1")
    if "direction" in definition and definition["direction"] not in DIRECTIONS:
        problems.append(f"{path}.direction must be one of {', '.join(DIRECTIONS)}")
    bounds = definition.get("bounds")
    if bounds is not None and not (
        isinstance(bounds, list)
        and len(bounds) == 2
        and all(item is None or is_number(item) for item in bounds)
    ):
        problems.append(f"{path}.bounds must be [min | null, max | null]")
    display = definition.get("display")
    if display is not None:
        if not isinstance(display, dict) or display.get("scale") not in DISPLAY_SCALES:
            problems.append(f"{path}.display.scale must be one of {', '.join(DISPLAY_SCALES)}")
        elif display["scale"] == "absolute":
            full_at = display.get("fullAt")
            if not (is_number(full_at) and full_at > 0):
                problems.append(f"{path}.display.fullAt must be a number > 0")
    if "perItem" in definition and not isinstance(definition["perItem"], bool):
        problems.append(f"{path}.perItem must be a boolean")
    pool = definition.get("pool")
    if pool is not None:
        if not isinstance(pool, dict) or pool.get("form") not in POOL_FORMS:
            problems.append(f"{path}.pool.form must be one of {', '.join(POOL_FORMS)}")
        elif pool["form"] != "none" and not (isinstance(pool.get("weight"), str) and pool["weight"]):
            problems.append(f"{path}.pool.weight is required for form {pool['form']!r}")
    if "reducer" in definition and not (
        isinstance(definition["reducer"], str) and definition["reducer"]
    ):
        problems.append(f"{path}.reducer must name a reducer")
    if "requires" in definition and not _string_list(definition["requires"]):
        problems.append(f"{path}.requires must be a list of reference roles")
    if "dependsOn" in definition and not _string_list(definition["dependsOn"]):
        problems.append(f"{path}.dependsOn must be a list of globs")
    return problems


def reducer_problems(reducer: Any, path: str) -> list[str]:
    if not isinstance(reducer, dict):
        return [f"{path} must be an object"]
    if reducer.get("recompute") not in RECOMPUTE_OPS:
        return [f"{path}.recompute must be one of {', '.join(RECOMPUTE_OPS)}"]
    return []


def stratification_problems(stratification: Any, path: str) -> list[str]:
    if not isinstance(stratification, dict):
        return [f"{path} must be an object"]
    problems = []
    edges = stratification.get("edges")
    categories = stratification.get("categories")
    if edges is not None:
        if not valid_edges(edges):
            problems.append(
                f"{path}.edges must be at least two numbers (the first and last may be null)"
            )
        elif any(
            a is not None and b is not None and b <= a for a, b in zip(edges, edges[1:])
        ):
            problems.append(f"{path}.edges must be strictly ascending")
        else:
            labels = stratification.get("labels")
            if not (_string_list(labels) and len(labels) == len(edges) - 1):
                problems.append(f"{path}.labels must name the {len(edges) - 1} bins")
    elif categories is not None:
        if not _string_list(categories) or not categories:
            problems.append(f"{path}.categories must be a non-empty list of strings")
    else:
        problems.append(f"{path} needs edges with labels, or categories")
    if "total" in stratification and not (
        stratification["total"] is None or isinstance(stratification["total"], str)
    ):
        problems.append(f"{path}.total must be a string")
    return problems


def protocol_problems(protocol: Any) -> list[str]:
    """Every structural problem of a v2 ``protocol`` block."""
    if not isinstance(protocol, dict):
        return ["protocol must be an object"]
    problems: list[str] = []
    version = protocol.get("schemaVersion")
    match = _VERSION_RE.match(version) if isinstance(version, str) else None
    if not match or match.group(1) != "2":
        problems.append("schemaVersion must be '2.<minor>'")
    if "task" in protocol and not (protocol["task"] is None or isinstance(protocol["task"], str)):
        problems.append("task must be a string")
    if "parameters" not in protocol:
        problems.append("parameters is required")
    parameters = protocol.get("parameters", {})
    problems.extend(parameters_problems(parameters))
    parameters = parameters if isinstance(parameters, dict) else {}

    observations = protocol.get("observations", {})
    if not isinstance(observations, dict):
        problems.append("observations must be an object")
        observations = {}
    for key in observations:
        if key in parameters:
            problems.append(f"{key} is both a parameter and an observation")

    descriptors = protocol.get("descriptors", {})
    if not isinstance(descriptors, dict):
        problems.append("descriptors must be an object")
        descriptors = {}
    for key, descriptor in descriptors.items():
        problems.extend(descriptor_problems(descriptor, f"descriptors.{key}"))

    references = protocol.get("references", {})
    if not isinstance(references, dict):
        problems.append("references must be an object")
        references = {}
    for role, reference in references.items():
        path = f"references.{role}"
        if not isinstance(reference, dict):
            problems.append(f"{path} must be an object")
            continue
        if not isinstance(reference.get("identity"), str) or not reference["identity"]:
            problems.append(f"{path}.identity is required")
        scheme = reference.get("identityScheme")
        if scheme is not None and not (isinstance(scheme, str) and _SCHEME_RE.match(scheme)):
            problems.append(f"{path}.identityScheme must have the form <family>/<integer>")
        view = reference.get("view")
        if view is not None:
            region = view.get("region") if isinstance(view, dict) else None
            if not (
                isinstance(view, dict)
                and isinstance(view.get("source"), str)
                and isinstance(region, list)
                and len(region) == 4
                and all(is_number(v) and v >= 0 for v in region)
            ):
                problems.append(f"{path}.view must be {{source, region: [top, left, height, width]}}")

    population = protocol.get("population")
    if population is not None:
        if not isinstance(population, dict):
            problems.append("population must be an object")
        else:
            manifest = population.get("manifest")
            if manifest is not None and not (isinstance(manifest, str) and _HASH_RE.match(manifest)):
                problems.append("population.manifest must be a sha256: hash")
            count = population.get("count")
            if count is not None and not (is_number(count) and count >= 0 and float(count).is_integer()):
                problems.append("population.count must be an integer >= 0")

    stratifications = protocol.get("stratifications", {})
    if not isinstance(stratifications, dict):
        problems.append("stratifications must be an object")
        stratifications = {}
    for sid, stratification in stratifications.items():
        problems.extend(stratification_problems(stratification, f"stratifications.{sid}"))

    reducers = protocol.get("reducers", {})
    if not isinstance(reducers, dict):
        problems.append("reducers must be an object")
        reducers = {}
    for value, reducer in reducers.items():
        problems.extend(reducer_problems(reducer, f"reducers.{value}"))

    axes = protocol.get("axes", {})
    if not isinstance(axes, dict):
        problems.append("axes must be an object")
        axes = {}
    single_roles: dict = {}
    for name, axis in axes.items():
        path = f"axes.{name}"
        if not isinstance(axis, dict) or axis.get("role") not in AXIS_ROLES:
            problems.append(f"{path}.role must be one of {', '.join(AXIS_ROLES)}")
            continue
        role = axis["role"]
        if role in ("space", "reduction"):
            if role in single_roles:
                problems.append(f"axes {single_roles[role]!r} and {name!r} both have role {role!r}")
            single_roles[role] = name
        if role == "stratum":
            sid = axis.get("stratification")
            if not isinstance(sid, str) or sid not in stratifications:
                problems.append(f"{path}.stratification must name a stratification")

    metrics = protocol.get("metrics", {})
    if not isinstance(metrics, dict):
        problems.append("metrics must be an object")
        metrics = {}
    for base, definition in metrics.items():
        problems.extend(metric_problems(definition, f"metrics.{base}"))
        reducer = definition.get("reducer") if isinstance(definition, dict) else None
        if isinstance(reducer, str) and reducer and reducer not in reducers:
            problems.append(f"metrics.{base}.reducer {reducer!r} is not in reducers")

    items = protocol.get("items")
    if items is not None:
        problems.extend(_items_problems(items))
    return problems


def _items_problems(items: Any) -> list[str]:
    if not isinstance(items, dict):
        return ["items must be an object"]
    problems = []
    ids = items.get("ids")
    if not _string_list(ids):
        return ["items.ids must be a list of strings"]
    if len(set(ids)) != len(ids):
        problems.append("items.ids must be unique")
    series = items.get("series", {})
    if not isinstance(series, dict):
        return problems + ["items.series must be an object"]
    for name, values in series.items():
        if not isinstance(values, list) or len(values) != len(ids):
            problems.append(f"items.series.{name} must have one value per id")
        elif not all(value is None or is_number(value) for value in values):
            problems.append(f"items.series.{name} values must be numbers or null")
    return problems


def check_protocol(protocol: Any) -> None:
    """Raise :class:`ProtocolValidationError` when the block has problems."""
    problems = protocol_problems(protocol)
    if problems:
        raise ProtocolValidationError(problems)


# -- vocabulary documents (vocabulary-1.schema.json) ------------------------------------

_VOCABULARY_ID_RE = re.compile(r"^[a-z0-9]+(\.[a-z0-9_]+)+$")
_DESCRIPTOR_KEY_RE = re.compile(r"^[a-z0-9_]+(\.[a-z0-9_]+)+$")
_METRIC_BASE_RE = re.compile(r"^[a-z0-9_]+(\.[a-z0-9_]+)*$")
_REQUIRED_METRIC_FIELDS = ("id", "kind", "rev", "displayName", "direction")
VOCABULARY_CONTEXTS = ("eval", "train")


def vocabulary_problems(document: Any) -> list[str]:
    """Every problem of a vocabulary document (``vocabulary-1.schema.json``).

    Dependency-free; the same rules as the JSON Schema, which the test suite
    checks against ``jsonschema`` for the shipped vocabularies.
    """
    if not isinstance(document, dict):
        return ["a vocabulary must be an object"]
    problems: list[str] = []
    vid = document.get("id")
    if not (isinstance(vid, str) and _VOCABULARY_ID_RE.match(vid)):
        problems.append("id must be a dotted lower-case id such as 'mylab.thermal'")
    version = document.get("version")
    if not (isinstance(version, str) and _VERSION_RE.match(version)):
        problems.append("version must be '<major>.<minor>'")
    if "context" in document and document["context"] not in VOCABULARY_CONTEXTS:
        problems.append(f"context must be one of {', '.join(VOCABULARY_CONTEXTS)}")
    for field in ("title", "description", "$schema"):
        if field in document and not isinstance(document[field], str):
            problems.append(f"{field} must be a string")

    sections = (
        ("descriptors", _DESCRIPTOR_KEY_RE, descriptor_problems, True),
        ("metrics", _METRIC_BASE_RE, metric_problems, True),
        ("reducers", None, reducer_problems, True),
        ("stratificationSchemes", None, _scheme_problems, False),
    )
    for section, key_re, term_problems, required in sections:
        if section not in document:
            if required:
                problems.append(f"{section} is required")
            continue
        terms = document[section]
        if not isinstance(terms, dict):
            problems.append(f"{section} must be an object")
            continue
        for key, term in terms.items():
            path = f"{section}.{key}"
            if key_re is not None and not key_re.match(key):
                problems.append(f"{path}: key does not match {key_re.pattern}")
            problems.extend(term_problems(term, path))
            if not isinstance(term, dict):
                continue
            if "vocabulary" in term and section != "stratificationSchemes":
                problems.append(
                    f"{path} must not carry a vocabulary tag (producers add it)"
                )
            if section == "metrics":
                for field in _REQUIRED_METRIC_FIELDS:
                    if field not in term:
                        problems.append(f"{path}.{field} is required")
                for field in ("id", "displayName", "description"):
                    if field in term and not isinstance(term[field], str):
                        problems.append(f"{path}.{field} must be a string")
                if "unit" in term and not (term["unit"] is None or isinstance(term["unit"], str)):
                    problems.append(f"{path}.unit must be a string or null")
            if section == "reducers":
                for field in ("over", "label", "description"):
                    if field in term and not isinstance(term[field], str):
                        problems.append(f"{path}.{field} must be a string")
    return problems


def _scheme_problems(scheme: Any, path: str) -> list[str]:
    if not isinstance(scheme, dict):
        return [f"{path} must be an object"]
    problems = []
    if not isinstance(scheme.get("label"), str):
        problems.append(f"{path}.label is required (a string)")
    if "description" in scheme and not isinstance(scheme["description"], str):
        problems.append(f"{path}.description must be a string")
    return problems


def _string_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)
