"""Training runs: naming schema 2.0, validation sets and passes (``docs/training.md``).

Producers check what they write with the ``*_problems`` functions; consumers
resolve a run's metric definitions with :func:`training_metric_definition` and
judge a validation pass like any evaluation: :func:`load_validation_pass` turns
a set document and a pass record into an :class:`EvalFile` for
:func:`~euler_metric_naming.contract.compare_files` and
:func:`~euler_metric_naming.contract.recompute_cell`. Nothing here knows a
modality, a metric or a key.

Every problem is a :class:`Problem` with a ``code`` and a dotted ``path`` from
the document root (namespace keys and metric names count as one segment);
both implementations report the same codes and paths in the same order.
"""
from __future__ import annotations

import copy
import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from ..axes import decompose
from ._json import is_number
from .closed import AXIS_ROLES
from .evalfile import EvalFile
from .hashing import manifest_hash
from .legacy import axis_declarations
from .normalize import normalize, schema_major
from .validate import metric_problems, parameter_value_problem
from .vocabulary import available_vocabularies, load_vocabulary

__all__ = [
    "RESERVED_RECORD_FIELDS",
    "TRAINING_SCHEMA_VERSION",
    "VALIDATION_SET_KIND",
    "VALIDATION_SET_SCHEMA_VERSION",
    "Problem",
    "evaluation_metric_ids",
    "is_training_naming_v2",
    "load_validation_pass",
    "metric_value_problem",
    "training_contract",
    "training_metric_definition",
    "training_metric_problem",
    "training_naming_problems",
    "validation_pass_contract",
    "validation_pass_ids",
    "validation_pass_problems",
    "validation_set_problems",
]

#: ``metric_naming.schema_version`` of a training naming 2.0 declaration.
TRAINING_SCHEMA_VERSION = "2.0"
#: ``schemaVersion`` of a validation set document.
VALIDATION_SET_SCHEMA_VERSION = "1.0"
#: ``kind`` of a validation set document.
VALIDATION_SET_KIND = "euler.validation_set"
#: Fields of a ``train.jsonl`` / ``val.jsonl`` record that are not metrics.
RESERVED_RECORD_FIELDS = ("step", "epoch", "wall_time", "elapsed_sec")

_VERSION_RE = re.compile(r"^(\d+)\.(\d+)$")
_NAMESPACE_RE = re.compile(r"^[a-z0-9]+\.train$")
_SEGMENT_RE = re.compile(r"^[a-z0-9_]+$")
_BASE_RE = re.compile(r"^[a-z0-9_]+(\.[a-z0-9_]+)*$")
_KEY_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_SINGLE_ROLES = ("space", "reduction")


@dataclass(frozen=True)
class Problem:
    """One problem: a stable ``code``, where it is (``path``) and what it is."""

    code: str
    path: str
    message: str

    def to_dict(self) -> dict:
        return {"code": self.code, "path": self.path, "message": self.message}


def _path(*parts: str) -> str:
    return ".".join(part for part in parts if part != "")


def _major(version: Any) -> Optional[int]:
    if not isinstance(version, str):
        return None
    match = _VERSION_RE.match(version)
    return int(match.group(1)) if match else None


def _is_int(value: Any) -> bool:
    return is_number(value) and float(value).is_integer()


def _finite_number(value: Any) -> bool:
    return is_number(value) and math.isfinite(float(value))


def _unique_strings(value: Any) -> bool:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        return False
    return len({unicodedata.normalize("NFC", item) for item in value}) == len(value)


# -- training naming 2.0 ------------------------------------------------------------------


def evaluation_metric_ids() -> frozenset:
    """Ids of the metrics of every shipped evaluation vocabulary.

    A vocabulary is an evaluation vocabulary unless its document says
    ``"context": "train"``; a training definition never carries one of these
    ids (``evaluation_id``).
    """
    ids = set()
    for vocabulary_id in available_vocabularies():
        document = load_vocabulary(vocabulary_id).document
        if document.get("context") == "train":
            continue
        for term in (document.get("metrics") or {}).values():
            if isinstance(term, dict) and isinstance(term.get("id"), str):
                ids.add(term["id"])
    return frozenset(ids)


def is_training_naming_v2(naming: Any) -> bool:
    """Whether a ``metric_naming`` declaration claims training naming 2.x."""
    return isinstance(naming, Mapping) and _major(naming.get("schema_version")) == 2


def _axes_problem(axes: Any) -> Optional[str]:
    if not isinstance(axes, Mapping):
        return "axes must be an object of axis declarations"
    positions = set()
    for name in sorted(axes):
        declaration = axes[name]
        if not isinstance(name, str) or not _SEGMENT_RE.match(name):
            return f"axis name {name!r} must match [a-z0-9_]+"
        if not isinstance(declaration, Mapping):
            return f"axis {name!r} must be an object"
        position = declaration.get("position")
        if not _is_int(position) or position < 0:
            return f"axis {name!r} needs an integer position >= 0"
        if int(position) in positions:
            return f"axis {name!r} shares its position with another axis"
        positions.add(int(position))
        values = declaration.get("values")
        if not (
            isinstance(values, list)
            and values
            and all(isinstance(value, str) and _SEGMENT_RE.match(value) for value in values)
        ):
            return f"axis {name!r} needs a non-empty list of [a-z0-9_]+ values"
        if "optional" in declaration and not isinstance(declaration["optional"], bool):
            return f"axis {name!r}: optional must be a boolean"
    return None


def training_naming_problems(naming: Any) -> list:
    """Problems of a training naming 2.0 declaration (``docs/training.md`` §2.7)."""
    if not isinstance(naming, Mapping):
        return [Problem("not_object", "", "the declaration must be an object")]
    problems: list = []
    if _major(naming.get("schema_version")) != 2:
        problems.append(
            Problem("schema_version", "schema_version", "schema_version must be '2.<minor>'")
        )
    namespaces = naming.get("namespaces")
    if not isinstance(namespaces, Mapping) or not namespaces:
        problems.append(Problem("namespaces", "namespaces", "namespaces must be a non-empty object"))
        return problems
    evaluation_ids = evaluation_metric_ids()
    for key in sorted(namespaces):
        entry = namespaces[key]
        base_path = _path("namespaces", key)
        if not isinstance(key, str) or not _NAMESPACE_RE.match(key):
            problems.append(
                Problem("namespace_key", base_path, f"namespace {key!r} must be '<scope>.train'")
            )
        if not isinstance(entry, Mapping):
            problems.append(Problem("not_object", base_path, "a namespace entry must be an object"))
            continue
        axes = entry.get("axes")
        axes_problem = _axes_problem(axes)
        if axes_problem is not None:
            problems.append(Problem("axes", _path(base_path, "axes"), axes_problem))
        if "axisRoles" in entry:
            problems.extend(_axis_role_problems(entry["axisRoles"], axes, base_path))
        metrics = entry.get("metrics")
        if not isinstance(metrics, Mapping):
            problems.append(
                Problem("metrics", _path(base_path, "metrics"), "metrics must be an object")
            )
            continue
        for base in sorted(metrics):
            definition = metrics[base]
            path = _path(base_path, "metrics", base)
            found = []
            if not isinstance(base, str) or not _BASE_RE.match(base):
                found.append(f"base name {base!r} must be [a-z0-9_]+ segments")
            found.extend(metric_problems(definition, path))
            if found:
                problems.append(Problem("definition_invalid", path, "; ".join(found)))
                continue
            if definition.get("id") in evaluation_ids:
                problems.append(
                    Problem(
                        "evaluation_id",
                        path,
                        f"{definition['id']!r} is an evaluation metric; record it in a validation set",
                    )
                )
    return problems


def _axis_role_problems(roles: Any, axes: Any, base_path: str) -> list:
    path = _path(base_path, "axisRoles")
    if not isinstance(roles, Mapping):
        return [Problem("not_object", path, "axisRoles must be an object")]
    declared = set(axes) if isinstance(axes, Mapping) else set()
    problems = []
    counts: dict = {}
    for name in sorted(roles):
        entry = roles[name]
        role = entry.get("role") if isinstance(entry, Mapping) else None
        if name not in declared:
            problems.append(Problem("axis_role", _path(path, name), f"axis {name!r} is not declared"))
        elif role not in AXIS_ROLES or role == "stratum":
            problems.append(
                Problem(
                    "axis_role",
                    _path(path, name),
                    f"role must be one of {', '.join(r for r in AXIS_ROLES if r != 'stratum')}",
                )
            )
        else:
            counts[role] = counts.get(role, 0) + 1
            if role in _SINGLE_ROLES and counts[role] > 1:
                problems.append(
                    Problem("axis_role", _path(path, name), f"at most one axis has the role {role!r}")
                )
    return problems


def _namespace_entry(naming: Any, name: str) -> tuple:
    """``(namespace, entry)`` of a metric name's first two segments, or ``(None, None)``."""
    namespaces = naming.get("namespaces") if isinstance(naming, Mapping) else None
    segments = name.split(".")
    if not isinstance(namespaces, Mapping) or len(segments) < 2:
        return None, None
    namespace = ".".join(segments[:2])
    entry = namespaces.get(namespace)
    return (namespace, entry) if isinstance(entry, Mapping) else (None, None)


def training_metric_problem(naming: Any, name: str) -> Optional[Problem]:
    """The problem of one ``train.jsonl`` / ``val.jsonl`` key under a 2.0 declaration.

    ``None`` when the key is a reserved record field or is declared:
    it has a namespace, decomposes under its axes and its base has a
    definition (``docs/training.md`` §2.5). The value is checked by
    :func:`metric_value_problem`.
    """
    if name in RESERVED_RECORD_FIELDS:
        return None
    segments = name.split(".")
    namespace, entry = _namespace_entry(naming, name)
    if entry is None:
        if len(segments) >= 2 and segments[1] == "eval":
            return Problem(
                "evaluation_name", name, "evaluation names belong in a validation set"
            )
        return Problem("no_namespace", name, "no declared namespace prefixes this key")
    declarations = axis_declarations(entry.get("axes"))
    try:
        if declarations is None:
            raise ValueError("malformed axes")
        base = decompose(name, namespace, declarations).metric
    except (ValueError, TypeError) as exc:
        return Problem("no_decomposition", name, f"does not decompose under {namespace!r}: {exc}")
    metrics = entry.get("metrics")
    if not (isinstance(metrics, Mapping) and isinstance(metrics.get(base), Mapping)):
        return Problem("no_definition", name, f"base metric {base!r} has no definition")
    return None


def metric_value_problem(value: Any) -> Optional[Problem]:
    """``not_a_number`` unless *value* is a finite JSON number (booleans are not)."""
    if _finite_number(value):
        return None
    return Problem("not_a_number", "", "values must be finite numbers")


def training_metric_definition(naming: Any, name: str) -> Optional[dict]:
    """The definition of a training metric name under its run's declaration, or ``None``."""
    namespace, entry = _namespace_entry(naming, name)
    if entry is None:
        return None
    declarations = axis_declarations(entry.get("axes"))
    if declarations is None:
        return None
    try:
        base = decompose(name, namespace, declarations).metric
    except (ValueError, TypeError):
        return None
    metrics = entry.get("metrics")
    definition = metrics.get(base) if isinstance(metrics, Mapping) else None
    return copy.deepcopy(definition) if isinstance(definition, Mapping) else None


def training_contract(naming: Any, namespace: str) -> Optional[dict]:
    """A contract-shaped view of one training namespace, for definition readers.

    ``{"metricSet": {"metricNamespace", "axes"}, "axes": <roles>, "metrics": …}``:
    what readers of contract metric definitions need to decompose a name and
    find its definition. It is not a protocol and is never judged.
    """
    namespaces = naming.get("namespaces") if isinstance(naming, Mapping) else None
    entry = namespaces.get(namespace) if isinstance(namespaces, Mapping) else None
    if not isinstance(entry, Mapping):
        return None
    axes = entry.get("axes")
    roles = entry.get("axisRoles")
    metrics = entry.get("metrics")
    return {
        "metricSet": {
            "metricNamespace": namespace,
            "axes": copy.deepcopy(axes) if isinstance(axes, Mapping) else {},
        },
        "axes": copy.deepcopy(roles) if isinstance(roles, Mapping) else {},
        "metrics": copy.deepcopy(metrics) if isinstance(metrics, Mapping) else {},
    }


# -- validation sets and passes -----------------------------------------------------------


def _record(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def validation_set_problems(document: Any) -> list:
    """Problems of a validation set document (``docs/training.md`` §3.5).

    Only what is specific to validation sets; the ``protocol`` block itself is
    checked by :func:`~euler_metric_naming.contract.check_protocol`.
    """
    if not isinstance(document, Mapping):
        return [Problem("not_object", "", "a validation set document must be an object")]
    problems = []
    if document.get("kind") != VALIDATION_SET_KIND:
        problems.append(Problem("kind", "kind", f"kind must be {VALIDATION_SET_KIND!r}"))
    if _major(document.get("schemaVersion")) != 1:
        problems.append(Problem("schema_version", "schemaVersion", "schemaVersion must be '1.<minor>'"))
    key = document.get("key")
    if not (isinstance(key, str) and _KEY_RE.match(key)):
        problems.append(Problem("key", "key", "key must match [a-z0-9][a-z0-9_-]{0,63}"))
    metric_set = document.get("metricSet")
    namespace = metric_set.get("metricNamespace") if isinstance(metric_set, Mapping) else None
    axes = metric_set.get("axes", {}) if isinstance(metric_set, Mapping) else None
    if not (isinstance(namespace, str) and namespace) or axis_declarations(axes) is None:
        problems.append(
            Problem("metric_set", "metricSet", "metricSet needs a metricNamespace and valid axes")
        )
    protocol = document.get("protocol")
    if not isinstance(protocol, Mapping) or schema_major(protocol.get("schemaVersion")) != 2:
        problems.append(Problem("protocol", "protocol", "protocol must be a schema-2 protocol block"))
        return problems
    items = protocol.get("items")
    if items is not None:
        ids = items.get("ids") if isinstance(items, Mapping) else None
        series = items.get("series") if isinstance(items, Mapping) else None
        if not isinstance(items, Mapping) or (ids is not None and not _unique_strings(ids)):
            problems.append(
                Problem("items", "protocol.items.ids", "items.ids must be unique strings")
            )
        elif series:
            problems.append(
                Problem("items", "protocol.items.series", "per-item series belong to passes")
            )
    return problems


def _set_parts(document: Any) -> tuple:
    """``(namespace, declarations, protocol)`` of a set document (lenient)."""
    document = document if isinstance(document, Mapping) else {}
    metric_set = document.get("metricSet") if isinstance(document.get("metricSet"), Mapping) else {}
    namespace = metric_set.get("metricNamespace")
    declarations = axis_declarations(metric_set.get("axes", {}))
    protocol = document.get("protocol") if isinstance(document.get("protocol"), Mapping) else {}
    return (namespace if isinstance(namespace, str) else None), declarations, protocol


def _name_problem(name: str, path: str, namespace, declarations, definitions) -> Optional[Problem]:
    try:
        if namespace is None or declarations is None:
            raise ValueError("the set has no usable metricSet")
        base = decompose(name, namespace, declarations).metric
    except (ValueError, TypeError) as exc:
        return Problem("no_decomposition", path, f"does not decompose: {exc}")
    if not isinstance(definitions.get(base), Mapping):
        return Problem("no_definition", path, f"base metric {base!r} has no definition in the set")
    return None


def validation_pass_ids(document: Any, record: Any) -> Optional[list]:
    """The item ids of a pass: its ``population.ids``, else the set's ``items.ids``."""
    population = record.get("population") if isinstance(record, Mapping) else None
    if isinstance(population, Mapping) and "ids" in population:
        ids = population.get("ids")
        return list(ids) if _unique_strings(ids) else None
    _, _, protocol = _set_parts(document)
    items = protocol.get("items")
    ids = items.get("ids") if isinstance(items, Mapping) else None
    return list(ids) if _unique_strings(ids) else None


def validation_pass_problems(document: Any, record: Any) -> list:
    """Problems of a pass record against its set document (``docs/training.md`` §3.5)."""
    if not isinstance(record, Mapping):
        return [Problem("not_object", "", "a pass record must be an object")]
    problems = []
    step = record.get("step")
    if not (_is_int(step) and step >= 0):
        problems.append(Problem("step", "step", "step must be an integer >= 0"))
    if "epoch" in record and not _is_int(record["epoch"]):
        problems.append(Problem("step", "epoch", "epoch must be an integer"))
    namespace, declarations, protocol = _set_parts(document)
    definitions = _record(protocol.get("metrics"))
    descriptors = _record(protocol.get("descriptors"))
    metrics = record.get("metrics")
    if not isinstance(metrics, Mapping) or not metrics:
        problems.append(Problem("metrics", "metrics", "metrics must be a non-empty object"))
    else:
        for name in sorted(metrics):
            path = _path("metrics", name)
            if not _finite_number(metrics[name]):
                problems.append(Problem("not_a_number", path, "values must be finite numbers"))
            found = _name_problem(name, path, namespace, declarations, definitions)
            if found is not None:
                problems.append(found)
    for block in ("parameters", "observations"):
        if block not in record:
            continue
        values = record[block]
        if not isinstance(values, Mapping):
            problems.append(Problem("not_object", block, f"{block} must be an object"))
            continue
        for key in sorted(values):
            path = _path(block, key)
            if not isinstance(descriptors.get(key), Mapping):
                problems.append(Problem("undescribed", path, f"{key!r} has no descriptor in the set"))
            if block == "parameters":
                problem = parameter_value_problem(values[key])
                if problem:
                    problems.append(Problem("parameter_value", path, f"{key!r} {problem}"))
    if "population" in record:
        population = record["population"]
        if not isinstance(population, Mapping):
            problems.append(Problem("not_object", "population", "population must be an object"))
        elif not _unique_strings(population.get("ids")):
            problems.append(Problem("population", "population.ids", "ids must be unique strings"))
    ids = validation_pass_ids(document, record)
    for block in ("per_item", "series"):
        if block not in record:
            continue
        values = record[block]
        if not isinstance(values, Mapping):
            problems.append(Problem("not_object", block, f"{block} must be an object"))
            continue
        for name in sorted(values):
            path = _path(block, name)
            column = values[name]
            if ids is None or not isinstance(column, list) or len(column) != len(ids):
                problems.append(Problem("length", path, "one value per item id is required"))
            found = _name_problem(name, path, namespace, declarations, definitions)
            if found is not None:
                problems.append(found)
    return problems


def validation_pass_contract(document: Any, record: Any) -> dict:
    """The normalized contract of the evaluation a pass stands for (§3.4)."""
    _, _, protocol = _set_parts(document)
    record = record if isinstance(record, Mapping) else {}
    merged = copy.deepcopy(dict(protocol))
    for block in ("parameters", "observations"):
        overlay = record.get(block)
        base = merged.get(block) if isinstance(merged.get(block), dict) else {}
        if isinstance(overlay, Mapping):
            base = {**base, **copy.deepcopy(dict(overlay))}
        merged[block] = base
    items = merged.get("items") if isinstance(merged.get("items"), dict) else {}
    items = {key: value for key, value in items.items() if key != "series"}
    population = record.get("population")
    if isinstance(population, Mapping) and "ids" in population:
        ids = population.get("ids")
        previous = merged.get("population") if isinstance(merged.get("population"), dict) else {}
        updated = {
            key: copy.deepcopy(value)
            for key, value in previous.items()
            if key in ("unit", "idScheme")
        }
        if _unique_strings(ids):
            updated["count"] = len({unicodedata.normalize("NFC", item) for item in ids})
            updated["manifest"] = manifest_hash(ids)
            items["ids"] = list(ids)
        else:
            items.pop("ids", None)
        for key in ("count", "manifest", "accounting"):
            if key in population:
                updated[key] = copy.deepcopy(population[key])
        merged["population"] = updated
    series = record.get("series")
    items["series"] = (
        {name: list(values) for name, values in series.items() if isinstance(values, list)}
        if isinstance(series, Mapping)
        else {}
    )
    merged["items"] = items
    metric_set = document.get("metricSet") if isinstance(document, Mapping) else None
    return normalize(merged, metric_set if isinstance(metric_set, Mapping) else None)


def load_validation_pass(document: Any, record: Any) -> EvalFile:
    """The evaluation a pass stands for, as an :class:`EvalFile` (§3.4).

    Its aggregates are the pass's ``metrics``, its per-item values the pass's
    ``per_item`` lists (aligned with the pass's ids), its item set the pass's
    ids. :func:`~euler_metric_naming.contract.compare_files` and
    :func:`~euler_metric_naming.contract.recompute_cell` take it like any
    loaded results file.
    """
    record = record if isinstance(record, Mapping) else {}
    contract = validation_pass_contract(document, record)
    ids = validation_pass_ids(document, record)
    metrics = {
        name: value
        for name, value in _record(record.get("metrics")).items()
        if isinstance(name, str) and _finite_number(value)
    }
    per_file: dict = {}
    if ids is not None:
        for item in ids:
            per_file[item] = {}
        for name, column in _record(record.get("per_item")).items():
            if not isinstance(column, list) or len(column) != len(ids):
                continue
            for item, value in zip(ids, column):
                if _finite_number(value):
                    per_file[item][name] = value
    metric_set = document.get("metricSet") if isinstance(document, Mapping) else None
    metric_set = dict(metric_set) if isinstance(metric_set, Mapping) else {}
    series = contract["items"].get("series") if isinstance(contract.get("items"), dict) else {}
    axes = axis_declarations(contract["metricSet"]["axes"]) or {}
    return EvalFile(
        raw=dict(record),
        namespace=contract["metricSet"]["metricNamespace"],
        axes=axes,
        metric_set=metric_set,
        protocol={key: value for key, value in contract.items() if key != "metricSet"},
        contract=contract,
        metrics=metrics,
        per_file=per_file,
        file_ids=list(per_file),
        series={name: values for name, values in (series or {}).items() if isinstance(values, list)},
        item_ids=list(ids) if ids is not None else None,
    )
