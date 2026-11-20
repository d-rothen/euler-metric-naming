"""Load a whole results file (``eval.json``) the way euler-view flattens it.

Metric names are the dotted paths of the metric tree; per-file values live in
``per_file_metrics`` whose item ids are the ``children`` keys from the root
down plus the file entry's ``id``, joined with ``~`` (``euler.item_id/1``).
"""
from __future__ import annotations

import json
import math
import os
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Union

from ..axes import AxisDeclaration
from ._json import is_number
from .closed import ITEM_ID_SEPARATOR
from .hashing import manifest_hash
from .normalize import normalize

#: Top-level keys that are not metric roots (as euler-view's importer).
RESERVED_TOP_LEVEL_KEYS = frozenset(
    {
        "metricSet",
        "metric_set",
        "dataset_info",
        "meta",
        "protocol",
        "per_file_metrics",
        "metric_namespace",
        "metricNamespace",
        "task_key",
        "taskKey",
        "metric_producer_key",
        "metricProducerKey",
        "metric_producer_version",
        "metricProducerVersion",
        "metric_source_kind",
        "metricSourceKind",
        "metric_set_metadata",
        "metricSetMetadata",
    }
)
_METRIC_KEY_RE = re.compile(r"^[a-z0-9_]+$")


@dataclass
class EvalFile:
    """A loaded results file."""

    raw: dict
    namespace: Optional[str]
    axes: dict
    metric_set: dict
    protocol: Optional[dict]
    contract: dict
    metrics: dict = field(default_factory=dict)
    per_file: dict = field(default_factory=dict)
    file_ids: list = field(default_factory=list)
    series: dict = field(default_factory=dict)
    item_ids: Optional[list] = None

    @property
    def metric_names(self) -> list:
        return list(self.metrics)

    @property
    def per_item_names(self) -> set:
        """Names with per-item values: per-file metric names and ``items.series``."""
        names = set(self.series)
        for values in self.per_file.values():
            names.update(values)
        return names

    def per_item_values(self, name: str) -> dict:
        """``{item id: value}`` of a per-item name (per-file metrics, then series)."""
        values = {
            item: metrics[name] for item, metrics in self.per_file.items() if name in metrics
        }
        series = self.series.get(name)
        ids = self.item_ids
        if not values and isinstance(series, list) and ids is not None and len(series) == len(ids):
            values = {item: value for item, value in zip(ids, series)}
        return values


def _flatten(node: Any, name: str, out: dict) -> None:
    if is_number(node):
        if math.isfinite(float(node)):
            out[name] = node
        return
    if not isinstance(node, dict):
        return
    if isinstance(node.get("type"), str):
        return  # typed leaf (distribution), not a scalar metric
    for key, value in node.items():
        if isinstance(key, str) and _METRIC_KEY_RE.match(key):
            _flatten(value, f"{name}.{key}", out)


def flatten_metrics(tree: Mapping[str, Any]) -> dict:
    """Aggregate scalar metrics of a results file: ``{dotted name: value}``."""
    out: dict = {}
    for key, value in tree.items():
        if key in RESERVED_TOP_LEVEL_KEYS or not _METRIC_KEY_RE.match(key):
            continue
        _flatten(value, key, out)
    return out


def flatten_per_file(node: Any, path: Optional[list] = None, out: Optional[dict] = None) -> dict:
    """``{item id: {dotted name: value}}`` of a ``per_file_metrics`` tree."""
    out = {} if out is None else out
    path = [] if path is None else path
    if not isinstance(node, dict):
        return out
    children = node.get("children")
    if isinstance(children, dict):
        for key, child in children.items():
            flatten_per_file(child, path + [key], out)
    files = node.get("files")
    if isinstance(files, list):
        for entry in files:
            if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
                continue
            item = ITEM_ID_SEPARATOR.join(path + [entry["id"]])
            values = out.setdefault(item, {})
            metrics = entry.get("metrics")
            if isinstance(metrics, dict):
                for key, value in metrics.items():
                    if isinstance(key, str) and _METRIC_KEY_RE.match(key):
                        _flatten(value, key, values)
    return out


def load_eval_file(source: Union[str, os.PathLike, Mapping[str, Any]]) -> EvalFile:
    """Load a results file from a path, a JSON string or a parsed dict."""
    if isinstance(source, Mapping):
        raw = dict(source)
    elif isinstance(source, str) and source.lstrip().startswith("{"):
        raw = json.loads(source)
    else:
        with open(source, encoding="utf-8") as handle:
            raw = json.load(handle)
    metric_set = raw.get("metricSet")
    if not isinstance(metric_set, dict):
        metric_set = raw.get("metric_set") if isinstance(raw.get("metric_set"), dict) else {}
    protocol = raw.get("protocol") if isinstance(raw.get("protocol"), dict) else None
    contract = normalize(protocol, metric_set, raw.get("meta"))
    namespace = contract["metricSet"]["metricNamespace"]
    axes = {}
    for name, declaration in contract["metricSet"]["axes"].items():
        try:
            axes[name] = AxisDeclaration.from_dict(declaration)
        except (KeyError, TypeError, ValueError):
            continue
    per_file = flatten_per_file(raw.get("per_file_metrics"))
    items = contract.get("items") if isinstance(contract.get("items"), dict) else {}
    series = items.get("series") if isinstance(items.get("series"), dict) else {}
    ids = items.get("ids")
    if isinstance(ids, list) and all(isinstance(item, str) for item in ids):
        item_ids: Optional[list] = list(ids)
    elif "per_file_metrics" in raw:
        item_ids = list(per_file)
    else:
        item_ids = None
    return EvalFile(
        raw=raw,
        namespace=namespace,
        axes=axes,
        metric_set=metric_set,
        protocol=protocol,
        contract=contract,
        metrics=flatten_metrics(raw),
        per_file=per_file,
        file_ids=list(per_file),
        series={key: value for key, value in series.items() if isinstance(value, list)},
        item_ids=item_ids,
    )


def file_evidence(eval_file: EvalFile) -> tuple:
    """``(items, per_item_names)`` of a loaded file for :func:`compare`.

    ``items`` is the NFC-normalized set of evaluated item ids, or ``None``
    when unknown — also when ``population.manifest`` is recorded and the ids
    found in the file do not hash to it.
    """
    ids = eval_file.item_ids
    items: Optional[frozenset] = None
    if ids is not None:
        normalized = frozenset(unicodedata.normalize("NFC", item) for item in ids)
        population = eval_file.contract.get("population")
        manifest = population.get("manifest") if isinstance(population, dict) else None
        if isinstance(manifest, str) and manifest_hash(normalized) != manifest:
            items = None
        else:
            items = normalized
    return items, frozenset(eval_file.per_item_names)
