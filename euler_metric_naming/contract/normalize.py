"""Normalize a ``protocol`` block of any schema version into a contract.

A *contract* is the input of the verdict engine (``docs/contract.md`` §8): the
protocol object with every block present (defaults for missing ones), plus
``metricSet: {metricNamespace, axes}`` from the file's envelope, which the
engine needs to decompose metric names. Schema 1.x goes through the legacy
adapter (§10); 2.x passes through; other majors keep their parameters and are
judged by §8.8.
"""
from __future__ import annotations

import copy
import re
from typing import Any, Mapping, Optional

from .closed import SUPPORTED_MAJORS
from .legacy import empty_contract, metric_set_info, normalize_legacy

_VERSION_RE = re.compile(r"^(\d+)(?:\.(\d+))?$")

_RECORD_BLOCKS = (
    "vocabularies",
    "parameters",
    "descriptors",
    "references",
    "observations",
    "population",
    "stratifications",
    "reducers",
    "axes",
    "metrics",
    "items",
)


def schema_major(version: Any) -> Optional[int]:
    """The major of a ``"MAJOR.MINOR"`` (or ``"MAJOR"``) schema version, else None."""
    if not isinstance(version, str):
        return None
    match = _VERSION_RE.match(version.strip())
    return int(match.group(1)) if match else None


def is_supported(contract: Mapping[str, Any]) -> bool:
    """Whether the engine can judge this contract beyond §8.8."""
    return schema_major(contract.get("schemaVersion")) in SUPPORTED_MAJORS


def normalize(
    protocol: Any,
    metric_set: Optional[Mapping[str, Any]] = None,
    meta: Optional[Mapping[str, Any]] = None,
) -> dict:
    """Return the normalized contract of a results file's ``protocol`` block.

    *metric_set* is the file's ``metricSet`` envelope (namespace, axes, and
    for v1 ``metadata``); *meta* is the file's ``meta`` block (v1 dimensions).
    """
    source = protocol if isinstance(protocol, Mapping) else {}
    if schema_major(source.get("schemaVersion")) == 1:
        return normalize_legacy(source, metric_set, meta)
    contract = empty_contract(source.get("schemaVersion"))
    for key, value in source.items():
        contract[key] = copy.deepcopy(value)
    if not isinstance(contract.get("task"), str):
        contract["task"] = None
    for block in _RECORD_BLOCKS:
        if not isinstance(contract.get(block), dict):
            contract[block] = {}
    contract["metricSet"] = metric_set_info(metric_set)
    return contract
