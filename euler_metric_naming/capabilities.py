"""What this installation of euler-metric-naming can do: ``metric_vocabulary_report``.

Registered in the suite's ``euler.capabilities`` group, so ``euler-suite
report`` (and euler-view, per cluster environment) sees which naming grammar
and contract versions this environment speaks, every vocabulary it can
snapshot terms from and who provided it, the JSON Schemas it ships, and every
installed vocabulary that was rejected and why::

    {
      "contract": {"kind": "metric_vocabulary_report", "version": "1.0"},
      "package": {"name": "euler-metric-naming", "version": "0.5.0"},
      "naming": {"grammar": "{scope}.{context}.{axes...}.{metric}", ...},
      "protocol": {"schema_version": "2.0", "supported_majors": [1, 2]},
      "training_naming": {"schema_version": "2.0"},
      "validation_set": {"kind": "euler.validation_set", "schema_version": "1.0"},
      "extension_group": {"group": "euler.vocabularies", "api_version": "1.0"},
      "vocabularies": [{"id", "version", "tag", "context", "title",
                        "terms": {"descriptors", "metrics", "reducers",
                                  "stratificationSchemes"},
                        "provider": {"kind": "builtin" | "entry_point", ...}}],
      "schemas": [{"name": "vocabulary-1", "file", "id"}],
      "extension_problems": [{"group", "name", "code", "message", "distributions"}]
    }
"""
from __future__ import annotations

import json
from importlib import resources
from typing import Any

METRIC_VOCABULARY_REPORT_KIND = "metric_vocabulary_report"
METRIC_VOCABULARY_REPORT_VERSION = "1.0"
NAMING_GRAMMAR = "{scope}.{context}.{axes...}.{metric}"


def _schemas() -> list[dict[str, Any]]:
    folder = resources.files("euler_metric_naming.contract") / "schema"
    shipped = []
    for entry in sorted(folder.iterdir(), key=lambda item: item.name):
        if not entry.name.endswith(".schema.json"):
            continue
        schema = json.loads(entry.read_text(encoding="utf-8"))
        shipped.append(
            {
                "name": entry.name[: -len(".schema.json")],
                "file": f"contract/schema/{entry.name}",
                "id": schema.get("$id"),
            }
        )
    return shipped


def capability_report() -> dict[str, Any]:
    """euler-metric-naming's ``euler.capabilities`` entry (``metric_vocabulary_report`` 1.0)."""
    from euler_dataset_contract.extensions import GROUPS

    from . import __version__
    from .contract.closed import SCHEMA_VERSION, SUPPORTED_MAJORS
    from .contract.training import (
        TRAINING_SCHEMA_VERSION,
        VALIDATION_SET_KIND,
        VALIDATION_SET_SCHEMA_VERSION,
    )
    from .contract.vocabulary import VOCABULARIES_GROUP, vocabulary_registry
    from .namespace import _RESERVED_SCOPES

    registry = vocabulary_registry()
    group = GROUPS.get(VOCABULARIES_GROUP)
    return {
        "contract": {
            "kind": METRIC_VOCABULARY_REPORT_KIND,
            "version": METRIC_VOCABULARY_REPORT_VERSION,
        },
        "package": {"name": "euler-metric-naming", "version": __version__},
        "naming": {
            "grammar": NAMING_GRAMMAR,
            "contexts": ["train", "eval"],
            "reserved_scopes": sorted(_RESERVED_SCOPES),
        },
        "protocol": {
            "schema_version": SCHEMA_VERSION,
            "supported_majors": list(SUPPORTED_MAJORS),
        },
        "training_naming": {"schema_version": TRAINING_SCHEMA_VERSION},
        "validation_set": {
            "kind": VALIDATION_SET_KIND,
            "schema_version": VALIDATION_SET_SCHEMA_VERSION,
        },
        "extension_group": {
            "group": VOCABULARIES_GROUP,
            "api_version": group.api_version if group is not None else None,
        },
        "vocabularies": [vocab.describe() for vocab in registry.vocabularies.values()],
        "schemas": _schemas(),
        "extension_problems": [problem.describe() for problem in registry.problems],
    }


__all__ = [
    "METRIC_VOCABULARY_REPORT_KIND",
    "METRIC_VOCABULARY_REPORT_VERSION",
    "NAMING_GRAMMAR",
    "capability_report",
]
