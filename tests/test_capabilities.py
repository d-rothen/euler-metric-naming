"""The ``euler.capabilities`` entry: ``metric_vocabulary_report`` 1.0."""
from __future__ import annotations

import copy
import importlib.metadata
import json
import sys
import types
from pathlib import Path

import pytest
from euler_dataset_contract import extensions

from euler_metric_naming import __version__
from euler_metric_naming.capabilities import capability_report
from euler_metric_naming.contract import reload_vocabularies

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def fresh_registry(monkeypatch):
    reload_vocabularies()
    yield
    monkeypatch.undo()
    reload_vocabularies()


def test_report_shape():
    report = capability_report()
    assert report["contract"] == {"kind": "metric_vocabulary_report", "version": "1.0"}
    assert report["package"] == {"name": "euler-metric-naming", "version": __version__}
    assert report["naming"]["grammar"] == "{scope}.{context}.{axes...}.{metric}"
    assert report["protocol"] == {"schema_version": "2.0", "supported_majors": [1, 2]}
    assert report["training_naming"] == {"schema_version": "2.0"}
    assert report["validation_set"] == {"kind": "euler.validation_set", "schema_version": "1.0"}
    assert report["extension_group"] == {"group": "euler.vocabularies", "api_version": "1.0"}
    assert report["extension_problems"] == []
    json.dumps(report)  # JSON-serializable, as the suite report requires

    by_id = {entry["id"]: entry for entry in report["vocabularies"]}
    assert sorted(by_id) == [
        "euler.core", "euler.depth", "euler.points3d", "euler.rays",
        "euler.rgb", "euler.sky", "euler.train",
    ]
    depth = by_id["euler.depth"]
    assert depth["version"] == "1.1" and depth["tag"] == "euler.depth@1.1"
    assert depth["context"] == "eval" and by_id["euler.train"]["context"] == "train"
    assert depth["provider"]["kind"] == "builtin"
    assert depth["terms"]["metrics"] > 0 and set(depth["terms"]) == {
        "descriptors", "metrics", "reducers", "stratificationSchemes"
    }
    schemas = {entry["name"]: entry for entry in report["schemas"]}
    assert set(schemas) == {
        "protocol-2", "training-naming-2", "validation-pass-1",
        "validation-set-1", "vocabulary-1",
    }
    assert schemas["vocabulary-1"]["id"].endswith("/contract/vocabulary-1.schema.json")


def test_report_lists_installed_vocabularies_and_problems(monkeypatch):
    module = types.ModuleType("report_vocab")
    module.GOOD = {
        "id": "mylab.thermal", "version": "1.0",
        "descriptors": {}, "metrics": {}, "reducers": {},
    }
    module.SHADOW = dict(copy.deepcopy(module.GOOD), id="euler.rgb")
    monkeypatch.setitem(sys.modules, "report_vocab", module)

    class Dist:
        metadata = {"Name": "mylab"}
        version = "2.0.0"

    class EP:
        def __init__(self, name, value):
            self.name, self.value, self.group, self.dist = name, value, "euler.vocabularies", Dist()

    monkeypatch.setattr(
        extensions,
        "_raw_entry_points",
        lambda group: [EP("mylab.thermal", "report_vocab:GOOD"), EP("euler.rgb", "report_vocab:SHADOW")]
        if group == "euler.vocabularies"
        else [],
    )
    reload_vocabularies()
    report = capability_report()
    thermal = next(entry for entry in report["vocabularies"] if entry["id"] == "mylab.thermal")
    assert thermal["provider"] == {
        "kind": "entry_point", "name": "mylab.thermal", "target": "report_vocab:GOOD",
        "distribution": "mylab", "version": "2.0.0",
    }
    assert thermal["terms"] == {
        "descriptors": 0, "metrics": 0, "reducers": 0, "stratificationSchemes": 0
    }
    (problem,) = report["extension_problems"]
    assert problem["code"] == "shadows_builtin" and problem["name"] == "euler.rgb"
    assert problem["distributions"] == ["mylab"]


def test_registered_as_a_capability_provider():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '[project.entry-points."euler.capabilities"]' in text
    assert 'euler-metric-naming = "euler_metric_naming.capabilities:capability_report"' in text


def test_the_suite_report_collects_it():
    installed = [
        ep for ep in importlib.metadata.entry_points(group="euler.capabilities")
        if ep.name == "euler-metric-naming"
    ]
    if not installed:
        pytest.skip("euler-metric-naming is not installed (entry points need metadata)")
    from euler_dataset_contract.suite import collect_capabilities

    capabilities, findings = collect_capabilities(only=["euler-metric-naming"])
    entry = capabilities["euler-metric-naming"]
    assert entry["status"] == "ok", entry
    assert entry["report"]["contract"]["kind"] == "metric_vocabulary_report"
