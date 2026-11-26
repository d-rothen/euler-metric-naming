"""One vocabulary registry: shipped and installed (``euler.vocabularies``) alike."""
from __future__ import annotations

import copy
import json
import sys
import types
from dataclasses import dataclass
from typing import Any

import pytest
from euler_dataset_contract import extensions
from euler_dataset_contract.extensions import ExtensionError

from euler_metric_naming import __version__
from euler_metric_naming.contract import (
    VOCABULARIES_GROUP,
    ProtocolBuilder,
    UnknownVocabularyError,
    VocabularyError,
    VocabularySet,
    available_vocabularies,
    builtin_vocabularies,
    load_vocabulary,
    reload_vocabularies,
    vocabulary_documents,
    vocabulary_problems,
    vocabulary_registry,
)

SHIPPED = [
    "euler.core",
    "euler.depth",
    "euler.points3d",
    "euler.rays",
    "euler.rgb",
    "euler.sky",
    "euler.train",
]

THERMAL = {
    "id": "mylab.thermal",
    "version": "1.2",
    "title": "Thermal imaging",
    "descriptors": {
        "thermal.emissivity": {"label": "emissivity", "role": "definition"},
    },
    "metrics": {
        "kelvin_mae": {
            "id": "thermal.kelvin_mae",
            "kind": "score",
            "rev": 1,
            "displayName": "MAE (K)",
            "direction": "lower",
            "reducer": "image_mean",
        },
    },
    "reducers": {},
}


@dataclass
class FakeDist:
    name: str
    version: str

    @property
    def metadata(self) -> dict:
        return {"Name": self.name}


@dataclass
class FakeEntryPoint:
    name: str
    value: str
    group: str = VOCABULARIES_GROUP
    dist: Any = None


def ep(name: str, value: str, dist: str = "mylab-vocab", version: str = "0.3.0") -> FakeEntryPoint:
    return FakeEntryPoint(name, value, VOCABULARIES_GROUP, FakeDist(dist, version))


@pytest.fixture(autouse=True)
def fresh_registry(monkeypatch):
    monkeypatch.delenv("EULER_EXTENSIONS_STRICT", raising=False)
    reload_vocabularies()
    yield
    monkeypatch.undo()
    reload_vocabularies()


@pytest.fixture
def plugin(monkeypatch, tmp_path):
    module = types.ModuleType("mylab_vocab")
    module.__file__ = str(tmp_path / "mylab_vocab.py")
    module.THERMAL = copy.deepcopy(THERMAL)
    monkeypatch.setitem(sys.modules, "mylab_vocab", module)
    return module


def install(monkeypatch, *entry_points: FakeEntryPoint) -> None:
    monkeypatch.setattr(
        extensions,
        "_raw_entry_points",
        lambda group: [item for item in entry_points if item.group == group],
    )
    reload_vocabularies()


def test_shipped_vocabularies_go_through_the_registry():
    assert builtin_vocabularies() == SHIPPED
    assert available_vocabularies() == SHIPPED
    depth = load_vocabulary("euler.depth")
    assert depth.builtin
    assert depth.provider == {
        "kind": "builtin",
        "distribution": "euler-metric-naming",
        "version": __version__,
    }
    assert vocabulary_registry().problems == ()


@pytest.mark.parametrize("vocabulary_id", SHIPPED)
def test_shipped_vocabularies_are_valid_documents(vocabulary_id):
    assert vocabulary_problems(load_vocabulary(vocabulary_id).document) == []


def test_installed_mapping_is_usable_by_name(monkeypatch, plugin):
    install(monkeypatch, ep("mylab.thermal", "mylab_vocab:THERMAL"))
    thermal = load_vocabulary("mylab.thermal")
    assert thermal.tag == "mylab.thermal@1.2"
    assert not thermal.builtin
    assert thermal.provider == {
        "kind": "entry_point",
        "name": "mylab.thermal",
        "target": "mylab_vocab:THERMAL",
        "distribution": "mylab-vocab",
        "version": "0.3.0",
    }
    assert available_vocabularies() == sorted(SHIPPED + ["mylab.thermal"])
    assert builtin_vocabularies() == SHIPPED
    assert "mylab.thermal" in vocabulary_documents()
    assert "mylab.thermal" not in vocabulary_documents(builtin_only=True)

    builder = ProtocolBuilder("thermal.dense", ["euler.core", "mylab.thermal"])
    builder.param("thermal.emissivity", 0.95).metric("kelvin_mae")
    protocol = builder.build()
    assert protocol["descriptors"]["thermal.emissivity"]["vocabulary"] == "mylab.thermal@1.2"
    assert protocol["metrics"]["kelvin_mae"]["vocabulary"] == "mylab.thermal@1.2"
    assert protocol["reducers"]["image_mean"]["vocabulary"].startswith("euler.core@")


def test_json_paths_absolute_relative_and_callables(monkeypatch, plugin, tmp_path):
    (tmp_path / "vocab").mkdir()
    for vid in ("mylab.abs", "mylab.rel", "mylab.call"):
        document = dict(copy.deepcopy(THERMAL), id=vid)
        (tmp_path / "vocab" / f"{vid}.json").write_text(json.dumps(document))
    plugin.ABSOLUTE = tmp_path / "vocab" / "mylab.abs.json"
    plugin.RELATIVE = "vocab/mylab.rel.json"  # next to the module that names it
    plugin.make = lambda: str(tmp_path / "vocab" / "mylab.call.json")
    install(
        monkeypatch,
        ep("mylab.abs", "mylab_vocab:ABSOLUTE"),
        ep("mylab.rel", "mylab_vocab:RELATIVE"),
        ep("mylab.call", "mylab_vocab:make"),
    )
    for vid in ("mylab.abs", "mylab.rel", "mylab.call"):
        assert load_vocabulary(vid).id == vid
    assert vocabulary_registry().problems == ()


def test_a_shipped_vocabulary_cannot_be_shadowed(monkeypatch, plugin):
    plugin.FAKE_DEPTH = dict(copy.deepcopy(THERMAL), id="euler.depth")
    install(monkeypatch, ep("euler.depth", "mylab_vocab:FAKE_DEPTH"))
    assert load_vocabulary("euler.depth").builtin
    (problem,) = vocabulary_registry().problems
    assert problem.code == "shadows_builtin"
    assert problem.name == "euler.depth"
    assert problem.distributions == ("mylab-vocab",)
    assert "cannot be redefined" in problem.message


def test_invalid_documents_are_rejected_and_reported(monkeypatch, plugin):
    broken = copy.deepcopy(THERMAL)
    broken["metrics"]["kelvin_mae"]["direction"] = "sideways"
    del broken["metrics"]["kelvin_mae"]["displayName"]
    plugin.BROKEN = dict(broken, id="mylab.broken")
    plugin.MISNAMED = dict(copy.deepcopy(THERMAL), id="mylab.other")
    plugin.NOT_A_VOCAB = 42
    plugin.MISSING = "nowhere.json"
    install(
        monkeypatch,
        ep("mylab.broken", "mylab_vocab:BROKEN"),
        ep("mylab.misnamed", "mylab_vocab:MISNAMED"),
        ep("mylab.int", "mylab_vocab:NOT_A_VOCAB"),
        ep("mylab.missing", "mylab_vocab:MISSING"),
    )
    problems = {problem.name: problem for problem in vocabulary_registry().problems}
    assert {name: problem.code for name, problem in problems.items()} == {
        "mylab.broken": "invalid",
        "mylab.misnamed": "invalid",
        "mylab.int": "invalid",
        "mylab.missing": "invalid",
    }
    assert "metrics.kelvin_mae.direction must be one of" in problems["mylab.broken"].message
    assert "metrics.kelvin_mae.displayName is required" in problems["mylab.broken"].message
    assert "registered as 'mylab.misnamed'" in problems["mylab.misnamed"].message
    assert "not int" in problems["mylab.int"].message
    assert available_vocabularies() == SHIPPED
    with pytest.raises(UnknownVocabularyError, match="was not registered: .*direction"):
        load_vocabulary("mylab.broken")


def test_conflicts_and_import_failures_follow_the_protocol(monkeypatch, plugin):
    install(
        monkeypatch,
        ep("mylab.thermal", "mylab_vocab:THERMAL", dist="lab-a"),
        ep("mylab.thermal", "mylab_vocab:THERMAL", dist="lab-b"),
        ep("mylab.gone", "no_such_module_xyz:VOCAB"),
    )
    codes = {problem.name: problem.code for problem in vocabulary_registry().problems}
    assert codes == {"mylab.thermal": "conflict", "mylab.gone": "load_failed"}
    with pytest.raises(UnknownVocabularyError, match="lab-a, lab-b"):
        load_vocabulary("mylab.thermal")


def test_strict_mode_raises(monkeypatch, plugin):
    install(monkeypatch, ep("euler.core", "mylab_vocab:THERMAL"))
    monkeypatch.setenv("EULER_EXTENSIONS_STRICT", "1")
    with pytest.raises(ExtensionError, match="cannot be redefined"):
        reload_vocabularies()


def test_unknown_ids_name_the_registry():
    with pytest.raises(UnknownVocabularyError) as info:
        load_vocabulary("mylab.nothing")
    assert isinstance(info.value, KeyError)
    assert "registered: euler.core, euler.depth" in str(info.value)
    assert "'euler.vocabularies'" in str(info.value)


def test_inline_documents_are_validated():
    assert VocabularySet([copy.deepcopy(THERMAL)]).metric("kelvin_mae")["vocabulary"] == (
        "mylab.thermal@1.2"
    )
    with pytest.raises(VocabularyError, match="version must be"):
        VocabularySet([dict(copy.deepcopy(THERMAL), version="1")])


@pytest.mark.parametrize(
    "mutate, expected",
    [
        (lambda d: d.pop("reducers"), "reducers is required"),
        (lambda d: d.update(id="Thermal"), "id must be"),
        (lambda d: d.update(context="infer"), "context must be one of"),
        (lambda d: d["descriptors"].update({"flat": {"role": "definition"}}), "descriptors.flat: key"),
        (lambda d: d["descriptors"]["thermal.emissivity"].update(role="knob"), ".role must be one of"),
        (
            lambda d: d["metrics"]["kelvin_mae"].update(vocabulary="x@1.0"),
            "must not carry a vocabulary tag",
        ),
        (lambda d: d["reducers"].update(r={"recompute": "sum"}), "reducers.r.recompute"),
        (lambda d: d.update(stratificationSchemes={"s": {}}), "stratificationSchemes.s.label"),
    ],
)
def test_vocabulary_problems(mutate, expected):
    document = copy.deepcopy(THERMAL)
    mutate(document)
    assert any(expected in problem for problem in vocabulary_problems(document)), (
        vocabulary_problems(document)
    )
