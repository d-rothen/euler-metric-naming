"""JSON Schemas, vocabularies and the legacy table (validated with jsonschema)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from euler_metric_naming.contract import (
    available_vocabularies,
    closed,
    legacy_table,
    load_vocabulary,
    protocol_problems,
)

jsonschema = pytest.importorskip("jsonschema")
referencing = pytest.importorskip("referencing")

CONTRACT = Path(__file__).resolve().parents[2] / "euler_metric_naming" / "contract"


def _schema(name):
    return json.loads((CONTRACT / "schema" / name).read_text(encoding="utf-8"))


PROTOCOL = _schema("protocol-2.schema.json")
VOCABULARY = _schema("vocabulary-1.schema.json")
REGISTRY = referencing.Registry().with_resources(
    [
        (PROTOCOL["$id"], referencing.Resource.from_contents(PROTOCOL)),
        (VOCABULARY["$id"], referencing.Resource.from_contents(VOCABULARY)),
    ]
)


def _validator(schema):
    cls = jsonschema.validators.validator_for(schema)
    cls.check_schema(schema)
    return cls(schema, registry=REGISTRY)


def test_schema_ids():
    assert PROTOCOL["$id"] == "https://github.com/d-rothen/euler-metric-naming/contract/protocol-2.schema.json"
    assert PROTOCOL["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert VOCABULARY["$id"].endswith("/contract/vocabulary-1.schema.json")


def test_schema_enums_are_the_closed_set():
    defs = PROTOCOL["$defs"]
    assert defs["role"]["enum"] == list(closed.ROLES)
    assert defs["compare"]["enum"] == list(closed.COMPARE_RULES)
    assert defs["observe"]["enum"] == list(closed.OBSERVE_KINDS)
    assert defs["axisRoleName"]["enum"] == list(closed.AXIS_ROLES)
    assert defs["conditionOp"]["enum"] == list(closed.CONDITION_OPS)
    assert defs["reducer"]["properties"]["recompute"]["enum"] == list(closed.RECOMPUTE_OPS)
    metric = defs["metricDefinition"]["properties"]
    assert metric["kind"]["enum"] == list(closed.METRIC_KINDS)
    assert metric["direction"]["enum"] == list(closed.DIRECTIONS)
    assert metric["display"]["properties"]["scale"]["enum"] == list(closed.DISPLAY_SCALES)
    assert metric["pool"]["properties"]["form"]["enum"] == list(closed.POOL_FORMS)


@pytest.mark.parametrize("vocabulary_id", available_vocabularies())
def test_vocabularies_validate(vocabulary_id):
    document = load_vocabulary(vocabulary_id).document
    errors = sorted(_validator(VOCABULARY).iter_errors(document), key=str)
    assert not errors, [e.message for e in errors[:5]]
    assert document["id"] == vocabulary_id
    assert document["version"] == "1.0"


def test_vocabulary_ids_and_terms():
    assert available_vocabularies() == [
        "euler.core",
        "euler.depth",
        "euler.points3d",
        "euler.rays",
        "euler.rgb",
        "euler.sky",
    ]
    for vocabulary_id in available_vocabularies():
        document = load_vocabulary(vocabulary_id).document
        for key, descriptor in document["descriptors"].items():
            problems = protocol_problems(
                {"schemaVersion": "2.0", "parameters": {}, "descriptors": {key: descriptor}}
            )
            assert problems == [], (key, problems)
        for base, definition in document["metrics"].items():
            assert definition["rev"] == 1, base
            problems = protocol_problems(
                {"schemaVersion": "2.0", "parameters": {}, "metrics": {base: definition}}
            )
            assert problems == [], (base, problems)


def test_descriptor_keys_are_unique_across_vocabularies():
    seen = {}
    for vocabulary_id in available_vocabularies():
        for key in load_vocabulary(vocabulary_id).descriptors:
            assert key not in seen, (key, seen.get(key), vocabulary_id)
            seen[key] = vocabulary_id


def test_fixture_protocols_validate_against_the_schema():
    validator = _validator(PROTOCOL)
    for path in sorted((CONTRACT / "fixtures" / "evals").glob("*.json")):
        protocol = json.loads(path.read_text(encoding="utf-8"))["protocol"]
        if not protocol["schemaVersion"].startswith("2."):
            continue
        errors = list(validator.iter_errors(protocol))
        if path.stem in ("d10u", "d10v"):  # unknown roles on purpose
            assert errors and all("tuning" in e.message for e in errors), path.name
        else:
            assert not errors, (path.name, [e.message for e in errors[:3]])


def test_schema_accepts_unknown_properties_and_rejects_open_enums():
    validator = _validator(PROTOCOL)
    ok = {"schemaVersion": "2.3", "parameters": {"a.b": 1}, "future": {"x": 1},
          "descriptors": {"a.b": {"role": "definition", "newField": True}}}
    assert not list(validator.iter_errors(ok))
    bad = {"schemaVersion": "2.0", "parameters": {}, "axes": {"bin": {"role": "stratum"}}}
    assert list(validator.iter_errors(bad))
    bad_list = {"schemaVersion": "2.0", "parameters": {"k": list(range(65))}}
    assert list(validator.iter_errors(bad_list))
    bad_condition = {
        "schemaVersion": "2.0",
        "parameters": {},
        "descriptors": {"k.x": {"role": "definition", "appliesWhen": {"param": "a", "observation": "b", "op": "exists"}}},
    }
    assert list(validator.iter_errors(bad_condition))


def test_legacy_table_descriptors_are_vocabulary_snapshots():
    table = legacy_table()
    for key, descriptor in table["descriptors"].items():
        vocabulary_id, version = descriptor["vocabulary"].rsplit("@", 1)
        if vocabulary_id == "euler.legacy":
            continue
        vocabulary = load_vocabulary(vocabulary_id)
        if vocabulary.version != version:
            continue  # the table is frozen; newer vocabularies may differ
        expected = dict(vocabulary.descriptors[key], vocabulary=descriptor["vocabulary"])
        assert descriptor == expected, key
    for name, reducer in table["reducers"].items():
        assert reducer["recompute"] == ("none" if name == "pixel_pool" else load_vocabulary("euler.core").reducers[name]["recompute"])
