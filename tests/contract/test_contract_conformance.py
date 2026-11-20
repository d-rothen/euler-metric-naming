"""The shared conformance fixtures (docs/contract.md §11) against the Python implementation."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from euler_metric_naming.contract import (
    CanonicalizationError,
    canonicalize,
    compare_files,
    definition_key,
    definition_key_object,
    evaluate_condition,
    file_evidence,
    format_percent,
    format_sig3,
    glob_match,
    hash_value,
    legacy_table,
    load_eval_file,
    manifest_hash,
    manifest_ids,
    measure,
    normalize,
    phrase,
    recompute_cell,
)

FIXTURES = Path(__file__).resolve().parents[2] / "euler_metric_naming" / "contract" / "fixtures"


def _load(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _ids(cases):
    return [case["name"] for case in cases]


# -- jcs -----------------------------------------------------------------------------

JCS = _load("jcs.json")["cases"]
SPECIAL = {"NaN": math.nan, "Infinity": math.inf, "-Infinity": -math.inf}


@pytest.mark.parametrize("case", JCS, ids=_ids(JCS))
def test_jcs(case):
    if "special" in case:
        value = SPECIAL[case["special"]]
    elif "inputText" in case:
        value = json.loads(case["inputText"])
    else:
        value = case["input"]
    if case.get("error"):
        with pytest.raises(CanonicalizationError):
            canonicalize(value)
    else:
        assert canonicalize(value) == case["canonical"]


# -- hashes --------------------------------------------------------------------------

HASHES = _load("hashes.json")


@pytest.mark.parametrize("case", HASHES["values"], ids=_ids(HASHES["values"]))
def test_hash_values(case):
    assert canonicalize(case["input"]) == case["canonical"]
    assert hash_value(case["input"]) == case["hash"]


@pytest.mark.parametrize("case", HASHES["manifests"], ids=_ids(HASHES["manifests"]))
def test_manifests(case):
    assert canonicalize(manifest_ids(case["ids"])) == case["canonical"]
    assert manifest_hash(case["ids"]) == case["hash"]


def test_d1_manifest_matches_population():
    d1 = _load("evals/d1.json")
    case = next(c for c in HASHES["manifests"] if c["name"] == "the d1 population")
    assert d1["protocol"]["population"]["manifest"] == case["hash"]


@pytest.mark.parametrize("case", HASHES["referenceIdentities"], ids=_ids(HASHES["referenceIdentities"]))
def test_reference_identities(case):
    assert canonicalize(case["input"]) == case["canonical"]
    assert hash_value(case["input"]) == case["hash"]


@pytest.mark.parametrize("case", HASHES["definitionKeys"], ids=_ids(HASHES["definitionKeys"]))
def test_definition_keys(case):
    for name in case["files"]:
        contract = load_eval_file(FIXTURES / "evals" / name).contract
        assert definition_key_object(contract) == case["object"]
        assert canonicalize(definition_key_object(contract)) == case["canonical"]
        assert definition_key(contract) == case["hash"]


# -- conditions ----------------------------------------------------------------------

CONDITIONS = _load("conditions.json")
TRUTH = {"true": True, "false": False, "unknown": None}


@pytest.mark.parametrize("case", CONDITIONS["cases"], ids=_ids(CONDITIONS["cases"]))
def test_conditions(case):
    result = evaluate_condition(
        case["condition"], CONDITIONS["parameters"], CONDITIONS["observations"]
    )
    assert result is TRUTH[case["expected"]]


# -- globs and measures ----------------------------------------------------------------

GLOBS = _load("globs.json")["cases"]


@pytest.mark.parametrize("case", GLOBS, ids=[f"{c['pattern']}|{c['name']}" for c in GLOBS])
def test_globs(case):
    assert glob_match(case["pattern"], case["name"]) is case["match"]


MEASURES = _load("measures.json")


@pytest.mark.parametrize("case", MEASURES["measures"], ids=_ids(MEASURES["measures"]))
def test_measures(case):
    d = measure(case["kind"], case["a"], case["b"])
    if case["d"] is None:
        assert d is None
        return
    assert d is not None and abs(d - case["d"]) <= 5e-7
    assert phrase(case["kind"], "L", case["a"], case["b"], d) == case["phrase"]


@pytest.mark.parametrize("case", MEASURES["percent"], ids=[str(c["d"]) for c in MEASURES["percent"]])
def test_percent(case):
    assert format_percent(case["d"]) == case["text"]


@pytest.mark.parametrize("case", MEASURES["sig3"], ids=[repr(c["value"]) for c in MEASURES["sig3"]])
def test_sig3(case):
    assert format_sig3(case["value"]) == case["text"]


# -- legacy ----------------------------------------------------------------------------

LEGACY = _load("legacy.json")["cases"]


@pytest.mark.parametrize("case", LEGACY, ids=_ids(LEGACY))
def test_legacy(case):
    table = legacy_table()
    contract = normalize(case["protocol"], case["metricSet"], case["meta"])
    descriptors = contract.pop("descriptors")
    reducers = contract.pop("reducers")
    assert contract == case["expected"]
    assert sorted(descriptors) == case["descriptorKeys"]
    for key, descriptor in descriptors.items():
        assert descriptor == table["descriptors"][key]
    assert reducers == table["reducers"]


# -- verdicts --------------------------------------------------------------------------

VERDICTS = _load("verdicts.json")["cases"]


@pytest.mark.parametrize("case", VERDICTS, ids=_ids(VERDICTS))
def test_verdicts(case):
    file_a = load_eval_file(FIXTURES / "evals" / case["a"])
    file_b = load_eval_file(FIXTURES / "evals" / case["b"])
    verdict = compare_files(file_a, file_b, case["metric"])
    assert verdict.tier == case["tier"]
    assert verdict.label == case["label"]
    assert verdict.codes == case["codes"]
    assert verdict.shared == case["shared"]
    if "measures" in case:
        measures = [
            {"key": r.key, "kind": r.measure["kind"], "d": r.measure["d"]}
            for r in verdict.reasons
            if r.measure is not None
        ]
        assert [(m["key"], m["kind"]) for m in measures] == [
            (m["key"], m["kind"]) for m in case["measures"]
        ]
        for got, expected in zip(measures, case["measures"]):
            assert abs(got["d"] - expected["d"]) <= 5e-7
    if "hint" in case:
        assert verdict.reasons[0].hint == case["hint"]
    if "recompute" in case:
        items_a, _ = file_evidence(file_a)
        items_b, _ = file_evidence(file_b)
        shared = items_a & items_b
        got_a = recompute_cell(file_a.contract, file_a, case["metric"], shared)
        got_b = recompute_cell(file_b.contract, file_b, case["metric"], shared)
        assert abs(got_a - case["recompute"]["a"]) <= 1e-9
        assert abs(got_b - case["recompute"]["b"]) <= 1e-9


def test_verdicts_are_symmetric_in_tier_and_codes():
    """compare(B, A) has the tier and reason codes of compare(A, B)."""
    for case in VERDICTS:
        file_a = load_eval_file(FIXTURES / "evals" / case["a"])
        file_b = load_eval_file(FIXTURES / "evals" / case["b"])
        forward = compare_files(file_a, file_b, case["metric"])
        backward = compare_files(file_b, file_a, case["metric"])
        assert backward.tier == forward.tier, case["name"]
        assert backward.codes == forward.codes, case["name"]
        assert backward.shared == forward.shared, case["name"]


def test_fixture_files_validate_as_v2():
    from euler_metric_naming.contract import protocol_problems

    for path in sorted((FIXTURES / "evals").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        protocol = payload["protocol"]
        if protocol["schemaVersion"].startswith("2."):
            problems = protocol_problems(protocol)
            if path.stem in ("d10u", "d10v"):
                assert problems and all("role" in p for p in problems), path.name
            else:
                assert problems == [], (path.name, problems)


def test_normalize_does_not_mutate_its_input():
    payload = _load("evals/l1.json")
    before = copy.deepcopy(payload)
    normalize(payload["protocol"], payload["metricSet"], payload["meta"])
    assert payload == before
