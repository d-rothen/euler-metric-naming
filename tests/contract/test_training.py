"""Training naming 2.0, validation sets and passes (docs/training.md).

The shared fixtures in contract/fixtures/training/ are run here and by the TS
conformance suite; the rest covers the Python producer API.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from euler_metric_naming import MetricDescription, MetricNamespace
from euler_metric_naming.contract import (
    ProtocolBuilder,
    check_protocol,
    compare_files,
    evaluation_metric_ids,
    file_evidence,
    is_training_naming_v2,
    load_eval_file,
    load_validation_pass,
    metric_value_problem,
    recompute_cell,
    training_contract,
    training_metric_definition,
    training_metric_problem,
    training_naming_problems,
    validation_pass_contract,
    validation_pass_problems,
    validation_set_problems,
)

FIXTURES = Path(__file__).resolve().parents[2] / "euler_metric_naming" / "contract" / "fixtures"
DOCUMENTS = json.loads((FIXTURES / "training" / "documents.json").read_text(encoding="utf-8"))
CASES = json.loads((FIXTURES / "training" / "cases.json").read_text(encoding="utf-8"))


def _codes(problems):
    return [{"code": problem.code, "path": problem.path} for problem in problems]


def _load(ref):
    kind, name = ref.split(":", 1)
    if kind == "eval":
        return load_eval_file(FIXTURES / "evals" / name)
    entry = DOCUMENTS["passes"][name]
    return load_validation_pass(DOCUMENTS["sets"][entry["set"]], entry["record"])


# -- the shared fixtures -------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES["declarations"], ids=lambda c: c["declaration"])
def test_declaration_problems(case):
    declaration = DOCUMENTS["declarations"][case["declaration"]]
    assert _codes(training_naming_problems(declaration)) == case["problems"]


@pytest.mark.parametrize("case", CASES["keys"], ids=lambda c: c["key"])
def test_key_problems_and_definitions(case):
    declaration = DOCUMENTS["declarations"][case["declaration"]]
    problem = training_metric_problem(declaration, case["key"])
    assert (problem.code if problem else None) == case["problem"]
    expected = case["definition"]
    definition = training_metric_definition(declaration, case["key"])
    if expected is None:
        assert definition is None
    else:
        namespace = declaration["namespaces"][expected["namespace"]]
        assert definition == namespace["metrics"][expected["base"]]


@pytest.mark.parametrize("case", CASES["sets"], ids=lambda c: c["set"])
def test_set_problems(case):
    document = DOCUMENTS["sets"][case["set"]]
    assert _codes(validation_set_problems(document)) == case["problems"]
    if not case["problems"]:
        check_protocol(document["protocol"])


@pytest.mark.parametrize("case", CASES["passes"], ids=lambda c: c["pass"])
def test_pass_problems(case):
    entry = DOCUMENTS["passes"][case["pass"]]
    problems = validation_pass_problems(DOCUMENTS["sets"][entry["set"]], entry["record"])
    assert _codes(problems) == case["problems"]


@pytest.mark.parametrize("case", CASES["contracts"], ids=lambda c: c["pass"])
def test_pass_contracts(case):
    entry = DOCUMENTS["passes"][case["pass"]]
    contract = validation_pass_contract(DOCUMENTS["sets"][entry["set"]], entry["record"])
    assert contract["items"].get("ids") == case["itemIds"]
    if "populationCount" in case:
        assert contract["population"]["count"] == case["populationCount"]
    for key, value in case.get("parameters", {}).items():
        assert contract["parameters"][key] == value, key
    for key, value in case.get("observations", {}).items():
        assert contract["observations"][key] == value, key
    assert sorted(contract["items"]["series"]) == case["seriesNames"]
    loaded = _load(f"pass:{case['pass']}")
    assert loaded.item_ids == case["itemIds"]


@pytest.mark.parametrize("case", CASES["verdicts"], ids=lambda c: c["name"])
def test_training_verdicts(case):
    file_a, file_b = _load(case["a"]), _load(case["b"])
    verdict = compare_files(file_a, file_b, case["metric"])
    assert verdict.tier == case["tier"]
    assert verdict.label == case["label"]
    assert verdict.codes == case["codes"]
    assert verdict.shared == case["shared"]
    if "hint" in case:
        assert verdict.reasons[0].hint == case["hint"]
    if "recompute" in case:
        shared = file_evidence(file_a)[0] & file_evidence(file_b)[0]
        got_a = recompute_cell(file_a.contract, file_a, case["metric"], shared)
        got_b = recompute_cell(file_b.contract, file_b, case["metric"], shared)
        assert abs(got_a - case["recompute"]["a"]) <= 1e-9
        assert abs(got_b - case["recompute"]["b"]) <= 1e-9
    backward = compare_files(file_b, file_a, case["metric"])
    assert (backward.tier, backward.codes, backward.shared) == (
        verdict.tier,
        verdict.codes,
        verdict.shared,
    )


def test_a_pass_on_the_evaluated_items_has_the_evaluations_aggregates():
    passed, evaluated = _load("pass:p_same"), _load("eval:d1.json")
    assert passed.metrics
    for name, value in passed.metrics.items():
        assert value == pytest.approx(evaluated.metrics[name], abs=1e-12), name


def test_documents_are_the_generators_output():
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, str(root / "scripts" / "generate_fixtures.py"), "--check"],
        capture_output=True,
        text=True,
        cwd=root,
    )
    assert result.returncode == 0, result.stdout + result.stderr


# -- JSON Schemas -----------------------------------------------------------------------------


def _validators():
    jsonschema = pytest.importorskip("jsonschema")
    referencing = pytest.importorskip("referencing")
    schemas = {}
    for name in (
        "protocol-2.schema.json",
        "training-naming-2.schema.json",
        "validation-set-1.schema.json",
        "validation-pass-1.schema.json",
    ):
        path = FIXTURES.parent / "schema" / name
        schemas[name] = json.loads(path.read_text(encoding="utf-8"))
    registry = referencing.Registry().with_resources(
        [(schema["$id"], referencing.Resource.from_contents(schema)) for schema in schemas.values()]
    )
    out = {}
    for name, schema in schemas.items():
        cls = jsonschema.validators.validator_for(schema)
        cls.check_schema(schema)
        out[name] = cls(schema, registry=registry)
    return out


def test_documents_validate_against_the_schemas():
    validators = _validators()
    naming = validators["training-naming-2.schema.json"]
    assert not list(naming.iter_errors(DOCUMENTS["declarations"]["n_ok"]))
    assert list(naming.iter_errors(DOCUMENTS["declarations"]["n_broken"]))
    assert list(naming.iter_errors(DOCUMENTS["declarations"]["n_v1"]))
    sets = validators["validation-set-1.schema.json"]
    for name in ("v_cerema", "v_range", "v_mask"):
        assert not list(sets.iter_errors(DOCUMENTS["sets"][name])), name
    assert list(sets.iter_errors(DOCUMENTS["sets"]["v_broken"]))
    passes = validators["validation-pass-1.schema.json"]
    for name in ("p_same", "p_subset", "p_no_items", "p_cap", "p_range", "p_mask", "p_lab"):
        assert not list(passes.iter_errors(DOCUMENTS["passes"][name]["record"])), name
    assert list(passes.iter_errors(DOCUMENTS["passes"]["p_bad"]["record"]))
    assert list(passes.iter_errors(DOCUMENTS["passes"]["p_duplicate_ids"]["record"]))


# -- producer API -------------------------------------------------------------------------------


def _namespace(**kwargs):
    return MetricNamespace(
        "euler_train.fixture",
        "1.0.0",
        ("depth", "rgb"),
        stages=("prior", "final"),
        **kwargs,
    )


#: euler-metric-naming 0.3.0's schema-1 payload of _namespace(descriptions=...).
SCHEMA_1_PAYLOAD = json.loads(
    '{"producer_key": "euler_train.fixture", "producer_version": "1.0.0", "namespaces": '
    '{"depth.train": {"axes": {"kind": {"position": 0, "optional": false, "values": '
    '["loss", "diag", "stat"], "description": "Metric kind"}, "stage": {"position": 1, '
    '"optional": true, "values": ["prior", "final"], "description": "Pipeline refinement '
    'stage"}}, "metricDescriptions": {"total": {"isHigherBetter": false, "displayName": '
    '"Total"}, "depth_mae": {"isHigherBetter": false, "min": 0.0, "unit": "m"}}}, '
    '"rgb.train": {"axes": {"kind": {"position": 0, "optional": false, "values": ["loss", '
    '"diag", "stat"], "description": "Metric kind"}, "stage": {"position": 1, "optional": '
    'true, "values": ["prior", "final"], "description": "Pipeline refinement stage"}}, '
    '"metricDescriptions": {"total": {"isHigherBetter": false, "displayName": "Total"}, '
    '"depth_mae": {"isHigherBetter": false, "min": 0.0, "unit": "m"}}}, "sys.train": '
    '{"axes": {}}}}'
)


def test_schema_1_payload_is_unchanged():
    payload = _namespace(
        descriptions={
            "total": MetricDescription(is_higher_better=False, display_name="Total"),
            "depth_mae": MetricDescription(is_higher_better=False, unit="m", min_value=0.0),
        }
    ).training_naming_config()
    assert json.dumps(payload) == json.dumps(SCHEMA_1_PAYLOAD)
    assert not is_training_naming_v2(payload)


def test_schema_2_payload():
    payload = _namespace(
        metrics={"total": MetricDescription(is_higher_better=False, display_name="Total")},
        namespace_metrics={"rgb.train": {"image_psnr": {"kind": "score", "direction": "higher"}}},
    ).training_naming_config()
    assert is_training_naming_v2(payload)
    assert payload["schema_version"] == "2.0"
    assert training_naming_problems(payload) == []
    depth = payload["namespaces"]["depth.train"]
    assert depth["axisRoles"] == {"kind": {"role": "kind"}, "stage": {"role": "stage"}}
    assert depth["metrics"] == {
        "total": {"kind": "score", "displayName": "Total", "direction": "lower"}
    }
    assert depth["metricDescriptions"] == {"total": {"isHigherBetter": False, "displayName": "Total"}}
    assert "image_psnr" in payload["namespaces"]["rgb.train"]["metrics"]
    assert "image_psnr" not in depth["metrics"]
    system = payload["namespaces"]["sys.train"]["metrics"]
    assert system["gpu_util_pct"]["vocabulary"] == "euler.train@1.0"
    assert system["gpu_util_pct"]["unit"] == "%"
    assert training_metric_problem(payload, "sys.train.gpu_mem_used_gb") is None


def test_schema_2_payload_rejects_mixed_or_misplaced_definitions():
    with pytest.raises(ValueError, match="not both"):
        _namespace(metrics={}, descriptions={"x": MetricDescription(is_higher_better=False)})
    with pytest.raises(ValueError, match="context 'train'"):
        MetricNamespace("p", "1", ("depth",), context="eval", metrics={})
    with pytest.raises(ValueError, match="namespace_metrics key"):
        _namespace(namespace_metrics={"segmentation.train": {}})
    with pytest.raises(TypeError):
        _namespace(metrics={"total": 1})
    with pytest.raises(ValueError, match="evaluation metric"):
        _namespace(metrics={"absrel": {"id": "depth.absrel", "direction": "lower"}}).training_naming_config()


def test_description_definition_round_trip():
    description = MetricDescription(
        is_higher_better=True, min_value=0.0, max_value=1.0, unit="dB",
        display_name="PSNR", description="d", scale="log",
    )
    definition = description.to_definition()
    assert definition == {
        "kind": "score", "displayName": "PSNR", "description": "d", "direction": "higher",
        "bounds": [0.0, 1.0], "unit": "dB",
    }
    back = MetricDescription.from_definition(definition)
    assert (back.is_higher_better, back.min_value, back.max_value, back.unit) == (True, 0.0, 1.0, "dB")
    none = MetricDescription(is_higher_better=None).to_definition()
    assert none["direction"] == "none" and none["display"] == {"scale": "none"}
    assert MetricDescription.from_definition(none).to_dict() == {}


def test_training_contract_is_a_definition_view():
    payload = DOCUMENTS["declarations"]["n_ok"]
    view = training_contract(payload, "depth.train")
    assert view["metricSet"]["metricNamespace"] == "depth.train"
    assert set(view["metrics"]) == {"total", "depth_mae"}
    assert view["axes"] == {"kind": {"role": "kind"}, "stage": {"role": "stage"}}
    assert training_contract(payload, "camera.train") is None


def test_value_problem():
    assert metric_value_problem(0.5) is None
    assert metric_value_problem(3) is None
    for value in (True, "0.5", None, float("nan"), float("inf")):
        assert metric_value_problem(value).code == "not_a_number"


def test_evaluation_ids_exclude_training_vocabularies():
    ids = evaluation_metric_ids()
    assert "depth.absrel" in ids and "rgb.psnr" in ids
    assert not any(item.startswith("train.") for item in ids)


def test_builder_describe_snapshots_without_a_value():
    builder = ProtocolBuilder("depth.dense", ["euler.core", "euler.depth"])
    builder.describe("spatial.method").describe("domain.region")
    builder.param("spatial.method", "none")
    builder.describe("spatial.method")  # already described: kept
    protocol = builder.build()
    assert "domain.region" in protocol["descriptors"]
    assert "domain.region" not in protocol.get("observations", {})
    assert protocol["descriptors"]["spatial.method"]["vocabulary"] == "euler.core@1.0"
    with pytest.raises(KeyError):
        builder.describe("lab.unknown")


def test_loading_a_pass_does_not_mutate_its_documents():
    document = copy.deepcopy(DOCUMENTS["sets"]["v_cerema"])
    record = copy.deepcopy(DOCUMENTS["passes"]["p_subset"]["record"])
    before = (copy.deepcopy(document), copy.deepcopy(record))
    load_validation_pass(document, record)
    validation_pass_problems(document, record)
    assert (document, record) == before


def test_unreadable_pass_ids_make_the_population_unknown():
    loaded = _load("pass:p_duplicate_ids")
    assert loaded.item_ids is None
    verdict = compare_files(loaded, _load("eval:d1.json"),
                            "depth.eval.metric.standard.image_mean.absrel")
    assert verdict.label == "not comparable · evaluated files not recorded"
