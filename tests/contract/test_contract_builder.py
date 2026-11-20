"""ProtocolBuilder, validation, the engine's finer rules, and package plumbing."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from euler_metric_naming import AxisDeclaration
from euler_metric_naming.contract import (
    ProtocolBuilder,
    ProtocolValidationError,
    VocabularySet,
    check_protocol,
    compare,
    effective_compare,
    hash_value,
    manifest_hash,
    normalize,
    per_item_name,
    protocol_problems,
    recompute,
    sqrt_stratification,
)

AXES = {
    "space": AxisDeclaration(0, ("native", "metric")),
    "category": AxisDeclaration(1, ("standard", "image_quality"), True),
    "reduction": AxisDeclaration(2, ("image_mean", "pixel_pool"), True),
}
METRIC_SET = {"metricNamespace": "depth.eval", "axes": {k: v.to_dict() for k, v in AXES.items()}}


def _builder():
    builder = ProtocolBuilder("depth.dense", ["euler.core", "euler.sky", "euler.depth"])
    builder.reference("gt", identity=hash_value("gt"))
    builder.axis("space", "space").axis("category", "family").axis("reduction", "reduction")
    return builder


def test_builder_snapshots_vocabulary_terms():
    builder = _builder()
    builder.param("depth.cap.max_m", 38).observe("depth.cap.clipped", 0.0)
    builder.population(["b", "a", "é"], unit="image")
    builder.metrics_for(
        ["depth.eval.metric.standard.image_mean.absrel", "depth.eval.metric.standard.pixel_pool.rmse"],
        "depth.eval",
        AXES,
    )
    protocol = builder.build()
    assert protocol["schemaVersion"] == "2.0"
    assert protocol["vocabularies"] == {"euler.core": "1.0", "euler.depth": "1.0"}
    assert protocol["descriptors"]["depth.cap.max_m"]["vocabulary"] == "euler.depth@1.0"
    assert protocol["descriptors"]["reference.gt"]["label"] == "GT"
    assert set(protocol["metrics"]) == {"absrel", "rmse"}
    assert protocol["metrics"]["absrel"]["vocabulary"] == "euler.depth@1.0"
    assert set(protocol["reducers"]) == {"image_mean", "pixel_pool"}
    assert protocol["population"]["count"] == 3
    assert protocol["population"]["manifest"] == manifest_hash(["a", "b", "é"])
    assert protocol["items"]["ids"] == ["b", "a", "é"]
    assert protocol_problems(protocol) == []


def test_builder_rejects_unknown_terms_without_descriptor():
    builder = _builder()
    with pytest.raises(KeyError, match="pass descriptor"):
        builder.param("depth.unknown.knob", 1)
    builder.param("depth.unknown.knob", 1, descriptor={"label": "knob", "role": "definition"})
    with pytest.raises(KeyError):
        builder.metric("not_a_metric")
    with pytest.raises(KeyError):
        builder.reducer("not_a_reducer")
    with pytest.raises(ProtocolValidationError):
        builder.param("depth.other", 1, descriptor={"role": "tuning"})


def test_builder_enforces_v1_parameter_shapes_and_namespaces():
    builder = _builder()
    with pytest.raises(ValueError):
        builder.param("depth.cap.max_m", {"nested": 1})
    with pytest.raises(ValueError):
        builder.param("depth.cap.max_m", list(range(65)))
    with pytest.raises(ValueError):
        builder.param("depth.cap.max_m", "x" * 1025)
    builder.param("depth.cap.max_m", (1, 2))
    with pytest.raises(ValueError, match="already a parameter"):
        builder.observe("depth.cap.max_m", 1)


def test_builder_population_and_series():
    builder = _builder()
    with pytest.raises(ValueError):
        builder.series("x", [1])
    with pytest.raises(ValueError, match="unique"):
        builder.population(["é", "é"])
    builder.population(["a", "b"])
    with pytest.raises(ValueError, match="2 items"):
        builder.series("depth.eval.metric.standard.count", [1])
    builder.series("depth.eval.metric.standard.count", [10, None])
    assert builder.build()["items"]["series"] == {"depth.eval.metric.standard.count": [10, None]}


def test_builder_axes_and_stratifications():
    builder = _builder()
    with pytest.raises(ValueError):
        builder.axis("bin", "stratum")
    with pytest.raises(ValueError):
        builder.axis("bin", "bucket")
    builder.axis("bin", "stratum", stratification="depth.gt_range")
    with pytest.raises(ProtocolValidationError, match="stratification"):
        builder.build()
    builder.stratification("depth.gt_range", sqrt_stratification(0.01, 80))
    with pytest.raises(ProtocolValidationError, match="scheme"):
        builder.stratification("other", {**sqrt_stratification(0.01, 80), "scheme": "nope"})
    with pytest.raises(ProtocolValidationError, match="ascending"):
        builder.stratification("bad", {"edges": [1, 1], "labels": ["x"]})
    assert builder.build()["axes"]["bin"]["stratification"] == "depth.gt_range"


def test_vocabulary_set_ambiguity():
    both = VocabularySet(["euler.depth", "euler.rgb"])
    with pytest.raises(ValueError, match="ambiguous"):
        both.metric("psnr")
    assert both.metric("absrel")["id"] == "depth.absrel"
    assert both.descriptor("rgb.fid.backend")["vocabulary"] == "euler.rgb@1.0"


def test_check_protocol_reports_closed_set_violations():
    protocol = {
        "schemaVersion": "2.0",
        "parameters": {"a.b": 1},
        "observations": {"a.b": 2},
        "descriptors": {"a.b": {"role": "domain"}},
        "metrics": {"m": {"kind": "gauge", "pool": {"form": "mean"}, "rev": 0}},
        "reducers": {"r": {"recompute": "sum"}},
        "axes": {"x": {"role": "space"}, "y": {"role": "space"}},
        "items": {"ids": ["a", "a"], "series": {"s": [1]}},
    }
    with pytest.raises(ProtocolValidationError) as raised:
        check_protocol(protocol)
    text = str(raised.value)
    for fragment in (
        "both a parameter and an observation",
        "observe is required",
        "kind must be",
        "pool.weight is required",
        "rev must be",
        "recompute must be",
        "both have role 'space'",
        "items.ids must be unique",
        "one value per id",
    ):
        assert fragment in text, fragment


def test_effective_compare_rules():
    assert effective_compare(None) == "equal"
    assert effective_compare({"role": "provenance"}) == "ignore"
    assert effective_compare({"role": "population"}) == "pair"
    assert effective_compare({"role": "domain"}) == "measure"
    assert effective_compare({"role": "domain", "compare": "equal"}) == "equal"
    assert effective_compare({"role": "definition", "compare": "fuzzy"}) == "equal"
    assert effective_compare({"role": "tuning", "compare": "ignore"}) == "equal"
    assert effective_compare({"compare": "ignore"}) == "equal"


def _contract(**protocol):
    base = {"schemaVersion": "2.0", "parameters": {}}
    base.update(protocol)
    return normalize(base, METRIC_SET)


CELL = "depth.eval.metric.standard.image_mean.absrel"


def test_merged_descriptor_strictest_compare_wins():
    a = _contract(parameters={"k.x": 1}, descriptors={"k.x": {"role": "provenance"}})
    b = _contract(parameters={"k.x": 2}, descriptors={"k.x": {"role": "definition", "label": "K"}})
    verdict = compare(a, b, CELL)
    assert verdict.label == "not comparable · K differs"
    assert compare(b, a, CELL).label == "not comparable · K differs"


def test_ignore_and_pair_keys_need_no_value_on_both_sides():
    a = _contract(parameters={"core.seed": 1, "pop.split": "x"},
                  descriptors={"core.seed": {"role": "provenance"}, "pop.split": {"role": "population"}})
    b = _contract()
    assert compare(a, b, CELL).codes == ["population_unknown"]


def test_descriptor_default_fills_a_missing_value():
    descriptor = {"role": "definition", "label": "K", "default": "drop"}
    a = _contract(parameters={"k.x": "drop"}, descriptors={"k.x": descriptor})
    b = _contract()
    assert compare(a, b, CELL).codes == ["population_unknown"]
    c = _contract(parameters={"k.x": "keep"}, descriptors={"k.x": descriptor})
    assert compare(c, b, CELL).label == "not comparable · K differs"


def test_unmeasurable_domain_is_incomparable():
    descriptor = {"role": "domain", "observe": "factor", "label": "F"}
    a = _contract(observations={"k.f": [1, 1]}, descriptors={"k.f": descriptor})
    b = _contract(observations={"k.f": "big"}, descriptors={"k.f": descriptor})
    verdict = compare(a, b, CELL)
    assert verdict.label == "not comparable · F differs"
    assert verdict.reasons[0].code == "parameter"


def test_requires_limits_reference_relevance():
    definition = {"rev": 1, "requires": ["gt"]}
    ref = {"identity": "sha256:" + "0" * 64}
    a = _contract(references={"gt": ref, "calibration": ref}, metrics={"absrel": definition})
    b = _contract(references={"gt": ref}, metrics={"absrel": definition})
    assert compare(a, b, CELL).codes == ["population_unknown"]
    c = _contract(references={"gt": ref, "calibration": ref}, metrics={"absrel": {"rev": 1}})
    d = _contract(references={"gt": ref}, metrics={"absrel": {"rev": 1}})
    assert compare(c, d, CELL).codes == ["not_recorded", "population_unknown"]


def test_same_region_with_different_identity_is_a_reference_difference():
    view = {"source": "cam", "region": [0, 0, 10, 10]}
    a = _contract(references={"gt": {"identity": "x", "view": view}})
    b = _contract(references={"gt": {"identity": "y", "view": copy.deepcopy(view)}})
    assert compare(a, b, CELL).label == "not comparable · reference.gt differs"


def test_population_without_items_uses_manifests_only():
    manifest = manifest_hash(["a"])
    a = _contract(population={"manifest": manifest})
    b = _contract(population={"manifest": manifest})
    assert compare(a, b, CELL).label == "same protocol"
    c = _contract(population={"manifest": manifest_hash(["b"])})
    assert compare(a, c, CELL).label == "not comparable · evaluated files not recorded"
    assert compare(a, c, CELL, {"a"}, {"a", "b"}).tier == "comparable"


def test_cell_without_reduction_is_not_pairable():
    a = _contract()
    b = _contract()
    name = "depth.eval.metric.image_quality.psnr"
    per_item = {"depth.eval.metric.image_quality.psnr"}
    verdict = compare(a, b, name, {"x", "y"}, {"x"}, per_item, per_item)
    assert verdict.codes == ["population_partial"]


def test_per_item_name_removes_the_reduction_segment():
    contract = _contract(axes={"reduction": {"role": "reduction"}})
    assert per_item_name(contract, CELL) == "depth.eval.metric.standard.absrel"


def test_recompute_operations():
    values = {"a": 1.0, "b": 3.0, "c": None, "d": 2.0}
    weights = {"a": 1, "b": 3, "c": 5, "d": 0}
    assert recompute("mean", values) == 2.0
    assert recompute("median", values) == 2.0
    assert recompute("median", {"a": 1, "b": 4}) == 2.5
    assert recompute("pool", values, weights=weights, form="mean") == 2.5
    assert recompute("pool", values, weights=weights, form="rms") == pytest.approx((28 / 4) ** 0.5)
    assert recompute("mean", values, items={"a"}) == 1.0
    assert recompute("none", values) is None
    assert recompute("mean", {}) is None


def test_package_data_is_declared():
    pyproject = (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text()
    for pattern in ("contract/schema/*.json", "contract/vocabularies/*.json", "contract/ts/*.ts"):
        assert pattern in pyproject


def test_public_api_is_exported():
    import euler_metric_naming.contract as contract

    for name in contract.__all__:
        assert hasattr(contract, name), name
    json.dumps(contract.closed_sets())
