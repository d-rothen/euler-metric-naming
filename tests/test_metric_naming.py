"""Tests for the euler-metric-naming package."""
from __future__ import annotations

import pytest

from euler_metric_naming import (
    AxisDeclaration,
    DecomposedMetric,
    MetricDescription,
    MetricNamespace,
    compare_stages,
    decompose,
    filter_glob,
    filter_kind,
    recompose,
    validate_metric_name,
)


# ═══════════════════════════════════════════════════════════════════════════
#  MetricNamespace — construction
# ══��════════════════════════════════════════════════════════════════════════


class TestMetricNamespaceInit:
    def test_basic_construction(self):
        ns = MetricNamespace(
            producer="euler_train.weather_metric",
            producer_version="0.1.0",
            modalities=("depth", "rgb"),
        )
        assert ns.producer == "euler_train.weather_metric"
        assert ns.producer_version == "0.1.0"
        assert ns.modalities == ("depth", "rgb")
        assert ns.stages is None
        assert ns.context == "train"

    def test_with_stages(self):
        ns = MetricNamespace(
            producer="euler_train.weather_metric",
            producer_version="0.1.0",
            modalities=("depth",),
            stages=("prior", "final"),
        )
        assert ns.stages == ("prior", "final")

    def test_eval_context(self):
        ns = MetricNamespace(
            producer="euler_eval.depth",
            producer_version="2.0.0",
            modalities=("depth",),
            context="eval",
        )
        assert ns.context == "eval"

    def test_invalid_context(self):
        with pytest.raises(ValueError, match="context must be"):
            MetricNamespace(
                producer="test",
                producer_version="0.1.0",
                modalities=("depth",),
                context="inference",
            )

    def test_reserved_scope_as_modality(self):
        with pytest.raises(ValueError, match="reserved scope prefix"):
            MetricNamespace(
                producer="test",
                producer_version="0.1.0",
                modalities=("sys",),
            )

    def test_invalid_modality_name(self):
        with pytest.raises(ValueError, match="must match"):
            MetricNamespace(
                producer="test",
                producer_version="0.1.0",
                modalities=("Depth",),
            )

    def test_invalid_stage_name(self):
        with pytest.raises(ValueError, match="must match"):
            MetricNamespace(
                producer="test",
                producer_version="0.1.0",
                modalities=("depth",),
                stages=("Prior-Stage",),
            )

    def test_accepts_list_inputs(self):
        ns = MetricNamespace(
            producer="test",
            producer_version="0.1.0",
            modalities=["depth", "rgb"],
            stages=["prior", "final"],
        )
        assert ns.modalities == ("depth", "rgb")
        assert ns.stages == ("prior", "final")


# ═══════════════════════════════════════════════════════════════════════════
#  MetricNamespace — key building (with stages)
# ═══════════════════════════════════════════════════════════════════════════


class TestKeyBuildingWithStages:
    @pytest.fixture()
    def ns(self):
        return MetricNamespace(
            producer="euler_train.weather_metric",
            producer_version="0.1.0",
            modalities=("depth", "rgb", "rays"),
            stages=("prior", "pred", "structure", "consolidation", "final",
                    "weather", "ray", "flag", "data"),
        )

    def test_loss_with_stage(self, ns):
        assert ns.loss("depth", "prior", "log_radius") == "depth.train.loss.prior.log_radius"

    def test_loss_final(self, ns):
        assert ns.loss("depth", "final", "log_radius") == "depth.train.loss.final.log_radius"

    def test_loss_final_depth(self, ns):
        assert ns.loss("depth", "final", "depth") == "depth.train.loss.final.depth"

    def test_loss_no_stage(self, ns):
        assert ns.loss("depth", metric="total") == "depth.train.loss.total"

    def test_diag_with_stage(self, ns):
        assert ns.diag("depth", "prior", "depth_mae") == "depth.train.diag.prior.depth_mae"

    def test_diag_final(self, ns):
        assert ns.diag("depth", "final", "depth_mae") == "depth.train.diag.final.depth_mae"

    def test_diag_consolidation(self, ns):
        assert ns.diag("depth", "consolidation", "delta_vs_prior") == "depth.train.diag.consolidation.delta_vs_prior"

    def test_stat_with_stage(self, ns):
        assert ns.stat("depth", "structure", "confidence") == "depth.train.stat.structure.confidence"

    def test_stat_flag(self, ns):
        assert ns.stat("depth", "flag", "refiner_enabled") == "depth.train.stat.flag.refiner_enabled"

    def test_stat_data(self, ns):
        assert ns.stat("depth", "data", "valid_fraction") == "depth.train.stat.data.valid_fraction"

    def test_rgb_loss(self, ns):
        assert ns.loss("rgb", "weather", "dehaze") == "rgb.train.loss.weather.dehaze"
        assert ns.loss("rgb", "weather", "visibility") == "rgb.train.loss.weather.visibility"
        assert ns.loss("rgb", "weather", "smoothness") == "rgb.train.loss.weather.smoothness"

    def test_rays_loss(self, ns):
        assert ns.loss("rays", "ray", "l1") == "rays.train.loss.ray.l1"
        assert ns.loss("rays", "ray", "cosine") == "rays.train.loss.ray.cosine"

    def test_sys_metric(self, ns):
        assert ns.sys("lr") == "sys.train.lr"

    def test_sys_metric_with_extra(self, ns):
        assert ns.sys("lr", "geometry_encoder") == "sys.train.lr.geometry_encoder"

    def test_sys_grad_norm(self, ns):
        assert ns.sys("grad_norm") == "sys.train.grad_norm"

    def test_unknown_modality(self, ns):
        with pytest.raises(ValueError, match="unknown modality"):
            ns.loss("video", "prior", "x")

    def test_unknown_stage(self, ns):
        with pytest.raises(ValueError, match="unknown stage"):
            ns.loss("depth", "nonexistent_stage", "x")

    def test_sys_requires_args(self, ns):
        with pytest.raises(TypeError, match="at least one"):
            ns.sys()

    def test_both_positional_and_keyword(self, ns):
        with pytest.raises(TypeError, match="not both"):
            ns.loss("depth", "prior", "x", metric="y")


# ═══════════════════════════════════════════════════════════════════════════
#  MetricNamespace — key building (without stages)
# ═══════════════════════════════════════════════════════════════════════════


class TestKeyBuildingWithoutStages:
    @pytest.fixture()
    def ns(self):
        return MetricNamespace(
            producer="euler_train.simple_baseline",
            producer_version="0.1.0",
            modalities=("depth",),
        )

    def test_loss_no_stage(self, ns):
        assert ns.loss("depth", metric="mae") == "depth.train.loss.mae"

    def test_loss_total(self, ns):
        assert ns.loss("depth", metric="total") == "depth.train.loss.total"

    def test_diag_no_stage(self, ns):
        assert ns.diag("depth", metric="rmse") == "depth.train.diag.rmse"

    def test_stage_not_allowed(self, ns):
        with pytest.raises(ValueError, match="stages not declared"):
            ns.loss("depth", "prior", "mae")

    def test_sys_still_works(self, ns):
        assert ns.sys("lr") == "sys.train.lr"


# ═══════════════════════════════════════════════════════════════════════════
#  MetricNamespace — eval context
# ═══════════════════════════════════════════════════════════════════════════


class TestEvalContext:
    @pytest.fixture()
    def ns(self):
        return MetricNamespace(
            producer="euler_eval.depth",
            producer_version="2.0.0",
            modalities=("depth",),
            context="eval",
        )

    def test_loss_not_available(self, ns):
        with pytest.raises(TypeError, match="only available in 'train'"):
            ns.loss("depth", metric="mae")

    def test_sys_uses_eval_context(self, ns):
        assert ns.sys("lr") == "sys.eval.lr"


# ═══════════════════════════════════════════════════════════════════════════
#  MetricNamespace — envelope generation
# ═══════════════════════════════════════════════════════════════════════════


class TestEnvelopeGeneration:
    def test_metric_set_envelope(self):
        ns = MetricNamespace(
            producer="euler_eval.depth",
            producer_version="2.0.0",
            modalities=("depth",),
            context="eval",
            descriptions={
                "absrel": MetricDescription(is_higher_better=False, scale="linear"),
            },
        )
        envelope = ns.metric_set_envelope("depth")
        assert envelope["metricNamespace"] == "depth.eval"
        assert envelope["producerKey"] == "euler_eval.depth"
        assert envelope["producerVersion"] == "2.0.0"
        assert envelope["sourceKind"] == "computed"
        assert "axes" in envelope
        assert "metricDescriptions" in envelope
        assert envelope["metricDescriptions"]["absrel"]["isHigherBetter"] is False

    def test_training_naming_config(self):
        ns = MetricNamespace(
            producer="euler_train.weather_metric",
            producer_version="0.1.0",
            modalities=("depth", "rgb"),
            stages=("prior", "final"),
            descriptions={
                "log_radius": MetricDescription(is_higher_better=False, scale="log"),
            },
        )
        config = ns.training_naming_config()
        assert config["producer_key"] == "euler_train.weather_metric"
        assert config["producer_version"] == "0.1.0"
        assert "depth.train" in config["namespaces"]
        assert "rgb.train" in config["namespaces"]
        assert "sys.train" in config["namespaces"]

        depth_ns = config["namespaces"]["depth.train"]
        assert "kind" in depth_ns["axes"]
        assert depth_ns["axes"]["kind"]["values"] == ["loss", "diag", "stat"]
        assert "stage" in depth_ns["axes"]
        assert depth_ns["axes"]["stage"]["values"] == ["prior", "final"]
        assert depth_ns["axes"]["stage"]["optional"] is True
        assert "metricDescriptions" in depth_ns

        sys_ns = config["namespaces"]["sys.train"]
        assert sys_ns["axes"] == {}

    def test_training_naming_config_no_stages(self):
        ns = MetricNamespace(
            producer="euler_train.simple",
            producer_version="0.1.0",
            modalities=("depth",),
        )
        config = ns.training_naming_config()
        depth_ns = config["namespaces"]["depth.train"]
        assert "kind" in depth_ns["axes"]
        assert "stage" not in depth_ns["axes"]

    def test_sys_included_for_train_context(self):
        ns = MetricNamespace(
            producer="test",
            producer_version="0.1.0",
            modalities=("depth",),
        )
        config = ns.training_naming_config()
        assert "sys.train" in config["namespaces"]


# ═══════════════════════════════════════════════════════════════════════════
#  AxisDeclaration
# ═══════════════════════════════════════════════════════════════════════════


class TestAxisDeclaration:
    def test_to_dict(self):
        axis = AxisDeclaration(
            position=0,
            values=("loss", "diag", "stat"),
            optional=False,
            description="Metric kind",
        )
        d = axis.to_dict()
        assert d == {
            "position": 0,
            "optional": False,
            "values": ["loss", "diag", "stat"],
            "description": "Metric kind",
        }

    def test_from_dict(self):
        data = {
            "position": 1,
            "optional": True,
            "values": ["prior", "final"],
        }
        axis = AxisDeclaration.from_dict(data)
        assert axis.position == 1
        assert axis.optional is True
        assert axis.values == ("prior", "final")
        assert axis.description is None

    def test_roundtrip(self):
        axis = AxisDeclaration(
            position=0,
            values=("a", "b"),
            optional=True,
            description="test",
        )
        assert AxisDeclaration.from_dict(axis.to_dict()) == axis


# ═══════════════════════════════════════════════════════════════════════════
#  Decompose / Recompose
# ═══════════════════════════════════════════════════════════════════════════


class TestDecompose:
    @pytest.fixture()
    def train_axes(self):
        return {
            "kind": AxisDeclaration(position=0, values=("loss", "diag", "stat"), optional=False),
            "stage": AxisDeclaration(position=1, values=("prior", "pred", "structure", "consolidation", "final"), optional=True),
        }

    def test_full_decompose(self, train_axes):
        result = decompose("depth.train.loss.prior.log_radius", "depth.train", train_axes)
        assert result.namespace == "depth.train"
        assert result.axes == {"kind": "loss", "stage": "prior"}
        assert result.metric == "log_radius"

    def test_no_stage(self, train_axes):
        result = decompose("depth.train.loss.total", "depth.train", train_axes)
        assert result.axes == {"kind": "loss", "stage": None}
        assert result.metric == "total"

    def test_stat_flag_ambiguity(self):
        """When consuming an axis value would leave 0 segments, skip it.

        'flag' could match stage values, but since there's nothing after
        it for the metric name, it must be treated as the metric instead.
        """
        axes = {
            "kind": AxisDeclaration(position=0, values=("loss", "diag", "stat"), optional=False),
            "stage": AxisDeclaration(position=1, values=("flag",), optional=True),
        }
        result = decompose("depth.train.stat.flag", "depth.train", axes)
        assert result.axes == {"kind": "stat", "stage": None}
        assert result.metric == "flag"

    def test_multi_segment_metric(self, train_axes):
        result = decompose("depth.train.diag.prior.depth_mae_m", "depth.train", train_axes)
        assert result.axes == {"kind": "diag", "stage": "prior"}
        assert result.metric == "depth_mae_m"

    def test_wrong_namespace(self, train_axes):
        with pytest.raises(ValueError, match="does not belong"):
            decompose("rgb.train.loss.x", "depth.train", train_axes)

    def test_required_axis_missing(self):
        axes = {
            "kind": AxisDeclaration(position=0, values=("loss",), optional=False),
        }
        with pytest.raises(ValueError, match="required axis"):
            decompose("depth.train.xyz.foo", "depth.train", axes)

    def test_no_axes(self):
        result = decompose("sys.train.lr", "sys.train", {})
        assert result.namespace == "sys.train"
        assert result.axes == {}
        assert result.metric == "lr"

    def test_no_axes_multi_segment(self):
        result = decompose("sys.train.lr.geometry_encoder", "sys.train", {})
        assert result.metric == "lr.geometry_encoder"

    def test_empty_after_namespace(self, train_axes):
        with pytest.raises(ValueError, match="no segments"):
            decompose("depth.train.", "depth.train", train_axes)


class TestRecompose:
    def test_basic(self):
        axes = {
            "kind": AxisDeclaration(position=0, values=("loss",), optional=False),
            "stage": AxisDeclaration(position=1, values=("prior",), optional=True),
        }
        result = recompose(
            "depth.train",
            axes,
            {"kind": "loss", "stage": "prior"},
            "log_radius",
        )
        assert result == "depth.train.loss.prior.log_radius"

    def test_optional_axis_none(self):
        axes = {
            "kind": AxisDeclaration(position=0, values=("loss",), optional=False),
            "stage": AxisDeclaration(position=1, values=("prior",), optional=True),
        }
        result = recompose(
            "depth.train",
            axes,
            {"kind": "loss", "stage": None},
            "total",
        )
        assert result == "depth.train.loss.total"

    def test_invalid_axis_value(self):
        axes = {
            "kind": AxisDeclaration(position=0, values=("loss",), optional=False),
        }
        with pytest.raises(ValueError, match="not in"):
            recompose("depth.train", axes, {"kind": "bad"}, "x")

    def test_required_axis_missing(self):
        axes = {
            "kind": AxisDeclaration(position=0, values=("loss",), optional=False),
        }
        with pytest.raises(ValueError, match="has no value"):
            recompose("depth.train", axes, {"kind": None}, "x")


# ═══════════════════════════════════════════════════════════════════════════
#  MetricDescription
# ═══════════════════════════════════════════════════════════════════════════


class TestMetricDescription:
    def test_to_dict(self):
        desc = MetricDescription(
            is_higher_better=False,
            scale="log",
            display_name="Log-Radius L1",
        )
        d = desc.to_dict()
        assert d == {
            "isHigherBetter": False,
            "scale": "log",
            "displayName": "Log-Radius L1",
        }

    def test_to_dict_omits_none(self):
        desc = MetricDescription()
        assert desc.to_dict() == {}

    def test_from_dict(self):
        data = {
            "isHigherBetter": True,
            "min": 0.0,
            "max": 1.0,
            "scale": "linear",
            "unit": "meters",
            "formatHint": ".4f",
            "displayName": "Depth MAE",
            "description": "Mean absolute error",
        }
        desc = MetricDescription.from_dict(data)
        assert desc.is_higher_better is True
        assert desc.min_value == 0.0
        assert desc.max_value == 1.0
        assert desc.scale == "linear"
        assert desc.unit == "meters"
        assert desc.format_hint == ".4f"
        assert desc.display_name == "Depth MAE"
        assert desc.description == "Mean absolute error"

    def test_roundtrip(self):
        desc = MetricDescription(
            is_higher_better=False,
            min_value=0.0,
            max_value=100.0,
            scale="percentage",
            unit="dB",
            format_hint=".2f",
            display_name="PSNR",
            description="Peak signal-to-noise ratio",
        )
        assert MetricDescription.from_dict(desc.to_dict()).to_dict() == desc.to_dict()

    def test_invalid_scale(self):
        with pytest.raises(ValueError, match="scale must be one of"):
            MetricDescription(scale="invalid")


# ═══════════════════════════════════════════════════════════════════════════
#  validate_metric_name
# ═══════════════════════════════════════════════════════════════════════════


class TestValidateMetricName:
    def test_valid(self):
        validate_metric_name("depth.train.loss.prior.log_radius")

    def test_valid_sys(self):
        validate_metric_name("sys.train.lr")

    def test_too_few_segments(self):
        with pytest.raises(ValueError, match="at least 2"):
            validate_metric_name("loss")

    def test_first_segment_underscore(self):
        with pytest.raises(ValueError, match="first segment"):
            validate_metric_name("_depth.train.loss")

    def test_uppercase(self):
        with pytest.raises(ValueError, match="must match"):
            validate_metric_name("Depth.train.loss")

    def test_hyphen(self):
        with pytest.raises(ValueError, match="must match"):
            validate_metric_name("depth.train.my-loss")


# ═══════════════════════════════════════════════════════════════════════════
#  Matching utilities
# ═══════════════════════════════════════════════════════════════════════════


class TestCompareStages:
    def test_basic(self):
        ns = MetricNamespace(
            producer="test",
            producer_version="0.1.0",
            modalities=("depth",),
            stages=("prior", "pred", "final"),
        )
        metrics = {
            "depth.train.diag.prior.depth_mae": 0.42,
            "depth.train.diag.pred.depth_mae": 0.31,
            "depth.train.diag.final.depth_mae": 0.28,
            "depth.train.loss.total": 1.23,
            "sys.train.lr": 3e-5,
        }
        result = compare_stages(metrics, ns, "depth", "depth_mae")
        assert result == {"prior": 0.42, "pred": 0.31, "final": 0.28}

    def test_includes_stageless(self):
        ns = MetricNamespace(
            producer="test",
            producer_version="0.1.0",
            modalities=("depth",),
            stages=("prior", "final"),
        )
        metrics = {
            "depth.train.loss.prior.log_radius": 0.15,
            "depth.train.loss.total": 1.23,
        }
        result = compare_stages(metrics, ns, "depth", "total")
        assert result == {None: 1.23}


class TestFilterKind:
    def test_basic(self):
        ns = MetricNamespace(
            producer="test",
            producer_version="0.1.0",
            modalities=("depth",),
            stages=("prior", "final"),
        )
        metrics = {
            "depth.train.loss.prior.log_radius": 0.15,
            "depth.train.loss.total": 1.23,
            "depth.train.diag.prior.depth_mae": 0.42,
            "depth.train.stat.flag.refiner_enabled": 1.0,
            "sys.train.lr": 3e-5,
        }
        result = filter_kind(metrics, ns, "depth", "loss")
        assert result == {
            "depth.train.loss.prior.log_radius": 0.15,
            "depth.train.loss.total": 1.23,
        }


class TestFilterGlob:
    def test_wildcard(self):
        metrics = {
            "depth.train.loss.prior.log_radius": 0.15,
            "depth.train.loss.final.log_radius": 0.09,
            "depth.train.diag.prior.depth_mae": 0.42,
            "rgb.train.loss.weather.dehaze": 0.03,
        }
        result = filter_glob(metrics, "depth.train.*.prior.*")
        assert result == {
            "depth.train.loss.prior.log_radius": 0.15,
            "depth.train.diag.prior.depth_mae": 0.42,
        }

    def test_no_match(self):
        metrics = {"depth.train.loss.total": 1.0}
        assert filter_glob(metrics, "rgb.*") == {}


# ═══════════════════════════════════════════════════════════════════════════
#  Full integration: metric-diffusion example from spec §11
# ═══════════════════════════════════════════════════════════════════════════


class TestFullIntegration:
    @pytest.fixture()
    def ns(self):
        return MetricNamespace(
            producer="euler_train.weather_metric",
            producer_version="0.1.0",
            modalities=("depth", "rgb", "rays"),
            stages=(
                "prior", "pred", "structure", "consolidation", "final",
                "weather", "ray", "flag", "data",
            ),
            descriptions={
                "log_radius": MetricDescription(is_higher_better=False, scale="log"),
                "depth_mae": MetricDescription(is_higher_better=False, unit="meters"),
                "confidence": MetricDescription(
                    is_higher_better=True,
                    scale="linear",
                    min_value=0.0,
                    max_value=1.0,
                ),
            },
        )

    def test_loss_dict_construction(self, ns):
        loss_dict = {}
        loss_dict[ns.loss("depth", "prior", "log_radius")] = 0.152
        loss_dict[ns.loss("depth", "final", "log_radius")] = 0.089
        loss_dict[ns.loss("depth", "pred", "log_radius")] = 0.120
        loss_dict[ns.loss("depth", "final", "depth")] = 0.045
        loss_dict[ns.loss("depth", "prior", "confidence")] = 0.010
        loss_dict[ns.loss("depth", "structure", "depth")] = 0.030
        loss_dict[ns.loss("depth", "consolidation", "structure_align")] = 0.005
        loss_dict[ns.loss("depth", metric="total")] = 1.234
        loss_dict[ns.loss("rgb", "weather", "dehaze")] = 0.031
        loss_dict[ns.loss("rgb", "weather", "visibility")] = 0.022

        assert loss_dict["depth.train.loss.prior.log_radius"] == 0.152
        assert loss_dict["depth.train.loss.total"] == 1.234
        assert loss_dict["rgb.train.loss.weather.dehaze"] == 0.031

    def test_diagnostic_metrics(self, ns):
        metrics = {}
        metrics[ns.diag("depth", "prior", "log_radius_l1")] = 0.200
        metrics[ns.diag("depth", "final", "depth_mae")] = 0.283
        metrics[ns.stat("depth", "structure", "confidence")] = 0.72
        metrics[ns.stat("depth", "consolidation", "anchor_strength")] = 0.85
        metrics[ns.stat("depth", "flag", "refiner_enabled")] = 1.0

        assert metrics["depth.train.diag.prior.log_radius_l1"] == 0.200
        assert metrics["depth.train.stat.structure.confidence"] == 0.72

    def test_full_record(self, ns):
        """Simulate a full training step record as in spec §11."""
        record = {
            "step": 500,
            "epoch": 3,
            "wall_time": 1704067842.0,
            "elapsed_sec": 120.5,
            ns.loss("depth", "prior", "log_radius"): 0.152,
            ns.loss("depth", "final", "log_radius"): 0.089,
            ns.loss("depth", metric="total"): 1.234,
            ns.diag("depth", "prior", "depth_mae"): 0.421,
            ns.diag("depth", "final", "depth_mae"): 0.283,
            ns.stat("depth", "structure", "confidence"): 0.72,
            ns.stat("depth", "flag", "refiner_enabled"): 1.0,
            ns.loss("rgb", "weather", "dehaze"): 0.031,
            ns.sys("lr"): 3e-05,
        }

        assert record["depth.train.loss.prior.log_radius"] == 0.152
        assert record["sys.train.lr"] == 3e-05

        # Compare stages
        stages = compare_stages(record, ns, "depth", "depth_mae")
        assert stages == {"prior": 0.421, "final": 0.283}

        improvement = stages["prior"] - stages["final"]
        assert abs(improvement - 0.138) < 1e-6

    def test_training_naming_config_full(self, ns):
        # Trigger sys usage
        ns.sys("lr")

        config = ns.training_naming_config()
        assert config["producer_key"] == "euler_train.weather_metric"

        namespaces = config["namespaces"]
        assert set(namespaces.keys()) == {
            "depth.train", "rgb.train", "rays.train", "sys.train",
        }

        depth_ns = namespaces["depth.train"]
        assert depth_ns["axes"]["kind"]["values"] == ["loss", "diag", "stat"]
        assert depth_ns["axes"]["kind"]["optional"] is False
        assert set(depth_ns["axes"]["stage"]["values"]) == {
            "prior", "pred", "structure", "consolidation", "final",
            "weather", "ray", "flag", "data",
        }
        assert depth_ns["axes"]["stage"]["optional"] is True

        assert "metricDescriptions" in depth_ns
        assert depth_ns["metricDescriptions"]["log_radius"]["isHigherBetter"] is False
        assert depth_ns["metricDescriptions"]["log_radius"]["scale"] == "log"

        sys_ns = namespaces["sys.train"]
        assert sys_ns["axes"] == {}

    def test_decompose_integration(self, ns):
        axes = ns.axes("depth")
        dec = decompose("depth.train.loss.prior.log_radius", "depth.train", axes)
        assert dec.axes["kind"] == "loss"
        assert dec.axes["stage"] == "prior"
        assert dec.metric == "log_radius"

        dec2 = decompose("depth.train.loss.total", "depth.train", axes)
        assert dec2.axes["kind"] == "loss"
        assert dec2.axes["stage"] is None
        assert dec2.metric == "total"
