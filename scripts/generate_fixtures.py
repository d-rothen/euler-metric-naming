#!/usr/bin/env python3
"""Generate the conformance eval files in ``euler_metric_naming/contract/fixtures/evals``.

The files are built with :class:`ProtocolBuilder` and :class:`MetricNamespace`,
the way producers should write them. Only the eval files are generated; the
expected verdicts in ``verdicts.json`` are written by hand from the spec.

    python scripts/generate_fixtures.py          # write
    python scripts/generate_fixtures.py --check  # exit 1 when out of date
"""
from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from euler_metric_naming import (  # noqa: E402
    AxisDeclaration,
    MetricDescription,
    MetricNamespace,
)
from euler_metric_naming.contract import (  # noqa: E402
    ProtocolBuilder,
    hash_value,
    recompute,
    sqrt_stratification,
)

EVALS = ROOT / "euler_metric_naming" / "contract" / "fixtures" / "evals"
SOURCE = "cerema/rgb_left@1024x1920"
DAY = [f"day-{i}" for i in range(6)]
NIGHT = [f"night-{i}" for i in range(6)]


class _EvalNamespace(MetricNamespace):
    """MetricNamespace with explicit eval axes (as euler-eval does)."""

    def __init__(self, *, axes, **kwargs):
        super().__init__(context="eval", axes=axes, **kwargs)


def _round(value):
    return round(value, 6)


# ---------------------------------------------------------------------------- depth

DEPTH_AXES = {
    "space": AxisDeclaration(0, ("native", "metric"), False, "Depth space semantics"),
    "category": AxisDeclaration(1, ("image_quality", "standard", "depth_metrics"), True, "Metric category"),
    "reduction": AxisDeclaration(
        2, ("image_mean", "image_median", "pixel_pool"), True, "Dataset reduction mode"
    ),
    "bin": AxisDeclaration(3, ("all", "near", "mid", "far"), True, "Depth benchmark bin"),
}
LEGACY_DEPTH_AXES = {
    **DEPTH_AXES,
    "category": AxisDeclaration(
        1, ("image_quality", "standard", "depth_metrics", "geometric_metrics"), True, "Metric category"
    ),
}
DEPTH_DESCRIPTIONS = {
    "absrel": MetricDescription(is_higher_better=False, display_name="AbsRel"),
    "rmse": MetricDescription(is_higher_better=False, unit="meters", display_name="RMSE"),
    "delta1": MetricDescription(
        is_higher_better=True, min_value=0.0, max_value=1.0, display_name="δ < 1.25"
    ),
    "fid": MetricDescription(is_higher_better=False, display_name="FID"),
}
POOL_FORM = {"absrel": "mean", "rmse": "rms", "delta1": "mean"}


def _item_index(item: str) -> int:
    name, number = item.rsplit("-", 1)
    return int(number) + (10 if name == "night" else 0)


def _depth_values(items, offset):
    """Per-item values by space: {space: {item: {metric: value}}}, and weights."""
    per_space = {}
    for space, shift in (("native", 0.02), ("metric", 0.0)):
        rows = {}
        for item in items:
            k = _item_index(item)
            o = offset + shift
            rows[item] = {
                "absrel": _round(0.10 + 0.01 * k + o),
                "rmse": _round(2.0 + 0.1 * k + o * 10),
                "delta1": _round(0.90 - 0.01 * k - o),
                "near.absrel": _round(0.05 + 0.005 * k + o),
                "count": 1000 + 100 * k,
                "near.count": 300 + 10 * k,
            }
        per_space[space] = rows
    return per_space


def _depth_tree(values, fid):
    tree = {}
    for space, rows in values.items():
        standard = {"image_mean": {}, "image_median": {}, "pixel_pool": {}}
        for metric in ("absrel", "rmse", "delta1"):
            column = {item: row[metric] for item, row in rows.items()}
            weights = {item: row["count"] for item, row in rows.items()}
            standard["image_mean"][metric] = _round(recompute("mean", column))
            standard["image_median"][metric] = _round(statistics.median(column.values()))
            standard["pixel_pool"][metric] = _round(
                recompute("pool", column, weights=weights, form=POOL_FORM[metric])
            )
        near = {item: row["near.absrel"] for item, row in rows.items()}
        near_w = {item: row["near.count"] for item, row in rows.items()}
        standard["image_mean"]["near"] = {"absrel": _round(recompute("mean", near))}
        standard["pixel_pool"]["near"] = {
            "absrel": _round(recompute("pool", near, weights=near_w, form="mean"))
        }
        tree[space] = {"image_quality": {"fid": fid}, "standard": standard}
    return {"eval": tree}


def _depth_per_file(values, *, binned=True):
    files = []
    items = list(next(iter(values.values())))
    for item in items:
        spaces = {}
        for space, rows in values.items():
            row = rows[item]
            standard = {"absrel": row["absrel"], "rmse": row["rmse"], "delta1": row["delta1"]}
            if binned:
                standard["near"] = {"absrel": row["near.absrel"]}
            spaces[space] = {"standard": standard}
        files.append({"id": item, "metrics": {"depth": {"eval": spaces}}})
    return files


def _depth_metric_names(values):
    names = []
    for space in values:
        prefix = f"depth.eval.{space}"
        names.append(f"{prefix}.image_quality.fid")
        for reduction in ("image_mean", "image_median", "pixel_pool"):
            for metric in ("absrel", "rmse", "delta1"):
                names.append(f"{prefix}.standard.{reduction}.{metric}")
        names.append(f"{prefix}.standard.image_mean.near.absrel")
        names.append(f"{prefix}.standard.pixel_pool.near.absrel")
    return names


def _envelope(metadata, axes=DEPTH_AXES, version="3.0.0"):
    namespace = _EvalNamespace(
        producer="euler-eval",
        producer_version=version,
        modalities=("depth",),
        axes=axes,
        descriptions=DEPTH_DESCRIPTIONS,
    )
    return namespace.metric_set_envelope("depth", metadata=metadata)


def _gt_identity(variant):
    return hash_value(
        {
            "dataset": {"id": "cerema", "variant": variant, "scope": "depth", "split": None},
            "manifest": "fixture",
        }
    )


def depth_v2(
    *,
    items=DAY,
    offset=0.0,
    fid=21.5,
    gt_variant="cropped",
    gt_region=(270, 170, 734, 1730),
    domain_box=(270, 170, 734, 1730),
    spatial_method="none",
    resample=(1, 1),
    cap=38,
    clipped=0.0123,
    bench=(0.01, 32),
    alignment_mode="auto_affine",
    applied=False,
):
    values = _depth_values(items, offset)
    builder = ProtocolBuilder("depth.dense", ["euler.core", "euler.sky", "euler.depth"])
    builder.reference(
        "gt",
        identity=_gt_identity(gt_variant),
        modality="depth",
        dataset={"id": "cerema", "variant": gt_variant, "split": None, "scope": "depth"},
        manifest={"hash": hash_value([gt_variant]), "count": 1600},
        view={"source": SOURCE, "region": list(gt_region)},
    )
    builder.params(
        {
            "depth.alignment.mode": alignment_mode,
            "depth.alignment.applied": applied,
            "depth.alignment.fit": "scale_shift",
            "depth.alignment.fit_trim": 0.0,
            "depth.input_space": "metric",
            "depth.cap.max_m": cap,
            "depth.benchmark.range_m": list(bench),
            "depth.invalid.policy": "exclude_sentinel",
            "depth.conversion": "none",
            "sky.gt.masking": False,
            "sky.pred.masking": False,
            "sky.pred.fill": False,
            "spatial.method": spatial_method,
            "spatial.policy": "auto",
            "spatial.resize_target": "gt_grid",
            "spatial.mask_resample": "nearest_exact",
        }
    )
    builder.observe("domain.region", {"frame": SOURCE, "box": list(domain_box)})
    builder.observe("prediction.resample", list(resample))
    builder.observe("depth.cap.clipped", clipped)
    builder.population(
        items,
        unit="image",
        accounting={"reference": 1600, "prediction": len(items), "paired": len(items)},
    )
    builder.stratification("depth.gt_range", sqrt_stratification(*bench))
    builder.axis("space", "space").axis("category", "family").axis("reduction", "reduction")
    builder.axis("bin", "stratum", stratification="depth.gt_range")
    names = _depth_metric_names(values)
    builder.metrics_for(names, "depth.eval", DEPTH_AXES)
    builder.metric("count")
    for space, rows in values.items():
        builder.series(f"depth.eval.{space}.standard.count", [rows[i]["count"] for i in items])
        builder.series(
            f"depth.eval.{space}.standard.near.count", [rows[i]["near.count"] for i in items]
        )
    protocol = builder.build()
    return {
        "metricSet": _envelope(
            {
                "calibration_applied": applied,
                "canonical_space": "metric",
                "emitted_spaces": ["native", "metric"],
            }
        ),
        "protocol": protocol,
        "dataset_info": {"num_pairs": len(items), "gt_name": "GT", "pred_name": "fixture"},
        "meta": {"version": "3.0.0", "modality": "depth"},
        "depth": _depth_tree(values, fid),
        "per_file_metrics": {"files": _depth_per_file(values)},
    }


# --------------------------------------------------------------------------- legacy


def _boundaries(lo, hi):
    from euler_metric_naming.contract import sqrt_equal_width_edges

    e = sqrt_equal_width_edges(lo, hi)
    return {"range": [lo, hi], "near": [e[0], e[1]], "mid": [e[1], e[2]], "far": [e[2], e[3]]}


def depth_v1(
    *,
    items=DAY,
    offset=0.0,
    fid=21.5,
    dataset="cerema_cropped",
    split=None,
    calibration=None,
    resolution=(734, 1730),
    gt_dims=(734, 1730),
    pred_dims=(734, 1730),
    spatial="none",
    revision=3,
    bench=(0.01, 32.0),
    boundaries=True,
    sky_depth=38.0,
    hierarchy=None,
    per_file=True,
):
    values = _depth_values(items, offset)
    calibration = calibration or {"dataset": None, "scope": None, "split": None}
    parameters = {
        "reference.gt.modality": "depth",
        "reference.gt.dataset": dataset,
        "reference.gt.scope": "depth",
        "reference.gt.split": split,
        "reference.calibration.dataset": calibration["dataset"],
        "reference.calibration.scope": calibration["scope"],
        "reference.calibration.split": calibration["split"],
        "reference.segmentation.dataset": None,
        "reference.segmentation.scope": None,
        "reference.segmentation.split": None,
        "population.samples": len(items),
        "population.gt_sky_masking": False,
        "population.missing_segmentation": None,
        "population.pred_sky_masking": False,
        "population.evaluated_resolution": list(resolution),
        "population.benchmark_depth_range_m": list(bench),
        "method.metrics_revision": revision,
        "method.depth_alignment": "auto_affine",
        "method.sky_depth_m": sky_depth,
        "method.pred_sky_mask": False,
        "method.spatial_alignment": spatial,
    }
    metadata = {
        "input_space_detected": "metric",
        "metric_space_source": "prediction",
        "calibration_mode": "auto_affine",
        "calibration_applied": False,
        "emitted_spaces": ["native", "metric"],
        "canonical_space": "metric",
    }
    if boundaries:
        metadata["benchmark"] = {"depth_range": list(bench), "boundaries": _boundaries(*bench)}
    files = _depth_per_file(values, binned=False)
    if hierarchy:
        per_file_tree = {"children": {hierarchy: {"files": files}}}
    else:
        per_file_tree = {"files": files}
    payload = {
        "metricSet": _envelope(metadata, axes=LEGACY_DEPTH_AXES, version="2.47.0"),
        "protocol": {"schemaVersion": "1.3", "parameters": parameters},
        "dataset_info": {"num_pairs": len(items), "gt_name": "GT", "pred_name": "fixture"},
        "meta": {
            "version": "2.47.0",
            "modality": "depth",
            "gt": {"dimensions": {"height": gt_dims[0], "width": gt_dims[1]}},
            "pred": {"dimensions": {"height": pred_dims[0], "width": pred_dims[1]}},
            "spatial_alignment": {
                "mode": "auto",
                "method": spatial,
                "evaluated_dimensions": {"height": resolution[0], "width": resolution[1]},
            },
            "eval_params": {"sky_masking": False, "depth_alignment_mode": "auto_affine"},
        },
        "depth": _depth_tree(values, fid),
    }
    if per_file:
        payload["per_file_metrics"] = per_file_tree
    return payload


# ---------------------------------------------------------------------------- audio

AUDIO_TAG = "lab.audio@0.1"
AUDIO_AXES = {
    "reduction": AxisDeclaration(0, ("clip_mean", "word_pool"), True, "Reduction over utterances"),
    "band": AxisDeclaration(1, ("all", "low", "mid", "high"), True, "Input SNR band"),
}
AUDIO_ITEMS = [f"spk{s}~utt-{u}" for s in (1, 2) for u in range(3)]
BANDS = {0: "low", 1: "mid", 2: "high"}


def _d(**fields):
    return {**fields, "vocabulary": AUDIO_TAG}


def audio(*, items=AUDIO_ITEMS, sample_rate=16000, truncated=12.0, edges=(0, 10, 20, 40), denoise="none"):
    rows = {}
    for item in items:
        speaker, utt = item.split("~")
        j = (int(speaker[3:]) - 1) * 3 + int(utt.split("-")[1])
        rows[item] = {
            "sisdr": _round(10.0 + j),
            "wer": _round(0.10 + 0.02 * j),
            "words": 20 + 5 * j,
            "band": BANDS[int(utt.split("-")[1])],
        }
    builder = ProtocolBuilder("audio.separation", [])
    builder.reference(
        "gt",
        identity=hash_value({"dataset": {"id": "libri2mix", "variant": "clean", "split": "test"}}),
        modality="audio",
        dataset={"id": "libri2mix", "variant": "clean", "split": "test", "scope": "mix"},
        descriptor=_d(label="GT", role="reference"),
    )
    builder.param(
        "audio.sample_rate_hz", sample_rate,
        descriptor=_d(label="sample rate", role="definition", unit="Hz"),
    )
    builder.param("audio.mode", "separate", descriptor=_d(label="mode", role="definition"))
    builder.param(
        "audio.denoise", denoise,
        descriptor=_d(
            label="denoising",
            role="definition",
            appliesWhen={"param": "audio.mode", "op": "eq", "value": "enhance"},
        ),
    )
    builder.observe(
        "audio.truncated_s", truncated,
        descriptor=_d(label="truncated audio", role="domain", observe="count", unit="s"),
    )
    builder.population(items, unit="utterance")
    builder.stratification(
        "audio.snr_band",
        {
            "label": "input SNR",
            "variable": "input_snr_db",
            "unit": "dB",
            "edges": list(edges),
            "labels": ["low", "mid", "high"],
            "closed": "left",
            "lastClosed": True,
            "total": "all",
        },
    )
    builder.axis("reduction", "reduction").axis("band", "stratum", stratification="audio.snr_band")
    builder.reducer("clip_mean", _d(recompute="mean", over="item", label="clip mean"))
    builder.reducer("word_pool", _d(recompute="pool", over="observation", label="word pool"))
    builder.metric(
        "sisdr",
        _d(id="audio.sisdr", kind="score", rev=1, displayName="SI-SDR", direction="higher",
           bounds=[None, None], unit="dB", display={"scale": "absolute", "fullAt": 1, "unit": "dB"},
           perItem=True, pool={"form": "none"}, dependsOn=["*"]),
    )
    builder.metric(
        "wer",
        _d(id="audio.wer", kind="score", rev=1, displayName="WER", direction="lower",
           bounds=[0, None], unit=None, display={"scale": "relative"}, perItem=True,
           pool={"form": "mean", "weight": "words"}, dependsOn=["*"]),
    )
    builder.metric(
        "words",
        _d(id="audio.words", kind="statistic", rev=1, displayName="words", direction="none",
           bounds=[0, None], unit=None, display={"scale": "none"}, perItem=True,
           pool={"form": "none"}, dependsOn=["*"]),
    )
    builder.series("audio.eval.words", [rows[i]["words"] for i in items])
    builder.series(
        "audio.eval.low.words", [rows[i]["words"] if rows[i]["band"] == "low" else None for i in items]
    )
    protocol = builder.build()

    sisdr = {i: r["sisdr"] for i, r in rows.items()}
    wer = {i: r["wer"] for i, r in rows.items()}
    words = {i: r["words"] for i, r in rows.items()}
    low = [i for i, r in rows.items() if r["band"] == "low"]
    tree = {
        "eval": {
            "clip_mean": {
                "sisdr": _round(recompute("mean", sisdr)),
                "low": {"sisdr": _round(recompute("mean", sisdr, low))},
            },
            "word_pool": {
                "wer": _round(recompute("pool", wer, weights=words, form="mean")),
                "low": {"wer": _round(recompute("pool", wer, low, words, "mean"))},
            },
        }
    }
    children = {}
    for item in items:
        speaker, utt = item.split("~")
        r = rows[item]
        metrics = {"sisdr": r["sisdr"], "wer": r["wer"]}
        if r["band"] == "low":
            metrics["low"] = {"sisdr": r["sisdr"], "wer": r["wer"]}
        children.setdefault(speaker, {"files": []})["files"].append(
            {"id": utt, "metrics": {"audio": {"eval": metrics}}}
        )
    namespace = _EvalNamespace(
        producer="lab-audio-eval",
        producer_version="0.1.0",
        modalities=("audio",),
        axes=AUDIO_AXES,
        descriptions={
            "sisdr": MetricDescription(is_higher_better=True, unit="dB", display_name="SI-SDR"),
            "wer": MetricDescription(is_higher_better=False, min_value=0.0, display_name="WER"),
        },
    )
    return {
        "metricSet": namespace.metric_set_envelope("audio", metadata={}),
        "protocol": protocol,
        "dataset_info": {"num_pairs": len(items)},
        "meta": {"version": "0.1.0", "modality": "audio"},
        "audio": tree,
        "per_file_metrics": {"children": children},
    }



# ------------------------------------------------------------------------------ rgb

RGB_AXES = {
    "category": AxisDeclaration(
        0, ("image_quality", "depth_binned_photometric"), True, "Metric category"
    ),
    "dbin": AxisDeclaration(1, ("all", "near", "mid", "far"), True, "Fixed GT-depth bin"),
}
RGB_ITEMS = [f"frame-{i}" for i in range(6)]
RGB_BINS = ("all", "near", "mid", "far")


def rgb(*, items=RGB_ITEMS, offset=0, edges=(0, 1, 5, None), fid=12.0):
    """An RGB-like file: no reduction axis; implicit reducers; open-ended bins."""
    rows = {}
    for item in items:
        k = int(item.split("-")[1])
        rows[item] = {"psnr": _round(30.0 + k + offset)}
        for index, bin_name in enumerate(RGB_BINS):
            rows[item][bin_name] = _round(0.05 + 0.01 * k + 0.01 * offset + 0.1 * index)
    builder = ProtocolBuilder("rgb.restoration", ["euler.core", "euler.rgb"])
    builder.reference(
        "gt",
        identity=hash_value({"dataset": {"id": "cerema", "scope": "rgb", "split": None}}),
        modality="rgb",
        dataset={"id": "cerema", "variant": None, "split": None, "scope": "rgb"},
    )
    builder.params(
        {
            "rgb.fid.backend": "builtin",
            "rgb.depth_bins.policy": "v3",
            "spatial.method": "none",
            "spatial.uniform": True,
        }
    )
    builder.population(items, unit="image")
    builder.stratification(
        "rgb.depth_bins",
        {
            "label": "GT depth (fixed photometric bins)",
            "variable": "gt_depth_m",
            "unit": "m",
            "edges": list(edges),
            "labels": ["near", "mid", "far"],
            "closed": "right",
            "total": "all",
        },
    )
    builder.axis("category", "family").axis("dbin", "stratum", stratification="rgb.depth_bins")
    names = ["rgb.eval.image_quality.psnr", "rgb.eval.image_quality.fid"] + [
        f"rgb.eval.depth_binned_photometric.{b}.mae" for b in RGB_BINS
    ]
    builder.metrics_for(names, "rgb.eval", RGB_AXES)
    protocol = builder.build()
    psnr = {i: r["psnr"] for i, r in rows.items()}
    tree = {
        "eval": {
            "image_quality": {"psnr": _round(recompute("mean", psnr)), "fid": fid},
            "depth_binned_photometric": {
                b: {"mae": _round(recompute("mean", {i: r[b] for i, r in rows.items()}))}
                for b in RGB_BINS
            },
        }
    }
    files = [
        {
            "id": item,
            "metrics": {
                "rgb": {
                    "eval": {
                        "image_quality": {"psnr": rows[item]["psnr"]},
                        "depth_binned_photometric": {b: {"mae": rows[item][b]} for b in RGB_BINS},
                    }
                }
            },
        }
        for item in items
    ]
    namespace = _EvalNamespace(
        producer="euler-eval",
        producer_version="3.0.0",
        modalities=("rgb",),
        axes=RGB_AXES,
        descriptions={
            "psnr": MetricDescription(is_higher_better=True, unit="dB", display_name="PSNR"),
            "fid": MetricDescription(is_higher_better=False, display_name="FID"),
            "mae": MetricDescription(is_higher_better=False, display_name="MAE"),
        },
    )
    return {
        "metricSet": namespace.metric_set_envelope("rgb", metadata={"metric_sets": ["core"]}),
        "protocol": protocol,
        "dataset_info": {"num_pairs": len(items)},
        "meta": {"version": "3.0.0", "modality": "rgb"},
        "rgb": tree,
        "per_file_metrics": {"files": files},
    }


# ---------------------------------------------------------------------------- build


def build() -> dict:
    files = {}
    files["d1"] = depth_v2()
    files["d1b"] = depth_v2(clipped=0)
    files["d2"] = depth_v2(offset=0.01, fid=24.0, domain_box=(270, 170, 728, 1728),
                           spatial_method="vae_crop", clipped=0.011)
    files["d3"] = depth_v2(offset=0.005, fid=22.0, gt_variant="full_crop",
                           gt_region=(270, 170, 707, 1730), domain_box=(270, 170, 707, 1730))
    files["d4"] = depth_v2(items=NIGHT, fid=30.0)
    files["d5"] = depth_v2(items=DAY + NIGHT[:3], offset=0.03, fid=19.0, cap=120, clipped=0.002,
                           bench=(0.01, 100))
    files["d5b"] = depth_v2(items=DAY + NIGHT[:3], offset=0.03, fid=19.0, cap=120, clipped=0)
    files["d6"] = depth_v2(items=NIGHT, offset=0.02, fid=31.0, spatial_method="resize_prediction",
                           resample=(1.04, 1.02), alignment_mode="affine", applied=True)
    files["d7"] = depth_v2(items=NIGHT, offset=0.025, fid=29.0, alignment_mode="affine", applied=True)
    files["d8"] = depth_v2(items=DAY[:3])

    d9 = depth_v2()
    d9["protocol"]["metrics"]["absrel"]["rev"] = 2
    d9["protocol"]["metrics"]["absrel"]["description"] += " Revision 2: sentinel-free GT."
    files["d9"] = d9

    d10 = depth_v2()
    d10["protocol"]["parameters"]["depth.debug.tile_px"] = 512
    files["d10"] = d10

    for name, bits, mode in (("d10u", 16, "linear"), ("d10v", 8, "log")):
        payload = depth_v2()
        protocol = payload["protocol"]
        protocol["parameters"]["depth.quantization.bits"] = bits
        protocol["parameters"]["depth.quantization.mode"] = mode
        protocol["descriptors"]["depth.quantization.bits"] = {
            "label": "quantization bits", "role": "tuning", "vocabulary": "lab.depth@0.1"}
        protocol["descriptors"]["depth.quantization.mode"] = {
            "label": "quantization mode", "role": "tuning", "compare": "ignore",
            "vocabulary": "lab.depth@0.1"}
        files[name] = payload

    u1 = depth_v2()
    u1["protocol"]["schemaVersion"] = "3.0"
    files["u1"] = u1
    u2 = copy.deepcopy(u1)
    u2["protocol"]["parameters"]["depth.cap.max_m"] = 40
    files["u2"] = u2

    files["l1"] = depth_v1()
    files["l2"] = depth_v1(offset=0.01, fid=24.0, resolution=(728, 1728), pred_dims=(728, 1728),
                           spatial="vae_crop")
    files["l3"] = depth_v1(items=DAY[:3])
    files["l4"] = depth_v1(per_file=False)
    muses = dict(dataset="muses", split="test", resolution=(1080, 1920), gt_dims=(1080, 1920),
                 pred_dims=(1080, 1920), bench=(0.01, 80.0), boundaries=False, sky_depth=None,
                 calibration={"dataset": "muses", "scope": "calibration", "split": "test"},
                 hierarchy="night", items=[f"frame-{i}" for i in range(4)])
    files["m1"] = depth_v1(revision=1, **muses)
    files["m3"] = depth_v1(revision=3, offset=0.004, **muses)

    files["s1"] = audio()
    files["s2"] = audio(truncated=12.5)
    files["s3"] = audio(sample_rate=48000)
    files["s4"] = audio(items=["spk1~utt-0", "spk1~utt-1", "spk2~utt-0"])
    files["s5"] = audio(edges=(0, 5, 20, 40))
    files["s6"] = audio(denoise="spectral")

    files["r1"] = rgb()
    files["r2"] = rgb(items=RGB_ITEMS[:4], offset=1, fid=14.0)
    files["r3"] = rgb(edges=(0, 1, 10, None))
    files["r4"] = rgb(edges=(None, 1, 5, None))
    return files


def render(payload) -> str:
    return json.dumps(payload, indent=1, ensure_ascii=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail when files are out of date")
    args = parser.parse_args()
    files = build()
    stale = []
    for name, payload in files.items():
        path = EVALS / f"{name}.json"
        text = render(payload)
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != text:
                stale.append(path.name)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
    if args.check:
        extra = sorted(p.name for p in EVALS.glob("*.json") if p.stem not in files)
        if stale or extra:
            print("out of date:", ", ".join(stale + extra))
            return 1
        print("fixtures up to date")
    else:
        print(f"wrote {len(files)} files to {EVALS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
