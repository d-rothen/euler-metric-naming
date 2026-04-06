# euler-metric-naming

Structured metric naming for the Euler ML ecosystem. Provides a single source of truth for metric name construction, validation, decomposition, and envelope generation across `euler-train`, `euler-eval`, and `euler-view`.

## Installation

```
pip install euler-metric-naming
```

Zero hard dependencies. Optional soft dependency on `euler-dataset-contract` for modality validation.

## Naming convention

All metric names use dot-separated segments: `{scope}.{context}.{axes...}.{metric}`.

```
depth.train.loss.prior.log_radius
├─────────┘ ├──┘ ├───┘ └─ base metric
│           │    └──────── stage axis (optional)
│           └───────────── kind axis
└───────────────────────── namespace (scope.context)
```

- **Scope** is a modality (`depth`, `rgb`, `rays`) or a reserved prefix (`sys`).
- **Context** is `train` or `eval`.
- **Axes** are declared per-namespace (e.g. `kind`, `stage`) and decomposed structurally, never parsed from the string.

Segment charset: `[a-z0-9_]` (first segment: `[a-z0-9]` only).

## Quick start

### Define a namespace

```python
from euler_metric_naming import MetricNamespace, MetricDescription

ns = MetricNamespace(
    producer="euler_train.weather_metric",
    producer_version="0.1.0",
    modalities=("depth", "rgb", "rays"),
    stages=("prior", "pred", "structure", "consolidation", "final"),
    descriptions={
        "log_radius": MetricDescription(is_higher_better=False, scale="log"),
        "depth_mae": MetricDescription(is_higher_better=False, unit="meters"),
    },
)
```

### Build metric keys

```python
# Losses — with pipeline stage
ns.loss("depth", "prior", "log_radius")   # "depth.train.loss.prior.log_radius"
ns.loss("depth", "final", "depth")         # "depth.train.loss.final.depth"

# Losses — without stage
ns.loss("depth", metric="total")           # "depth.train.loss.total"

# Diagnostics and stats
ns.diag("depth", "final", "depth_mae")    # "depth.train.diag.final.depth_mae"
ns.stat("depth", "structure", "confidence") # "depth.train.stat.structure.confidence"

# System metrics (learning rate, GPU stats, etc.)
ns.sys("lr")                               # "sys.train.lr"
ns.sys("lr", "geometry_encoder")           # "sys.train.lr.geometry_encoder"

# Other modalities
ns.loss("rgb", "weather", "dehaze")        # "rgb.train.loss.weather.dehaze"
ns.loss("rays", "ray", "l1")              # "rays.train.loss.ray.l1"
```

Invalid calls raise `ValueError`:

```python
ns.loss("depth", "nonexistent_stage", "x")  # ValueError: unknown stage
ns.loss("video", "prior", "x")              # ValueError: unknown modality
```

### Models without pipeline stages

```python
ns = MetricNamespace(
    producer="euler_train.simple_baseline",
    producer_version="0.1.0",
    modalities=("depth",),
)

ns.loss("depth", metric="mae")    # "depth.train.loss.mae"
ns.diag("depth", metric="rmse")   # "depth.train.diag.rmse"

ns.loss("depth", "prior", "mae")  # ValueError: stages not declared
```

## Usage with euler-train

`euler-train` accepts a `metric_naming` parameter that tells `euler-view` how to decompose, group, and filter the run's metrics. Generate it with `training_naming_config()` and pass it at run init:

```python
import euler_train
from euler_metric_naming import MetricNamespace

ns = MetricNamespace(
    producer="euler_train.weather_metric",
    producer_version="0.1.0",
    modalities=("depth", "rgb"),
    stages=("prior", "final"),
)

run = euler_train.init(
    config=config,
    metric_naming=ns.training_naming_config(),
    stream=stream,
)
```

Then use the namespace to build metric keys passed to `run.log()`:

```python
losses = {
    ns.loss("depth", "prior", "log_radius"): prior_loss.item(),
    ns.loss("depth", "final", "log_radius"): final_loss.item(),
    ns.loss("depth", metric="total"): total_loss.item(),
    ns.sys("lr"): scheduler.get_last_lr()[0],
}
run.log(losses, step=step, epoch=epoch)
```

The `.log()` API is unchanged -- it still accepts a flat `dict[str, float]`. The keys just follow the naming convention now. When `metric_naming` is present, euler-train also namespaces its auto-collected GPU stats under `sys.train.*` (e.g. `sys.train.gpu_util_pct`).

### What happens in euler-view

The `metric_naming` payload is stored in `meta.json` and included in the stream init event. euler-view extracts it into `model_runs.metric_naming` and uses it to:

1. Resolve `metric_namespace` for each metric row (e.g. `depth.train`, `sys.train`).
2. Decompose metric names into axis values for grouping/filtering in the UI.
3. Read metric descriptions for display formatting (scale, units, direction).

Runs without `metric_naming` continue to work as before -- metrics are stored with `metric_namespace = NULL` and displayed as opaque flat names.

### `training_naming_config()` output

```python
ns.training_naming_config()
# {
#     "producer_key": "euler_train.weather_metric",
#     "producer_version": "0.1.0",
#     "namespaces": {
#         "depth.train": {
#             "axes": {
#                 "kind": {"position": 0, "optional": false, "values": ["loss", "diag", "stat"]},
#                 "stage": {"position": 1, "optional": true, "values": ["prior", "final"]}
#             },
#             "metricDescriptions": {
#                 "log_radius": {"isHigherBetter": false, "scale": "log"},
#                 "depth_mae": {"isHigherBetter": false, "unit": "meters"}
#             }
#         },
#         "rgb.train": { ... },
#         "sys.train": {"axes": {}}
#     }
# }
```

`sys.train` is always included for training namespaces so that system metrics (`sys.train.lr`, `sys.train.gpu_util_pct`, etc.) are recognized by euler-view.

## Decomposition

Given a metric name and its namespace's axis declarations, `decompose` extracts axis values and the base metric:

```python
from euler_metric_naming import decompose

axes = ns.axes("depth")
result = decompose("depth.train.loss.prior.log_radius", "depth.train", axes)

result.namespace  # "depth.train"
result.axes       # {"kind": "loss", "stage": "prior"}
result.metric     # "log_radius"
```

Optional axes that don't match are set to `None`:

```python
result = decompose("depth.train.loss.total", "depth.train", axes)
result.axes  # {"kind": "loss", "stage": None}
result.metric  # "total"
```

The algorithm guarantees that at least one segment is always left for the base metric name. If consuming an optional axis value would leave zero remaining segments, the axis is skipped.

## Comparison and filtering

```python
from euler_metric_naming import compare_stages, filter_kind, filter_glob

metrics = {
    "depth.train.diag.prior.depth_mae": 0.42,
    "depth.train.diag.final.depth_mae": 0.28,
    "depth.train.loss.prior.log_radius": 0.15,
    "depth.train.loss.total": 1.23,
    "sys.train.lr": 3e-5,
}

# Compare one metric across pipeline stages
compare_stages(metrics, ns, "depth", "depth_mae")
# {"prior": 0.42, "final": 0.28}

# All losses for a modality
filter_kind(metrics, ns, "depth", "loss")
# {"depth.train.loss.prior.log_radius": 0.15, "depth.train.loss.total": 1.23}

# Glob matching
filter_glob(metrics, "depth.train.*.prior.*")
# {"depth.train.diag.prior.depth_mae": 0.42, "depth.train.loss.prior.log_radius": 0.15}
```

## Metric descriptions

Attach display metadata to base metric names (after stripping namespace and axes). The same description applies to all axis combinations of that metric.

```python
from euler_metric_naming import MetricDescription

descriptions = {
    "log_radius": MetricDescription(
        is_higher_better=False,
        scale="log",
        display_name="Log-Radius L1",
    ),
    "depth_mae": MetricDescription(
        is_higher_better=False,
        scale="linear",
        unit="meters",
    ),
    "confidence": MetricDescription(
        is_higher_better=True,
        min_value=0.0,
        max_value=1.0,
    ),
}

ns = MetricNamespace(
    producer="euler_train.my_model",
    producer_version="0.1.0",
    modalities=("depth",),
    descriptions=descriptions,
)
```

Descriptions are serialized to camelCase JSON in the `metricDescriptions` field of the naming config and metric set envelopes.

## Evaluation envelopes

For `euler-eval` producers, generate the `metricSet` envelope included in `eval.json`:

```python
eval_ns = MetricNamespace(
    producer="euler_eval.depth",
    producer_version="2.0.0",
    modalities=("depth",),
    context="eval",
    descriptions={
        "absrel": MetricDescription(is_higher_better=False, scale="linear"),
    },
)

eval_ns.metric_set_envelope("depth")
# {
#     "metricNamespace": "depth.eval",
#     "producerKey": "euler_eval.depth",
#     "producerVersion": "2.0.0",
#     "sourceKind": "computed",
#     "metadata": {},
#     "axes": { ... },
#     "metricDescriptions": { ... }
# }
```

## Package structure

```
euler_metric_naming/
    __init__.py        # Public API exports
    namespace.py       # MetricNamespace class
    axes.py            # AxisDeclaration, decompose(), recompose()
    descriptions.py    # MetricDescription dataclass
    matching.py        # compare_stages(), filter_kind(), filter_glob()
    _compat.py         # euler-dataset-contract soft import
```
