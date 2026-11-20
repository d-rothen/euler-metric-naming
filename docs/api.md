# API guide

This guide covers the serialized contracts and the less common operations that
are intentionally kept out of the README's quick path.

## Naming model

A fully qualified metric name has this shape:

```text
{scope}.{context}.{axes...}.{base-metric}
```

The namespace is `{scope}.{context}`. `context` is currently `train` or `eval`.
Each declared axis consumes at most one segment, in ascending `position` order;
everything left becomes the base metric. A base metric can therefore contain
more than one segment, as in `sys.train.lr.geometry_encoder`.

`validate_metric_name()` enforces at least two dot-separated segments, a first
segment matching `[a-z0-9]+`, and later segments matching `[a-z0-9_]+`.

## `MetricNamespace`

```python
MetricNamespace(
    producer,
    producer_version,
    modalities,
    stages=None,
    context="train",
    descriptions=None,
    axes=None,
)
```

| Argument | Meaning |
|---|---|
| `producer` | Stable identifier for the package or component emitting the values. |
| `producer_version` | Version recorded beside the declaration for provenance. |
| `modalities` | Allowed metric scopes, such as `depth`, `rgb`, or `rays`. `sys` is reserved. |
| `stages` | Optional training stage values. Adds an optional `stage` axis after `kind`. |
| `context` | Namespace context: `train` (default) or `eval`. |
| `descriptions` | Base metric names mapped to `MetricDescription` objects. |
| `axes` | Custom axis declarations, normally used by evaluation producers. Custom axes and `stages` are mutually exclusive. |

The `producer`, `producer_version`, `modalities`, `stages`, `context`, and
`descriptions` properties expose copies or immutable views of the constructor
state. `axes(modality)` returns the declarations for that modality after first
checking that it belongs to the namespace.

### Training key builders

The training convenience methods all return strings:

| Method | Staged form | Stage-less form |
|---|---|---|
| `loss` | `ns.loss("depth", "prior", "mae")` | `ns.loss("depth", metric="total")` |
| `diag` | `ns.diag("depth", "final", "rmse")` | `ns.diag("depth", metric="rmse")` |
| `stat` | `ns.stat("depth", "data", "valid_fraction")` | `ns.stat("depth", metric="valid_fraction")` |
| `sys` | `ns.sys("lr", "encoder")` | `ns.sys("lr")` |

`loss`, `diag`, and `stat` are available only in `train` context and use the
required `kind` axis. They are unavailable when custom axes are supplied,
because the package cannot infer how a producer wants those axes populated.
`sys` uses the current context and has no declared axes.

### Training naming payload

`training_naming_config()` returns the JSON-ready object passed to
`euler_train.init(metric_naming=...)`:

```json
{
  "producer_key": "euler_train.weather_model",
  "producer_version": "0.1.0",
  "namespaces": {
    "depth.train": {
      "axes": {
        "kind": {
          "position": 0,
          "optional": false,
          "values": ["loss", "diag", "stat"],
          "description": "Metric kind"
        },
        "stage": {
          "position": 1,
          "optional": true,
          "values": ["prior", "final"],
          "description": "Pipeline refinement stage"
        }
      },
      "metricDescriptions": {
        "depth_mae": {
          "isHigherBetter": false,
          "unit": "meters"
        }
      }
    },
    "sys.train": {"axes": {}}
  }
}
```

Every training payload includes `sys.train`, allowing system metrics collected
by `euler-train` to resolve to a known namespace. Namespace entries own
independent dictionaries, so callers may safely specialize a returned payload
without changing another modality's entry.

Calling this method from an `eval` namespace raises `TypeError`.

### Evaluation metric-set envelope

Evaluation producers can declare their own axes:

```python
from euler_metric_naming import AxisDeclaration, MetricNamespace

ns = MetricNamespace(
    producer="euler_eval.depth",
    producer_version="2.0.0",
    modalities=("depth",),
    context="eval",
    axes={
        "space": AxisDeclaration(
            position=0,
            values=("native", "metric"),
            description="Depth space semantics",
        ),
        "category": AxisDeclaration(
            position=1,
            values=("standard", "geometric"),
            optional=True,
        ),
        "reduction": AxisDeclaration(
            position=2,
            values=("image_mean", "pixel_pool"),
            optional=True,
        ),
    },
)

envelope = ns.metric_set_envelope(
    "depth",
    source_kind="computed",
    metadata={"alignment": "scale"},
)
```

The envelope is ready for the `metricSet` field in `eval.json`:

```json
{
  "metricNamespace": "depth.eval",
  "producerKey": "euler_eval.depth",
  "producerVersion": "2.0.0",
  "sourceKind": "computed",
  "metadata": {"alignment": "scale"},
  "axes": {
    "space": {
      "position": 0,
      "optional": false,
      "values": ["native", "metric"],
      "description": "Depth space semantics"
    },
    "category": {
      "position": 1,
      "optional": true,
      "values": ["standard", "geometric"]
    },
    "reduction": {
      "position": 2,
      "optional": true,
      "values": ["image_mean", "pixel_pool"]
    }
  }
}
```

The provided metadata is copied before being placed in the envelope. For
backward compatibility, omitting custom axes uses the default `kind` and
optional `stage` declarations; evaluation producers should pass the axes their
metric tree actually uses.

## Axis declarations

`AxisDeclaration` is a frozen dataclass with four fields:

```python
AxisDeclaration(
    position=0,
    values=("loss", "diag", "stat"),
    optional=False,
    description="Metric kind",
)
```

`to_dict()` produces a JSON-ready dictionary and `from_dict()` restores an
instance. Custom axes on one `MetricNamespace` must have unique positions.

## Decompose and recompose

```python
from euler_metric_naming import decompose, recompose

axes = ns.axes("depth")
parsed = decompose(
    "depth.eval.metric.standard.image_mean.absrel",
    "depth.eval",
    axes,
)

parsed.namespace  # "depth.eval"
parsed.axes
# {"space": "metric", "category": "standard", "reduction": "image_mean"}
parsed.metric      # "absrel"
parsed.recompose() # "depth.eval.metric.standard.image_mean.absrel"

recompose(
    "depth.eval",
    axes,
    {"space": "metric", "category": "standard", "reduction": None},
    "absrel",
)
# "depth.eval.metric.standard.absrel"
```

`decompose()` sorts declarations by `position`, then consumes a segment only
when it is one of that axis's declared values. A required mismatch raises
`ValueError`; an optional mismatch records `None`. It never consumes an axis if
doing so would leave no segment for the base metric.

The `DecomposedMetric.recompose()` convenience method preserves this positional
axis order. The standalone `recompose()` additionally checks required axes and
rejects values not declared on an axis.

## Descriptions

`MetricDescription` attaches presentation semantics to a base metric. The same
description applies to every axis combination of that metric.

| Python field | Serialized key | Purpose |
|---|---|---|
| `is_higher_better` | `isHigherBetter` | Direction used for comparisons. |
| `min_value` | `min` | Expected lower bound. |
| `max_value` | `max` | Expected upper bound. |
| `scale` | `scale` | `linear`, `log`, `percentage`, or `binary`. |
| `unit` | `unit` | Display unit such as `meters` or `dB`. |
| `format_hint` | `formatHint` | Consumer-specific formatting hint. |
| `display_name` | `displayName` | Human-readable label. |
| `description` | `description` | Longer explanation. |

Fields left as `None` are omitted. `to_dict()` and `from_dict()` convert between
the dataclass and the camelCase wire representation.

## Matching helpers

All matching helpers preserve the input dictionary's iteration order.

- `compare_stages(metrics, namespace, modality, metric_name)` returns values for
  one base metric keyed by stage. A stage-less value is keyed by `None`.
- `filter_kind(metrics, namespace, modality, kind)` keeps metrics whose `kind`
  axis matches.
- `filter_glob(metrics, pattern)` uses Python `fnmatch` semantics. `*` can cross
  dots; include literal dots when the hierarchy's shape matters, for example
  `depth.train.*.prior.*`.

Names outside the requested namespace and names that cannot be decomposed under
its declarations are ignored by `compare_stages` and `filter_kind`.

## Modality validation

With no extras installed, modality validation is limited to the metric segment
rules. Installing the `contract` extra enables a soft lookup against
`euler-dataset-contract`:

```bash
pip install "euler-metric-naming[contract]"
```

Unknown registry entries issue a warning rather than failing construction, so
custom modalities remain possible and the core package stays dependency-free.

## Comparison contract (`euler_metric_naming.contract`)

The evaluation protocol contract, schema 2, is defined normatively in
[contract.md](contract.md); terms are authored as described in
[vocabulary.md](vocabulary.md). Everything below is dependency-free.

### Producers: `ProtocolBuilder`

```python
from euler_metric_naming.contract import ProtocolBuilder, hash_value, sqrt_stratification

builder = ProtocolBuilder(
    "depth.dense",                                   # protocol.task
    ["euler.core", "euler.sky", "euler.depth"],      # vocabularies to take terms from
)
builder.reference(
    "gt",
    identity=hash_value({"dataset": dataset, "manifest": manifest, "head": head}),
    modality="depth",
    dataset={"id": "cerema", "variant": "cropped", "split": None, "scope": "depth"},
    view={"source": "cerema/rgb_left@1024x1920", "region": [270, 170, 734, 1730]},
)
builder.params({"depth.cap.max_m": 38, "depth.alignment.applied": False})
builder.observe("domain.region", {"frame": "cerema/rgb_left@1024x1920", "box": [270, 170, 734, 1730]})
builder.population(item_ids, unit="image")           # count + manifest (+ items.ids)
builder.stratification("depth.gt_range", sqrt_stratification(0.01, 32))
builder.axis("space", "space").axis("category", "family").axis("reduction", "reduction")
builder.axis("bin", "stratum", stratification="depth.gt_range")
builder.metrics_for(metric_names, "depth.eval", axes)  # definitions + reducers of every base
builder.series("depth.eval.metric.standard.count", counts)
protocol = builder.build()                           # JSON-ready; raises ProtocolValidationError
```

| Method | Records |
|---|---|
| `param(key, value, *, descriptor=None)` | a parameter (scalar or ≤ 64 scalars) and its descriptor snapshot |
| `params(mapping)` | several vocabulary parameters |
| `observe(key, value, *, descriptor=None)` | an observation |
| `reference(role, *, identity, modality=None, dataset=None, manifest=None, view=None, descriptor=None)` | `references.<role>` and the `reference.<role>` descriptor |
| `population(ids, *, unit=None, accounting=None, items=True)` | `population` (NFC ids, `count`, `manifest`); `items.ids` |
| `stratification(id, mapping)` | a stratification (`sqrt_stratification(lo, hi)` builds euler-eval's benchmark bins) |
| `axis(name, role, *, stratification=None)` | the role of a metricSet axis |
| `reducer(value, reducer=None)` / `metric(base, definition=None)` | a reducer / metric definition snapshot |
| `metrics_for(names, namespace, axes, *, missing="error")` | definitions of every base metric of `names`, and reducers of their reduction values |
| `series(name, values)` | a per-item series aligned with `items.ids` |
| `build()` | the validated `protocol` dict |

Terms come from the loaded vocabularies; an unknown key raises `KeyError`
unless its descriptor or definition is passed explicitly. `check_protocol(p)`
/ `protocol_problems(p)` validate any block structurally (closed sets, v1
parameter limits, shapes).

### Consumers: normalize, compare, recompute

```python
from euler_metric_naming.contract import compare_files, load_eval_file

a = load_eval_file("model_a/.eval/depth/eval.json")   # path, JSON text or dict; v1 or v2
b = load_eval_file("model_b/.eval/depth/eval.json")
verdict = compare_files(a, b, "depth.eval.metric.standard.image_mean.absrel")
verdict.tier     # "same" | "paired" | "comparable" | "incomparable"
verdict.label    # "comparable · domain differs by 0.9 %"
verdict.shared   # N shared files when the population rule compared id sets
verdict.to_dict()
```

| Function | Purpose |
|---|---|
| `normalize(protocol, metric_set=None, meta=None)` | any schema version → contract (v1 through the legacy adapter) |
| `load_eval_file(source)` → `EvalFile` | namespace, axes, contract, flattened `metrics`, `per_file` (item id → name → value), `item_ids`, `series`, `per_item_names` |
| `file_evidence(eval_file)` | `(items, per_item_names)`: NFC item set (None when unknown or not matching the manifest) |
| `compare(a, b, metric_name, items_a=None, items_b=None, per_item_a=None, per_item_b=None)` → `Verdict` | the §8 engine on two contracts |
| `compare_files(file_a, file_b, metric_name)` | `compare` with evidence from the files |
| `recompute(op, values, items=None, weights=None, form=None)` | mean / median / pool of per-item values |
| `recompute_cell(contract, eval_file, metric_name, items=None)` | an aggregate recomputed on (shared) items |
| `per_item_name(contract, metric_name)` | the per-item name of an aggregate |
| `definition_key(contract)`, `definition_key_object(contract)` | §9.4 grouping key |
| `manifest_hash(ids)`, `hash_value(value)`, `canonicalize(value)` | §9 hashes and RFC 8785 text |
| `evaluate_condition(cond, parameters, observations)` | §7 (True / False / None) |
| `measure(kind, a, b)`, `phrase(...)`, `format_percent`, `format_sig3` | §3.3 |
| `load_vocabulary(id)`, `available_vocabularies()`, `VocabularySet` | vocabulary lookup |
| `legacy_table()`, `normalize_legacy(...)` | the frozen v1 adapter |

### TypeScript artifact

`euler_metric_naming/contract/ts/` ships the same contract for TS consumers
(euler-view): `euler-contract.ts` (logic and types, a port of the Python),
`euler-contract.data.ts` (generated by `scripts/generate_ts.py`: closed set,
vocabularies, legacy table, schemas), and `euler-contract-node.ts` (optional
`node:crypto` wiring). The API mirrors the Python one in camelCase:
`normalize`, `loadEvalFile`, `fileEvidence`, `compare(a, b, name, {itemsA,
itemsB, perItemA, perItemB})`, `compareFiles`, `recompute`, `recomputeCell`,
`definitionKey`, `manifestHash`, `canonicalize`, `evaluateCondition`,
`measure`, `phrase`. Hash functions take an optional
`sha256Hex(bytes: Uint8Array) => string`; a pure-TS SHA-256 is the default.
Run its conformance suite with
`node --experimental-strip-types --test euler_metric_naming/contract/ts/conformance.test.ts`.
