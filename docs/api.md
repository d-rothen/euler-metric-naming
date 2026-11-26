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
    metrics=None,
    namespace_metrics=None,
    axis_roles=None,
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
| `metrics` | Training naming 2.0: metric definitions (contract §4.9 dicts, or `MetricDescription` objects) by base metric name, shared by every modality namespace. Not together with `descriptions`; `train` context only. |
| `namespace_metrics` | Training naming 2.0: definitions per namespace key (`"depth.train"`, `"sys.train"`); they win over `metrics` there. |
| `axis_roles` | Training naming 2.0: contract axis role per axis name (default `kind` → `kind`, `stage` → `stage`). |

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

### Training naming 2.0

With `metrics=` or `namespace_metrics=`, `training_naming_config()` writes a
training naming 2.0 declaration ([training.md](training.md)): every namespace
gets `axisRoles` and `metrics` (contract metric definitions), the
`metricDescriptions` derived from them for older consumers, and `sys.train`
the `euler.train` vocabulary terms (`lr`, `grad_norm`, and the GPU values
euler-train records). A run that declares 2.0 is held to it for every key it
logs: each must decompose under a declared namespace and have a definition.
The declaration is checked before it is returned; a definition that claims an
evaluation metric id raises `ValueError`.

```python
ns = MetricNamespace(
    producer="euler_train.weather_model",
    producer_version="0.2.0",
    modalities=("depth", "rgb"),
    stages=("prior", "final"),
    metrics={
        "total": MetricDescription(is_higher_better=False, display_name="Total loss"),
        "depth_mae": {"kind": "score", "direction": "lower", "unit": "m", "bounds": [0, None]},
    },
    namespace_metrics={"rgb.train": {"image_psnr": MetricDescription(is_higher_better=True, unit="dB")}},
)
euler_train.init(metric_naming=ns.training_naming_config())
```

`MetricDescription.to_definition()` and `MetricDescription.from_definition()`
convert between the two forms (`is_higher_better=None` becomes direction
`none`).

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

Modality names are looked up in the
[`euler-dataset-contract`](https://github.com/d-rothen/euler-dataset-contract)
registry (a dependency-free requirement of this package). Unknown names issue
a warning rather than failing construction, so custom modalities remain
possible.

## Capability report

`euler_metric_naming.capabilities.capability_report()` is registered in the
suite's `euler.capabilities` group, so `euler-suite report` and euler-view see
what an environment's installation can do (`metric_vocabulary_report` 1.0):

| Field | Holds |
|---|---|
| `contract` | `{"kind": "metric_vocabulary_report", "version": "1.0"}` |
| `package` | `{"name": "euler-metric-naming", "version"}` |
| `naming` | the name grammar `{scope}.{context}.{axes...}.{metric}`, contexts, reserved scopes |
| `protocol` | `schema_version` written (`2.0`) and `supported_majors` (`[1, 2]`) |
| `training_naming`, `validation_set` | the schema versions written |
| `extension_group` | `{"group": "euler.vocabularies", "api_version"}` |
| `vocabularies` | per registered vocabulary: `id`, `version`, `tag`, `context`, `title`, `terms` (counts of descriptors, metrics, reducers, stratificationSchemes) and `provider` (`{"kind": "builtin", "distribution", "version"}` or `{"kind": "entry_point", "name", "target", "distribution", "version"}`) |
| `schemas` | the shipped JSON Schemas: `name` (`vocabulary-1`, …), `file`, `id` |
| `extension_problems` | rejected installed vocabularies: `group`, `name`, `code` (`conflict`, `load_failed`, `invalid`, `api_version`, `shadows_builtin`), `message`, `distributions` |

## Comparison contract (`euler_metric_naming.contract`)

The evaluation protocol contract, schema 2, is defined normatively in
[contract.md](contract.md); terms are authored as described in
[vocabulary.md](vocabulary.md). Everything below needs nothing beyond the
dependency-free contract package.

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
| `reference(role, *, identity, modality=None, dataset=None, manifest=None, view=None, descriptor=None, identity_scheme="euler.reference/2")` | `references.<role>` (with `identityScheme`) and the `reference.<role>` descriptor |
| `population(ids, *, unit=None, accounting=None, items=True)` | `population` (NFC ids, `count`, `manifest`); `items.ids` |
| `stratification(id, mapping)` | a stratification (`sqrt_stratification(lo, hi)` builds euler-eval's benchmark bins) |
| `axis(name, role, *, stratification=None)` | the role of a metricSet axis |
| `describe(key, *, descriptor=None)` | a descriptor snapshot without a value, for keys a validation pass records (training.md §3.2) |
| `reducer(value, reducer=None)` / `metric(base, definition=None)` | a reducer / metric definition snapshot |
| `metrics_for(names, namespace, axes, *, missing="error")` | definitions of every base metric of `names`, and reducers of their reduction values |
| `series(name, values)` | a per-item series aligned with `items.ids` |
| `build()` | the validated `protocol` dict |

A metric definition that names an implicit `reducer` (for cells without a
reduction axis) brings that reducer along. Terms come from the loaded
vocabularies; an unknown key raises `KeyError`
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
| `relevant_keys(a, b, metric_name)` | §8.4 (informative): the `reference.<role>`, parameter and observation keys of either contract relevant to the cell, as the engine judges them; `b=None` judges `a` alone |
| `definition_key(contract)`, `definition_key_object(contract)` | §9.4 grouping key |
| `manifest_hash(ids)`, `hash_value(value)`, `canonicalize(value)` | §9 hashes and RFC 8785 text |
| `evaluate_condition(cond, parameters, observations)` | §7 (True / False / None) |
| `measure(kind, a, b)`, `phrase(...)`, `format_percent`, `format_sig3` | §3.3 |
| `load_vocabulary(id)`, `available_vocabularies()`, `builtin_vocabularies()`, `VocabularySet` | vocabulary lookup in the registry (shipped + `euler.vocabularies` entry points); unknown or rejected ids raise `UnknownVocabularyError` (a `KeyError`) with the reason |
| `vocabulary_registry()`, `reload_vocabularies()` | the registry (`.vocabularies` with `provider`, `.problems`) and a rebuild after installing a package in-process |
| `vocabulary_problems(document)` | every `vocabulary-1` problem of a document (dependency-free) |
| `vocabulary_documents(builtin_only=False)` | registered documents by id |
| `legacy_table()`, `normalize_legacy(...)` | the frozen v1 adapter |

### Training runs and validation sets

Specified in [training.md](training.md). Problems are `Problem(code, path,
message)`; codes and paths are shared with the TS artifact.

| Function | Purpose |
|---|---|
| `training_naming_problems(naming)` | problems of a training naming 2.0 declaration |
| `training_metric_problem(naming, key)` | the problem of one `train.jsonl` / `val.jsonl` key (`no_namespace`, `evaluation_name`, `no_decomposition`, `no_definition`), or `None` |
| `metric_value_problem(value)` | `not_a_number` unless a finite number |
| `training_metric_definition(naming, key)` | the key's definition in the run's declaration, or `None` |
| `training_contract(naming, namespace)` | a contract-shaped view (`metricSet`, `axes`, `metrics`) for definition readers |
| `is_training_naming_v2(naming)` | whether the declaration claims 2.x |
| `validation_set_problems(document)` | problems of a set document (the protocol block itself: `check_protocol`) |
| `validation_pass_problems(document, record)` | problems of a pass record against its set |
| `validation_pass_contract(document, record)` | the normalized contract of the evaluation a pass stands for |
| `load_validation_pass(document, record)` → `EvalFile` | the pass as a loaded results file, for `compare_files` and `recompute_cell` |
| `evaluation_metric_ids()` | ids of the evaluation vocabularies' metrics |

```python
from euler_metric_naming.contract import compare_files, load_eval_file, load_validation_pass

passed = load_validation_pass(set_document, pass_record)
evaluated = load_eval_file("pred/.eval/depth/eval.json")
compare_files(passed, evaluated, "depth.eval.metric.standard.image_mean.absrel").label
# "same protocol" | "compared on 64 shared files" | "not comparable · depth cap differs"
```

### TypeScript artifact

`euler_metric_naming/contract/ts/` ships the same contract for TS consumers
(euler-view): `euler-contract.ts` (logic and types, a port of the Python),
`euler-contract.data.ts` (generated by `scripts/generate_ts.py`: closed set,
vocabularies, legacy table, schemas), and `euler-contract-node.ts` (optional
`node:crypto` wiring). The API mirrors the Python one in camelCase:
`normalize`, `loadEvalFile`, `fileEvidence`, `compare(a, b, name, {itemsA,
itemsB, perItemA, perItemB})`, `compareFiles`, `relevantKeys(a, b | null,
name)`, `recompute`, `recomputeCell`,
`definitionKey`, `manifestHash`, `canonicalize`, `evaluateCondition`,
`measure`, `phrase`, and for training runs `trainingNamingProblems`,
`trainingMetricProblem`, `trainingMetricDefinition`, `trainingContract`,
`validationSetProblems`, `validationPassProblems`, `validationPassContract`,
`loadValidationPass`. The generated data also holds the JSON Schemas
`TRAINING_NAMING_SCHEMA`, `VALIDATION_SET_SCHEMA` and `VALIDATION_PASS_SCHEMA`.
Hash functions take an optional
`sha256Hex(bytes: Uint8Array) => string`; a pure-TS SHA-256 is the default.
Run its conformance suite with
`node --experimental-strip-types --test euler_metric_naming/contract/ts/conformance.test.ts`.
