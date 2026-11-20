# Training runs: naming schema 2.0, validation sets and validation passes

This document is the normative definition of how a training run names and
describes its metrics so that consumers can read them with the same generic
code they use for evaluations. The key words MUST, MUST NOT, SHOULD and MAY are
used as in RFC 2119. The cross-repository plan that rolls it out is
[training-alignment-plan.md](training-alignment-plan.md).

It builds on the evaluation-protocol contract, schema 2
([contract.md](contract.md)) and changes nothing in it: a validation set is an
ordinary evaluation in contract terms, and every verdict about it comes from
the unchanged algorithm of contract section 8.

The JSON Schemas are
[`training-naming-2.schema.json`](../euler_metric_naming/contract/schema/training-naming-2.schema.json),
[`validation-set-1.schema.json`](../euler_metric_naming/contract/schema/validation-set-1.schema.json) and
[`validation-pass-1.schema.json`](../euler_metric_naming/contract/schema/validation-pass-1.schema.json)
(draft 2020-12). Where this document and a schema disagree, this document
wins and the schema is a bug.

## 1. Two kinds of training metrics

A training run writes two kinds of numbers, and this document treats them
differently on purpose.

1. **Telemetry**: losses, training-batch diagnostics, statistics of the model,
   learning rates, GPU use. Nothing outside the run computes them. They are
   named `<scope>.train.<axes…>.<base>` (section 2), carry metric definitions
   (direction, bounds, unit, display scale), and are never judged against an
   evaluation. They live in `train.jsonl` and `val.jsonl` as before.
2. **Validation scores**: a quantity an evaluator also computes, such as
   AbsRel, δ < 1.25 or PSNR on a fixed held-out set. A run records them as a
   **validation set** (section 3): the evaluation of the live model on a fixed
   population, declared once with a contract-v2 `protocol` and written with
   the evaluator's own metric names (`depth.eval.metric.standard.image_mean.absrel`),
   plus one **validation pass** record per time the run evaluated it. They
   never go into `val.jsonl`.

A quantity that resembles an evaluation metric but is not computed the way
the evaluator computes it (a different unit, a different mask, a training
batch instead of a fixed set) is telemetry with an honest definition, not a
validation score.

## 2. Training naming schema 2.0

### 2.1 Where it lives

The declaration is the `metric_naming` object of a run's `meta.json` (written
by euler-train from `MetricNamespace.training_naming_config()`). Schema 2.0
adds `schema_version` and, per namespace, `axisRoles` and `metrics`:

```json
{
  "schema_version": "2.0",
  "producer_key": "euler_train.dehaze_baseline",
  "producer_version": "0.2.0",
  "namespaces": {
    "depth.train": {
      "axes": {
        "kind":  { "position": 0, "values": ["loss", "diag", "stat"], "optional": false },
        "stage": { "position": 1, "values": ["prior", "final"], "optional": true }
      },
      "axisRoles": { "kind": { "role": "kind" }, "stage": { "role": "stage" } },
      "metrics": {
        "total":     { "kind": "score", "direction": "lower", "displayName": "Total loss" },
        "depth_mae": { "kind": "score", "direction": "lower", "unit": "m", "bounds": [0, null],
                       "displayName": "Depth MAE (training batch)" }
      },
      "metricDescriptions": {
        "total":     { "isHigherBetter": false, "displayName": "Total loss" },
        "depth_mae": { "isHigherBetter": false, "unit": "m", "min": 0, "displayName": "Depth MAE (training batch)" }
      }
    },
    "sys.train": {
      "axes": {},
      "axisRoles": {},
      "metrics": {
        "lr": { "kind": "score", "direction": "none", "bounds": [0, null], "displayName": "Learning rate",
                "id": "train.lr", "vocabulary": "euler.train@1.0" }
      }
    }
  }
}
```

A declaration without `schema_version`, or with a `1.x` version, is a
schema-1 declaration: consumers read it as before and none of the rules below
apply. A declaration whose `schema_version` has a major other than 1 or 2 is
reported (`schema_version`) and read as schema 1.

### 2.2 Namespaces

- Every key of `namespaces` MUST have exactly two segments `<scope>.train`,
  where `<scope>` matches `[a-z0-9]+` (a modality such as `depth`, `rgb`,
  `segmentation`, or `sys` for system values). The context is always `train`:
  evaluation names belong to validation sets (section 3).
- Every namespace MUST have `axes` (axis declarations exactly as in schema 1:
  `position`, `values`, `optional`, `description`) and `metrics`.
- `axisRoles` MAY give each axis a role from the closed set of contract
  section 3.9. An axis without a role is `family`. Training axes are normally
  `kind` (role `kind`) and `stage` (role `stage`), both roles that only change
  display. A role name outside the closed set is reported (`axis_role`); an
  `axisRoles` entry for an undeclared axis is reported too.

### 2.3 Metric definitions

`metrics` maps a base metric name (the name after the namespace and all
consumed axis segments, euler-metric-naming `decompose`) to a metric definition
with the fields of contract section 4.9: `kind`, `displayName`, `description`,
`direction`, `bounds`, `unit`, `display`, and optionally `id`, `rev` and
`vocabulary`. The pooling, reducer and reference fields (`perItem`, `pool`,
`reducer`, `requires`, `dependsOn`) have no meaning for telemetry and SHOULD be
left out.

- A definition MUST pass the contract's metric-definition rules
  (`definition_invalid` otherwise).
- A definition MUST NOT carry the `id` of a metric of an evaluation vocabulary
  (`euler.core`, `euler.sky`, `euler.depth`, `euler.rgb`, `euler.rays`,
  `euler.points3d`; code `evaluation_id`). A telemetry value never claims to be
  the quantity an evaluator computes; if it is, it belongs in a validation set.
- Telemetry vocabulary terms (`euler.train`, section 2.6) MAY be snapshotted
  with their `vocabulary` tag, like evaluation terms.
- `kind: statistic` and `kind: observation` hide a value from metric lists,
  exactly as for evaluations.

Definitions travel with the run: a consumer judges a run's names from the
snapshot in its `meta.json`, never from a registry of its own. Its own registry
is a fallback for schema-1 runs only.

### 2.4 Legacy descriptions

So that consumers that predate schema 2.0 keep showing directions and units, a
producer SHOULD also write `metricDescriptions`, derived from the definitions:
`isHigherBetter` from `direction` (`higher` → true, `lower` → false, `none`
→ absent), `min`/`max` from `bounds`, `unit`, `displayName`, `description`.
Consumers that understand schema 2.0 MUST read `metrics` and ignore
`metricDescriptions`.

### 2.5 What a run declaring 2.0 promises

Every key a run writes to `train.jsonl` or `val.jsonl`, other than the
reserved record fields `step`, `epoch`, `wall_time` and `elapsed_sec`, MUST:

1. start with a declared namespace followed by `.` (else `no_namespace`; a key
   whose second segment is `eval` is reported as `evaluation_name` instead,
   because evaluation names belong to validation sets);
2. decompose under that namespace's axes (else `no_decomposition`);
3. have a definition for its base name in that namespace's `metrics` (else
   `no_definition`);
4. have a value that is a finite JSON number; booleans are not numbers (else
   `not_a_number`).

There is no mixing: a run that declares 2.0 is held to it for every key. Runs
that do not declare 2.0 are unaffected (the clean split: old runs stay as they
are).

Producers check these rules and report what fails (section 5); they do not
drop values. Consumers MUST NOT reject a run for them: a key that fails is
shown as "not declared" and its direction is unknown.

### 2.6 The `euler.train` vocabulary

[`euler.train`](../euler_metric_naming/contract/vocabularies/euler.train.json)
holds telemetry terms that many producers share, starting with the system
values euler-train records itself (`gpu_util_pct`, `gpu_mem_util_pct`,
`gpu_mem_used_gb`, `gpu_mem_total_gb`) and the common optimizer values (`lr`,
`grad_norm`). `training_naming_config()` adds all of them to `sys.train`. Their
ids start with `train.`, never with an evaluation area.

### 2.7 Problem codes of a declaration

| Code | Meaning |
|---|---|
| `not_object` | The declaration, a namespace entry or `axisRoles` is not an object |
| `schema_version` | `schema_version` is not a `MAJOR.MINOR` string with major 2 |
| `namespaces` | `namespaces` is missing, not an object, or empty |
| `namespace_key` | A namespace key is not `<scope>.train` |
| `axes` | `axes` is missing or not valid axis declarations (positions unique integers, values non-empty lists of `[a-z0-9_]+`) |
| `axis_role` | An `axisRoles` entry names an undeclared axis, a role outside the closed set, or a second `space` / `reduction` axis |
| `metrics` | `metrics` is missing or not an object |
| `definition_invalid` | A definition breaks the contract's metric-definition rules |
| `evaluation_id` | A definition carries the `id` of an evaluation vocabulary metric |

Each problem has a `code`, a `path` (dotted, from the declaration root, with
namespace keys and base names as single segments, e.g.
`namespaces.depth.train.metrics.depth_mae`) and a free-text `message`.
Conformance fixtures compare codes and paths, not messages.

## 3. Validation sets

### 3.1 Files

A validation set is identified by a **key** matching `[a-z0-9][a-z0-9_-]{0,63}`
(for example `cerema` or `muses_fog`). Inside the run directory it has two
files:

```text
<run>/validation/<key>/set.json       # the set document, written once
<run>/validation/<key>/passes.jsonl   # one pass record per line, appended
```

and a pointer in `meta.json`:

```json
"validation_sets": {
  "cerema": {
    "schema_version": "1.0",
    "set": "validation/cerema/set.json",
    "passes": "validation/cerema/passes.jsonl",
    "metric_namespace": "depth.eval",
    "label": "CEREMA fog, 100 frames"
  }
}
```

The pointer is small on purpose: euler-train rewrites `meta.json` on every
logged step, so the documents themselves never live in it.

Validation-set names MUST NOT appear in `train.jsonl` or `val.jsonl`. Several
sets usually share their names (each evaluates `depth.eval.metric.standard.image_mean.absrel`),
so writing them into one series would mix sets.

### 3.2 The set document

```json
{
  "kind": "euler.validation_set",
  "schemaVersion": "1.0",
  "key": "cerema",
  "label": "CEREMA fog, 100 frames",
  "metricSet": { "metricNamespace": "depth.eval", "axes": { "…": "…" }, "…": "…" },
  "protocol": { "schemaVersion": "2.0", "task": "depth.dense", "…": "…" }
}
```

- `kind` MUST be `euler.validation_set`; `schemaVersion` MUST have major 1;
  `key` MUST be the set's key.
- `metricSet` is a metricSet envelope exactly as in an `eval.json`
  (`metricNamespace`, `axes`, and the informative `producerKey`,
  `producerVersion`, `sourceKind`, `metadata`, `metricDescriptions`). A
  producer SHOULD use the namespace and axes its evaluator writes for the same
  evaluation, so that a validation pass and an evaluation name the same cell
  with the same string and compare without any mapping.
- `protocol` is a contract-v2 `protocol` block (contract section 4). It holds
  everything that is the same for every pass: references, configured
  parameters, stratifications, axis roles, reducers, metric definitions, and
  the population. Its `items.series` MUST be empty (per-item series belong to
  passes). It MUST pass the contract's protocol validation; the validators of
  this document check only what is specific to validation sets and leave the
  protocol block to `check_protocol` (Python) or `PROTOCOL_SCHEMA` (TS).
- `protocol.descriptors` MUST describe every key a pass records in its own
  `parameters` and `observations` (section 3.3), including keys the set
  document does not give a value for. This is how a resolved choice (the
  spatial method the first sample needed, whether an alignment ran) can be
  recorded per pass and still be described by the vocabulary snapshot.
- `protocol.population` SHOULD list the items the set evaluates
  (`ProtocolBuilder.population(ids)`, which also writes `items.ids`), so a
  pass that evaluates exactly these items does not repeat them.
- `protocol.metrics` MUST define every base metric a pass writes.

A set document is immutable for the life of a run. Declaring the same key
again with a different document is an error: the curve of one set must not
mix two protocols.

### 3.3 The pass record

One JSON object per line of `passes.jsonl`:

```json
{
  "step": 7554, "epoch": 5, "wall_time": 1794651078.2,
  "metrics": {
    "depth.eval.metric.standard.image_mean.absrel": 0.4307,
    "depth.eval.metric.standard.pixel_pool.absrel": 0.4400
  },
  "parameters": { "spatial.method": "none", "spatial.uniform": true, "depth.alignment.applied": false },
  "observations": { "domain.region": { "frame": "sha256:…", "box": [0, 0, 384, 768] } },
  "population": { "ids": ["scene-1~f0", "scene-1~f1"] },
  "per_item": { "depth.eval.metric.standard.absrel": [0.41, 0.45] },
  "series": { "depth.eval.metric.standard.count": [181234, 176002] }
}
```

| Field | Required | Meaning |
|---|---|---|
| `step`, `epoch` | `step` yes | The training step (integer ≥ 0) and epoch (integer) the model had |
| `wall_time` | no | Unix time of the record |
| `metrics` | yes | Aggregate values by full metric name; each name MUST decompose under the set's `metricSet` axes and its base MUST have a definition in the set's `protocol.metrics` |
| `parameters` | no | Parameters resolved during this pass; they replace the set's value of the same key |
| `observations` | no | Observations of this pass (contract section 4.5); they replace the set's |
| `population` | no | `{"ids": [...], "accounting"?: {...}}` when this pass evaluated other items than the set lists; `ids` are `euler.item_id/1` ids in NFC, unique |
| `per_item` | no | Per-item score values by per-item name (contract 4.11), each a list aligned with the pass's ids (`null` for an item without a value) |
| `series` | no | Per-item statistic series (pooling weights) by name, aligned the same way |

The pass's ids are `population.ids` when given, else the set's
`protocol.items.ids`. A pass that writes `per_item` or `series` without any ids
is reported (`length`).

### 3.4 The evaluation a pass stands for

A consumer turns a set document `S` and a pass record `P` into a results file
evaluation and judges it with contract section 8, unchanged:

1. `metricSet` is `S.metricSet`.
2. The protocol starts as a copy of `S.protocol`. Then
   - `parameters` becomes `S.protocol.parameters` updated with `P.parameters`;
   - `observations` becomes `S.protocol.observations` updated with `P.observations`;
   - when `P.population` has `ids`: `population` keeps `unit` and `idScheme`
     of `S.protocol.population`, takes `count` = the number of distinct ids
     and `manifest` = the population manifest of the ids (contract 9.3), unless
     `P.population` records its own `count` / `manifest` (which then stand and
     are checked like any recorded manifest), and `accounting` from
     `P.population` when given; `items.ids` becomes `P.population.ids`;
   - `items.series` becomes `P.series` (or empty).
3. The result is normalized (contract section 8, "normalized contract").
4. The aggregate metrics are `P.metrics` (finite numbers only).
5. The per-item values of item `ids[i]` are `P.per_item[name][i]` for every
   per-item name (finite numbers only; a list whose length differs from the
   ids is not used at all).
6. The evaluated item set is the pass's ids (or unknown when there are none),
   and the per-item names are the names of `P.per_item` and `P.series`.

Python: `load_validation_pass(S, P)` returns an `EvalFile` for
`compare_files` and `recompute_cell`; `validation_pass_contract(S, P)` returns
the normalized contract. TS: `loadValidationPass` and
`validationPassContract`. Two passes, a pass and an evaluation, or two
evaluations are then judged the same way. A pass compared with the evaluation
of the same predictions under the same settings reads `same protocol`; a pass
on a subset of an evaluation's items, with per-item values, reads
`compared on N shared files`.

### 3.5 Problem codes of a set document and a pass

| Code | Where | Meaning |
|---|---|---|
| `not_object` | set, pass | Not a JSON object (or a field that must be an object is not) |
| `kind` | set | `kind` is not `euler.validation_set` |
| `schema_version` | set | `schemaVersion` does not have major 1 |
| `key` | set | `key` does not match the key pattern |
| `metric_set` | set | No string `metricNamespace`, or `axes` are not valid axis declarations |
| `protocol` | set | `protocol` is not an object or its `schemaVersion` major is not 2 |
| `items` | set | `protocol.items.ids` is not a list of unique strings, or `protocol.items.series` is not empty |
| `step` | pass | `step` is missing or not an integer ≥ 0, or `epoch` is not an integer |
| `metrics` | pass | `metrics` is missing, not an object, or empty |
| `not_a_number` | pass | A value in `metrics` is not a finite number |
| `no_decomposition` | pass | A metric or per-item name does not decompose under the set's axes |
| `no_definition` | pass | The base of a metric or per-item name has no definition in the set's protocol |
| `undescribed` | pass | A key of `parameters` or `observations` has no descriptor in the set's protocol |
| `parameter_value` | pass | A pass parameter is not a scalar or a list of at most 64 scalars |
| `population` | pass | `population.ids` is not a list of unique strings |
| `length` | pass | A `per_item` or `series` list does not have one entry per id, or there are no ids |

Paths are dotted from the document root (`metrics.<full name>`,
`parameters.<key>`, `per_item.<name>`). A consumer that meets any of these
still judges what it can: an undescribed key is compared strictly (contract
8.4: definition, equal), a name that does not decompose is `not comparable ·
metric name does not decompose`, unknown ids are `evaluated files not
recorded`. Nothing is treated as equal because it could not be read.

### 3.6 Stream events

A producer that mirrors a run to a consumer over a stream sends two event
types, and only to a consumer that advertised the capability
`validation_sets/1` (euler-view's stream session response lists its
capabilities). Consumers that do not advertise it never receive them; the
files are always written.

```json
{ "type": "validation_set", "eventId": "…", "key": "cerema", "document": { "…set document…" } }
{ "type": "validation", "eventId": "…", "key": "cerema", "record": { "…pass record…" } }
```

## 4. Checkpoint selection

A checkpoint entry of `meta.json` MAY record how it was selected:

| Field | Meaning |
|---|---|
| `selection_metric` | The full metric name the selection read, e.g. `sparsedepth.eval.metric.standard.image_mean.absrel` |
| `selection_value` | Its value at this checkpoint |
| `selection_direction` | `lower` or `higher`; SHOULD come from the metric's definition |
| `validation_set` | The validation set key, when the metric is a validation-set name |

## 5. The naming report

A producer that checks a 2.0 run records what failed in `meta.json`:

```json
"naming_report": {
  "schema_version": "2.0",
  "checked": true,
  "validator": "euler-metric-naming 0.4.0",
  "problems": [
    { "source": "val.jsonl", "key": "val/sky/mask_iou", "code": "no_namespace", "count": 12, "first_step": 1259 }
  ],
  "truncated": 0
}
```

`source` is `train.jsonl`, `val.jsonl`, or `validation/<key>` (pass problems
and declaration problems of a set). `count` is how often the problem occurred
and `first_step` the first step it occurred at. A producer keeps at most 200
entries and counts the rest in `truncated`. `checked: false` means no
validator was available (the producer then records why in `message`).

## 6. Versioning

- Training naming: `schema_version` `MAJOR.MINOR` of `metric_naming`; this
  document defines 2.0. A minor may only add optional fields that a consumer
  can ignore without reading a value as something it is not. Anything else is
  a new major.
- Validation sets: `schemaVersion` of the set document; this document defines
  1.0, with the same rule. Pass records have no version of their own; they
  follow their set's.
- The protocol inside a set follows the contract's own versioning (schema 2);
  vocabulary changes never change any of these versions.
- A consumer that meets an unsupported major of either reads the run as
  schema 1 (training naming) or reports the set as not interpretable and
  judges nothing from it (validation sets).

## 7. Conformance

[`contract/fixtures/training/`](../euler_metric_naming/contract/fixtures/training/)
holds the shared material both implementations pass: declarations with their
expected problems, metric keys with their expected problem or definition, set
documents and pass records with their expected problems, and verdicts between
validation passes and the conformance evaluations of `fixtures/evals/`. See the
fixtures' README for the formats.
