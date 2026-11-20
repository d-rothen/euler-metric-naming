# Evaluation protocol contract, version 2

This document is the normative definition of the `protocol` block (schema
`2.x`) that evaluation producers write into a results file, and of how a
consumer compares two evaluations with it. The key words MUST, MUST NOT,
SHOULD and MAY are used as in RFC 2119.

The JSON Schema is
[`euler_metric_naming/contract/schema/protocol-2.schema.json`](../euler_metric_naming/contract/schema/protocol-2.schema.json)
(draft 2020-12, `$id`
`https://github.com/d-rothen/euler-metric-naming/contract/protocol-2.schema.json`).
Where this document and the schema disagree, this document wins and the schema
is a bug.

## 1. Design goals

1. A consumer judges every evaluation, of any modality or task, with one
   generic algorithm (section 8). Adding a parameter, metric, stratification,
   reducer, observation or a whole modality changes only the producer and its
   vocabulary; consumers need no code change.
2. The semantics a consumer must understand form a **small closed set**
   (section 3). Everything else is data that travels with each file.
3. Anything a consumer cannot interpret is reported as a difference
   ("not comparable · <key> differs / not recorded"), never silently equal.
4. Identities that consumers compare (definition keys, population manifests)
   are byte-exact across languages (section 9).

## 2. Where the block lives and compatibility with v1 consumers

The whole contract is nested inside the existing top-level `protocol` object
of `eval.json`. Version-1 consumers (euler-view before the v2 importer) accept
it unchanged because they

- read only `protocol.schemaVersion` (any non-empty string) and
  `protocol.parameters`,
- silently drop every other property of `protocol`, and
- reject unknown **top-level** blocks and unknown properties of per-file
  entries, which v2 never adds.

Therefore `protocol.parameters` stays a flat map of JSON scalars or lists of
at most 64 scalars (strings at most 1024 characters, at most 512 keys, keys at
most 255 characters), exactly as in v1. Everything else of v2 lives in sibling
properties of `protocol`. Per-item sufficient statistics live in
`protocol.items` (not in the metric tree), so a v1 consumer never shows them
as metrics.

## 3. The closed set (what every consumer must understand)

Changing anything in this section is a **major** version change.

### 3.1 Roles (`descriptor.role`)

| Role | Meaning | Default `compare` |
|---|---|---|
| `reference` | Identity of an input other than the prediction (GT, calibration, label space, benchmark items) | `equal` |
| `definition` | A setting or resolved choice that defines the value of the metrics it affects | `equal` |
| `population` | A setting whose only effect is which items are evaluated | `pair` |
| `domain` | A measured property of what was scored (evaluated region, resampling, truncation) | `measure` |
| `derived` | Follows from other keys (`derivedFrom`); compared through those | `ignore` |
| `provenance` | Recorded for reproducibility; never changes values | `ignore` |

### 3.2 Compare rules (`descriptor.compare`)

| Rule | A difference means |
|---|---|
| `equal` | not comparable (`incomparable`) |
| `pair` | a population difference, resolved by the population rule (8.7) |
| `measure` | comparable, with the measure of `descriptor.observe` (3.3); unmeasurable → `incomparable` |
| `ignore` | nothing |

An unknown role or compare value MUST be treated as `equal`.

### 3.3 Observation kinds and their measures (`descriptor.observe`)

A measure turns two values into a non-negative number `d` (0 = no
difference) and a phrase. A consumer implements exactly these five:

| Kind | Value shape | `d` | Phrase (`x` = `d` in percent, one decimal) |
|---|---|---|---|
| `region` | `{"frame": string, "box": [top, left, height, width]}` (numbers ≥ 0) | `1 − IoU(boxA, boxB)`; frames differ → unmeasurable | `<label> differs by x %` |
| `count` | number ≥ 0 | `|a − b| / max(|a|, |b|)`, 0 when both are 0 | `<label> differs by x %` |
| `fraction` | number in [0, 1] | `|a − b|` | `<label> differs by x pp` |
| `factor` | number > 0, or a list of them of equal length | `max_i(max(a_i/b_i, b_i/a_i)) − 1` | `<label> (A vs B)` |
| `size` | list of numbers ≥ 0 | relative difference of the products, as `count` | `<label> differs by x %` |

A value that does not have the declared shape, a missing value on one side,
or an unknown kind is unmeasurable. The tolerance is zero: every `d > 0` is
reported.

Number formatting in phrases, identical in every implementation:

- `x` (percent, one decimal): `t = floor(d × 1000 + 0.5)` in IEEE double
  arithmetic, printed as `t / 10` with exactly one decimal (`3.7`, `0.9`,
  `12.0`); a non-zero `d` with `t = 0` prints as `<0.1`. For `fraction` the
  unit is `pp` instead of `%`.
- Factor values `A`, `B`: each number formatted like ECMAScript
  `Number.prototype.toPrecision(3)`, then trailing zeros of the fraction and a
  trailing `.` removed (`1.0417` → `1.04`, `1` → `1`); a list joins its
  numbers with `×` (`1.04×1.01`).
- Stratum edges in `bins differ` phrases use the same 3-significant-digit
  format.

### 3.4 Display scales (`metric.display.scale`)

| Scale | Fields | Comparison colouring |
|---|---|---|
| `relative` | — | relative difference of the values |
| `complement` | — (bound from `metric.bounds[1]`, default 1) | relative difference of `bound − value` |
| `absolute` | `fullAt` (number > 0), `unit` | plain difference, full tone at `fullAt` |
| `none` | — | no colouring |

### 3.5 Directions (`metric.direction`)

`lower`, `higher`, `none`.

### 3.6 Metric kinds (`metric.kind`)

| Kind | Meaning |
|---|---|
| `score` | A metric shown to users (default) |
| `statistic` | A per-item sufficient statistic (e.g. a count used as a pooling weight). Hidden from metric lists |
| `observation` | A measured property, not a score. Hidden from metric lists |

### 3.7 Reducer recompute operations (`reducer.recompute`)

How a consumer recomputes an aggregate from per-item values on a subset of
items:

| Op | Recompute |
|---|---|
| `mean` | arithmetic mean of the per-item values (nulls skipped) |
| `median` | median of the per-item values (mean of the two middle values for even n) |
| `pool` | the metric's `pool` form, weighted by its weight statistic (3.8) |
| `none` | not recomputable from per-item values |

### 3.8 Pool forms (`metric.pool.form`)

| Form | Pooled value from per-item values `v_i` and weights `w_i` |
|---|---|
| `mean` | `Σ w_i v_i / Σ w_i` |
| `rms` | `sqrt(Σ w_i v_i² / Σ w_i)` |
| `none` | not poolable |

Items with a null value or weight, or weight 0, are skipped.

### 3.9 Axis roles (`protocol.axes[*].role`)

`space`, `reduction`, `stratum`, `variant`, `family`, `stage`, `kind`. Only
`reduction`, `stratum` and `space` change consumer behaviour (8.1); the others
are carried for display. An unknown axis role is treated as `family`.

### 3.10 Condition operators (section 7)

`all`, `any`, `not`, and the comparisons `exists`, `eq`, `ne`, `lt`, `le`,
`gt`, `ge`, `in`.

### 3.11 Verdict tiers

| Tier | Label pattern |
|---|---|
| `same` | `same protocol` |
| `paired` | `compared on N shared files` |
| `comparable` | `comparable · <reason>` |
| `incomparable` | `not comparable · <reason>` |

Severity order: `same` < `paired` < `comparable` < `incomparable`.

## 4. The `protocol` object

All properties except `schemaVersion` and `parameters` are optional; a missing
optional block means "not recorded" and is handled strictly (section 8).
Consumers MUST ignore unknown properties anywhere.

```json
{
  "schemaVersion": "2.0",
  "task": "depth.dense",
  "vocabularies": { "euler.core": "1.0", "euler.depth": "1.0" },
  "parameters":   { "<key>": scalar | [scalar, ...] },
  "descriptors":  { "<key>": Descriptor },
  "references":   { "<role>": Reference },
  "observations": { "<key>": ObservationValue },
  "population":   Population,
  "stratifications": { "<id>": Stratification },
  "reducers":     { "<reduction axis value>": Reducer },
  "axes":         { "<metricSet axis name>": AxisRole },
  "metrics":      { "<base metric name>": MetricDefinition },
  "items":        Items
}
```

### 4.1 Keys

Parameter and observation keys are dotted lower-case paths whose first segment
names the vocabulary area (`depth.cap.max_m`, `sky.gt.masking`,
`spatial.resize_target`, `lm.fewshot.k`). Reference roles are compared under
the descriptor key `reference.<role>`. Parameter, observation and
`reference.<role>` keys share one namespace: a key MUST NOT be both a
parameter and an observation.

### 4.2 Descriptor

Every key in `parameters` and `observations`, and every `reference.<role>`,
SHOULD have a descriptor in `descriptors`. The descriptors are a snapshot of
the vocabulary terms the file uses, so consumers can judge keys they have
never seen.

| Field | Type | Meaning |
|---|---|---|
| `label` | string | Human label used in verdict phrases (default: the key) |
| `role` | 3.1 | Required |
| `compare` | 3.2 | Default by role |
| `observe` | 3.3 | Measure kind, for `compare: measure` |
| `unit` | string \| null | Display unit of the value |
| `affects` | Affects | Which metric cells the key can change (default: all) |
| `appliesWhen` | Condition | The key only matters when this holds (default: always) |
| `inertIf` | Condition | The key had no effect in this evaluation when this holds (default: never) |
| `derivedFrom` | [key] | For role `derived` |
| `default` | JSON value | The value meant when the key is not recorded (for keys added in later vocabulary versions) |
| `hint` | string | Advice shown with an `incomparable` verdict (e.g. "compare the native space, or two fitted runs") |
| `description` | string | Free text |
| `vocabulary` | string | `<vocabulary id>@<version>` the term comes from |

`Affects` is `{"metrics": [glob, ...], "axes": {"<axis role>": [value, ...]}}`.
A metric cell is affected when its base metric name matches one of the
`metrics` globs (default `["*"]`) **and**, for each listed axis role, the
cell's value for that role is in the list. A cell without a value for a listed
axis role (e.g. an unbinned metric and `"stratum"`) is not affected.

Globs: `*` matches any run of characters, dots included; every other character
is literal; matching is case-sensitive and anchored at both ends.

### 4.3 Reference

```json
{
  "modality": "depth",
  "dataset": { "id": "cerema", "variant": "full_crop", "split": null, "scope": "depth" },
  "identity": "sha256:…",
  "manifest": { "hash": "sha256:…", "count": 1600 },
  "view": { "source": "cerema/rgb_left@1024x1920", "region": [270, 170, 707, 1730] }
}
```

- `identity` (required): a string that is equal exactly when the producer
  considers the inputs identical. euler-eval uses
  `sha256(JCS({dataset, manifest, head}))`, where `head` is the canonical
  dataset head without volatile fields.
- `view` (optional): the reference is a sub-region of a larger source frame.
  `region` is `[top, left, height, width]` in source pixels.
- A role that was not configured is absent (not `null`).

### 4.4 Population

```json
{
  "unit": "image",
  "idScheme": "euler.item_id/1",
  "count": 680,
  "manifest": "sha256:…",
  "accounting": { "reference": 1600, "prediction": 680, "paired": 680,
                  "dropped": { "segmentation": 0, "sky_mask": 0 } }
}
```

`manifest` is the population manifest hash (9.3) of the ids of all evaluated
items. `accounting` is informational.

**Item ids (`euler.item_id/1`).** An item's id is the per-file tree path of
the results file: the `children` keys from the root down, then the file
entry's `id`, joined with `~` (U+007E). Producers MUST emit ids in Unicode
NFC. This is the same string euler-view stores as `per_file_eval.file_id`.

### 4.5 Observations

`observations` maps keys to values whose shape is given by the key's
descriptor `observe` kind (3.3). Observations are never part of the
definition key.

### 4.6 Stratification

```json
{
  "label": "GT depth",
  "variable": "gt_depth_m",
  "unit": "m",
  "edges": [0.01, 3.811, 14.475, 32.0],
  "labels": ["near", "mid", "far"],
  "closed": "left",
  "lastClosed": true,
  "total": "all",
  "categories": null,
  "after": ["depth.cap.max_m"],
  "scheme": "sqrt_equal_width",
  "description": "…"
}
```

Either `edges` (n+1 numbers, ascending) with `labels` (n strings), or
`categories` (list of strings) for categorical strata. `total` names the
stratum label that covers the union. The **identity projection** of a
stratification is the object without `label`, `description` and `unit`. Two
stratifications are the same exactly when their identity projections are
equal (8.1).

### 4.7 Reducer

`{"recompute": "mean" | "median" | "pool" | "none", "over": "item" | "observation" | "dataset", "label": "…", "description": "…"}`
keyed by the value of the axis whose role is `reduction`.

### 4.8 Axis role

`{"role": "<3.9>", "stratification": "<stratification id>"}` keyed by the
`metricSet.axes` axis name. `stratification` is required for role `stratum`.

### 4.9 Metric definition

Keyed by the base metric name: the metric name after the namespace and all
consumed axis segments are removed (euler-metric-naming `decompose`).

| Field | Type | Default |
|---|---|---|
| `id` | global id, e.g. `depth.absrel` | — |
| `kind` | 3.6 | `score` |
| `rev` | integer ≥ 1, the definition revision | — (not recorded) |
| `displayName`, `description` | string | — |
| `direction` | 3.5 | — |
| `bounds` | `[min \| null, max \| null]` | — |
| `unit` | string \| null | — |
| `display` | `{"scale": 3.4, "fullAt"?, "unit"?}` | `{"scale": "relative"}` |
| `perItem` | boolean: per-item values are written | `false` |
| `pool` | `{"form": 3.8, "weight": "<base name of a statistic>"}` | `{"form": "none"}` |
| `requires` | [reference role] | all roles (strict) |
| `dependsOn` | [glob over keys] | `["*"]` (strict) |

A consumer's own registry (display names, directions) is a fallback for
metrics without a definition only.

### 4.10 Items

```json
{
  "ids": ["scene:1~time_of_day:day~weather:fog100~frame-0", "…"],
  "series": { "<full per-item metric name>": [number | null, "…"] }
}
```

Per-item statistic and observation series, aligned with `ids`. A consumer
stores them like per-file metrics. Score metrics stay in `per_file_metrics`.

### 4.11 Metric names, per-item names and weights

- The aggregate metric name is `namespace.axes….base`.
- Its **per-item name** is the same name with the segment of the axis whose
  role is `reduction` removed. Per-item values of score metrics are written in
  `per_file_metrics` under that name.
- The **weight series** of an aggregate with `recompute: pool` is the
  per-item name with the base replaced by `metric.pool.weight`
  (e.g. `depth.eval.metric.standard.near.absrel` →
  `depth.eval.metric.standard.near.count`), found in `items.series`.

## 5. Vocabularies

A vocabulary is a JSON document
`{"id": "euler.depth", "version": "1.2", "descriptors": {…}, "metrics": {…}, "reducers": {…}, "stratificationSchemes": {…}}`.
Vocabularies live in
[`euler_metric_naming/contract/vocabularies/`](../euler_metric_naming/contract/vocabularies/).
Producers copy the terms they use into each file (`descriptors`, `metrics`,
`reducers`). A consumer MAY keep a table of the vocabularies it has seen; it
MUST NOT need it to judge a file.

Versioning of a vocabulary: adding a term is a minor bump; changing a term's
`role`, `compare`, `observe`, `affects`, `appliesWhen`, `inertIf` or a
metric's `requires`, `dependsOn`, `pool` is a major bump. Changing what a
metric computes MUST increase that metric's `rev`.

## 6. Versioning of the protocol schema

- `schemaVersion` is `"MAJOR.MINOR"`. This document defines major 2.
- **Minor** (2.0 → 2.1): adding optional properties, new descriptor fields
  that consumers may ignore without changing verdicts, documentation.
  Consumers MUST accept any minor of a major they support and ignore
  unknown properties.
- **Major**: any change to section 3 (roles, compare rules, observation kinds
  and measures, display scales, directions, metric kinds, reducer and pool
  operations, axis roles, condition operators and their semantics, tiers),
  to the meaning of an existing property, or to section 9 (hashing).
- A consumer that meets an unsupported major MUST compare such evaluations
  only by strict equality of `parameters` over the union of keys (the v1
  rule) and say so.
- Vocabulary changes never change `schemaVersion`.

## 7. Condition language

Used by `appliesWhen` and `inertIf`. A condition is a JSON object of one of
these forms:

```json
{ "all": [Condition, …] }
{ "any": [Condition, …] }
{ "not": Condition }
{ "param": "<key>", "op": "<op>", "value": <JSON> }
{ "observation": "<key>", "op": "<op>", "value": <JSON> }
```

`value` is omitted for `exists`. The operand of `param` is
`parameters[key]`; of `observation` it is `observations[key]`.

**Values.** Evaluation uses three truth values: `true`, `false`, `unknown`.

| Op | Operand missing or `null` | Otherwise |
|---|---|---|
| `exists` | `false` | `true` |
| `eq` | `unknown` | JSON equality (8.0) of operand and `value` |
| `ne` | `unknown` | not `eq` |
| `lt`, `le`, `gt`, `ge` | `unknown` | numeric comparison when both are numbers (not booleans), else `unknown` |
| `in` | `unknown` | `true` when `value` is a list containing an element JSON-equal to the operand; `unknown` when `value` is not a list |

- `all`: `false` if any member is `false`, else `unknown` if any is
  `unknown`, else `true` (empty list → `true`).
- `any`: `true` if any member is `true`, else `unknown` if any is `unknown`,
  else `false` (empty list → `false`).
- `not`: swaps `true` and `false`; `unknown` stays `unknown`.
- A malformed condition (unknown op, wrong shape) evaluates to `unknown`.

**Use.** `appliesWhen` is evaluated on each side with that side's parameters
and observations; a key is skipped only when it is `false` on **both** sides
(`unknown` counts as applying). `inertIf` is evaluated likewise; a key is
skipped only when it is `true` on **both** sides (`unknown` counts as not
inert).

## 8. Comparing two evaluations (the verdict algorithm)

**8.0 JSON equality.** Numbers are equal when numerically equal (`1 == 1.0`;
`true` is not a number). Strings compare code point by code point. Lists are
equal when they have the same length and pairwise-equal elements. Objects are
equal when they have the same key set and equal values. `null` equals only
`null`.

Input: two normalized contracts `A` and `B` (a v1 file is first normalized by
the legacy adapter, section 10), a metric name `m` present in both, and, when
known, the evaluated item id sets and whether per-item series exist on each
side. Output: a verdict `{tier, label, shared?, reasons[]}`; each reason is
`{tier, code, key?, label, text, measure?, hint?}`.

The verdict is computed per metric cell; the tier is the most severe reason
tier (`same` when there is none). The label is the label pattern of the tier
(3.11) with the first reason of that tier as `<reason>`. Reasons are ordered
by tier (most severe first), then by the step that produced them, then by key.

**8.1 Cell.** Decompose `m` with the metricSet axes of each side; record the
values of the axes whose role is `space`, `reduction` and `stratum`; look up
the metric definition by base name (`A`'s if present, else `B`'s; when both
exist and differ in `rev`, see 8.2). A name that does not decompose on one side
yields `incomparable`, code `schema`.

**8.2 Revision.** Both definitions record `rev` and they differ →
`incomparable`, code `revision`, text `<metric> definition revised (rev a vs b)`.
Exactly one side records `rev` → `incomparable`, code `not_recorded`.

**8.3 Stratification.** If the cell has a stratum value: the axis's
stratification on each side (via `axes[axis].stratification`); missing on
either side → `incomparable` `not_recorded`; identity projections differ →
`incomparable`, code `stratification`, text `bins differ (<bin> <a> vs <b>)`
where each side is `<lo>–<hi>` of that bin (the total bin spans the first to
the last edge), followed by ` <unit>` when the stratification has one, e.g.
`bins differ (near 0.01–3.81 m vs 0.01–11.6 m)`. When the bin has no edges on
a side (categorical, or the label is missing there) the text is
`bins differ (<bin>)`.

**8.4 Relevance of a key.** A key `k` is relevant to the cell when
(i) it matches a glob of the metric's `dependsOn` (default `*`), (ii) its
merged descriptor's `affects` matches the cell (4.2), (iii) for `reference.<r>`
keys, `r` is in the metric's `requires` (default: all), and (iv) neither the
`appliesWhen` nor the `inertIf` rule of section 7 skips it.

**Merged descriptor.** When both sides describe `k`, the stricter wins field
by field: `compare` by `equal` > `measure` > `pair` > `ignore`; conditions
are taken from the side whose descriptor is used for `compare`. When only one
side describes `k`, that descriptor is used. When neither does, `k` is
`{"role": "definition", "compare": "equal"}`.

**8.5 References.** For each relevant `reference.<r>` present on either side:

- missing on one side → `incomparable`, code `not_recorded`;
- `identity` equal → nothing;
- both have `view` with equal `source` → `comparable`, code `domain`, measure
  `region` over the two view regions (frame = `source`), text
  `<label> region differs by x %`;
- otherwise → `incomparable`, code `reference`, text `<label> differs`.

**8.6 Parameters and observations.** For each relevant key in the union of
both sides' `parameters` and `observations`:

- missing on one side: use the descriptor's `default` for that side if
  defined; otherwise `incomparable`, code `not_recorded`, text
  `<label> not recorded`;
- by merged `compare`:
  - `ignore` → nothing;
  - `equal` → JSON-unequal values → `incomparable`, code `parameter`,
    text `<label> differs`, with `hint`;
  - `pair` → unequal values → nothing here; the population rule decides;
  - `measure` → `d > 0` → `comparable`, code `domain`, with the measure phrase;
    unmeasurable → `incomparable`, code `parameter`.

**8.7 Population.** Let `IA`, `IB` be the evaluated item id sets.

- Both manifests recorded and equal, or both id sets known and equal →
  nothing.
- Otherwise, when both id sets are known: `N = |IA ∩ IB|`.
  - `N = 0` → `incomparable`, code `population_disjoint`,
    text `no shared files`.
  - The cell is **pairable** when its reduction (if any) has
    `recompute ≠ none`, per-item series exist on both sides, and for `pool`
    the metric has `pool.form ≠ none` and weight series exist on both sides.
    Pairable → `paired`, code `population_paired`, `shared = N`, text
    `compared on N shared files`. The consumer recomputes both values on the
    shared items.
  - Not pairable → `comparable`, code `population_partial`,
    text `computed on different files: N of M shared` with
    `M = max(|IA|, |IB|)`.
- Id sets unknown on either side → `incomparable`, code
  `population_unknown`, text `evaluated files not recorded`. Equal counts
  never stand in for equal files.

**8.8 Unsupported major.** If either side's `schemaVersion` major is not
supported, compare `parameters` by strict equality over the union of keys;
a difference → `incomparable`, code `schema`.

## 9. Canonical forms and hashes

**9.1 JSON canonicalization** is RFC 8785 (JCS): objects with keys sorted by
UTF-16 code units, no whitespace, strings escaped as ECMAScript
`JSON.stringify` does (`\"`, `\\`, `\b`, `\f`, `\n`, `\r`, `\t`, other
controls as lower-case `\u00xx`, everything else literal), numbers serialized
as ECMAScript `Number.prototype.toString` (shortest round-trip, exponent form
`1e+21` / `1e-7` outside [1e-6, 1e21), `-0` → `0`). Non-finite numbers and
lone surrogates are errors. Integers outside ±(2^53 − 1) are errors.

**9.2 Hash format.** `"sha256:" + lower-case hex of SHA-256 over the UTF-8
bytes of the JCS text`.

**9.3 Population manifest.** The hash of the JSON array of the item ids,
NFC-normalized, de-duplicated and sorted by Unicode code point (equivalently
by their UTF-8 bytes).

**9.4 Definition key.** The hash of this object:

```json
{
  "task": <protocol.task or null>,
  "references": { "<role>": {"source": view.source} | {"identity": identity} },
  "parameters": { "<key>": value  for keys whose effective compare is "equal"
                  and whose appliesWhen is not false on this side },
  "stratifications": { "<id>": identity projection },
  "metrics": { "<base>": rev  for definitions with a rev }
}
```

`references` uses `view.source` when the reference has a view (so crops of
one source share a key) and `identity` otherwise. Equal definition keys mean
no definitional (`equal`) difference for any metric; a consumer MAY group
evaluations by it. Verdicts are still computed per cell.

## 10. The legacy adapter (schema 1.x)

Files with `schemaVersion` `1.x` are normalized into a v2 contract by a frozen
table (`contract/legacy/protocol-1.json`) and the rules below. The adapter may
also read the file's `metricSet` (axes, metadata) and `meta`.

| v1 key | v2 |
|---|---|
| `reference.<r>.{dataset,scope,split}` (+ `reference.gt.modality`) | `references.<r>` with `identity = sha256(JCS({modality, dataset, scope, split}))`; all-null → absent |
| `population.samples` | `population.count` |
| `population.gt_sky_masking` | `sky.gt.masking` |
| `population.missing_segmentation` | `sky.gt.missing_policy` (default `"drop"`, applies when `sky.gt.masking`) |
| `population.pred_sky_masking` | `sky.pred.masking` |
| `population.evaluated_resolution` | observation `domain.grid` (`size`) |
| `population.benchmark_depth_range_m` | stratification `depth.gt_range` (edges from `metricSet.metadata.benchmark.boundaries`, else recomputed with the square-root rule) and parameter `depth.benchmark.range_m` (`derived`) |
| `method.metrics_revision` r | policy parameters: `spatial.resize_target` (`prediction_grid` if r < 2 else `gt_grid`), `spatial.mask_resample` (`nearest` / `nearest_exact`), `rgb.depth_bins.policy` (`v1` / `v2`), `rays.fov_policy` (`v1` / `v2`), `depth.invalid.policy` (`score_sentinel` if r < 3 else `exclude_sentinel`), `rgb.input.alignment_policy` (`upscale` if r < 4 else `follow_gt`) |
| `method.depth_alignment` | `depth.alignment.mode` (`provenance`) and, when `metricSet.metadata.calibration_applied` is known, `depth.alignment.applied` (`definition`, affects `space = metric`) |
| `method.sky_depth_m` | `depth.cap.max_m` |
| `method.pred_sky_mask` | `sky.pred.fill` |
| `method.spatial_alignment` | `spatial.method` (`derived`); plus observation `prediction.resample` (`factor`) for `resize_prediction` / `warp_prediction` and parameter `reference.gt.resample` (`definition`, a factor list) for `resize` / `warp` on RGB and rays, both from `meta.gt.dimensions` / `meta.pred.dimensions` when present |
| `method.fid_backend` | `rgb.fid.backend` (affects `fid`, `kid*`) |
| `method.input_alignment` | `rgb.input.alignment` (affects `hautiere_*`) |
| `method.points_3d_alignment` | `points3d.alignment.mode` (`provenance`) and `points3d.alignment.applied` from `calibration_applied` (`auto` ≡ `none`) |

Axis roles for v1 files: `space` → `space`, `category` → `family`,
`reduction` → `reduction`, `bin` → `stratum` (`depth.gt_range`), `variant` →
`variant`. Reducers: `image_mean` → `mean`, `image_median` → `median`,
`pixel_pool` → `none` (v1 has no weights). v1 metrics have no definitions, so
`requires` and `dependsOn` are strict.

## 11. Conformance

[`euler_metric_naming/contract/fixtures/`](../euler_metric_naming/contract/fixtures/)
holds the golden material every implementation MUST pass:

- `jcs.json` and `hashes.json`: canonical text and hashes;
- `conditions.json`: condition, context, expected truth value;
- `evals/`: golden results files (v2 and v1), including a synthetic modality
  no producer emits;
- `verdicts.json`: the expected verdict for each (file A, file B, metric)
  case: tier, label, reason codes, shared count, measures.

A consumer is conformant when it reproduces every expected value in these
files.
