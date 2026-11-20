# Authoring vocabulary terms

The [comparison contract](contract.md) keeps what consumers must understand
small and closed (§3). Everything else — which parameters exist, what they
mean, which metrics they affect, how metrics pool — is **data**: vocabulary
terms that producers copy into every results file. A consumer judges a file
from that snapshot alone, so adding a parameter, metric, stratification,
reducer, observation or a whole modality never needs a consumer change.

This guide is for producers (euler-eval, or any evaluator) adding terms.

## Where terms live

Vocabularies are JSON documents in
[`euler_metric_naming/contract/vocabularies/`](../euler_metric_naming/contract/vocabularies/),
validated by
[`vocabulary-1.schema.json`](../euler_metric_naming/contract/schema/vocabulary-1.schema.json):

```json
{
  "id": "euler.depth",
  "version": "1.0",
  "title": "…",
  "descriptors": { "depth.cap.max_m": { "label": "depth cap", "role": "definition", "…": "…" } },
  "metrics": { "absrel": { "id": "depth.absrel", "kind": "score", "rev": 1, "…": "…" } },
  "reducers": { "image_mean": { "recompute": "mean", "over": "item", "label": "image mean" } },
  "stratificationSchemes": { "sqrt_equal_width": { "label": "…", "description": "…" } }
}
```

| Vocabulary | Holds |
|---|---|
| `euler.core` | reference roles (`reference.gt`, `reference.calibration`, …), the scored domain (`domain.region`, `domain.grid`), resampling (`prediction.resample`, `spatial.*`), `core.seed`, the generic reducers |
| `euler.sky` | GT sky masking, predicted-sky masking / fill / exclusion, sky IoU |
| `euler.depth` | alignment, depth cap, invalid values, conversion, benchmark range, the GT validity range and extra validity mask of in-training validation (1.1), the depth metrics and the `count` statistic |
| `euler.rgb` | FID backend, input alignment, depth-binned photometric settings, RGB metrics |
| `euler.rays` | FoV policy, ρ_A threshold, ray metrics |
| `euler.points3d` | gauge alignment, F_A threshold, point-map metrics |
| `euler.train` | training telemetry, not evaluation metrics (`"context": "train"`): `lr`, `grad_norm` and the GPU values euler-train records ([training.md](training.md) §2.6) |

Terms in a vocabulary carry no `vocabulary` field; when a producer copies a
term it adds `"vocabulary": "<id>@<version>"`. `ProtocolBuilder` does this
for you:

```python
from euler_metric_naming.contract import ProtocolBuilder

builder = ProtocolBuilder("depth.dense", ["euler.core", "euler.sky", "euler.depth"])
builder.param("depth.cap.max_m", 38)          # descriptor copied from euler.depth@1.1
builder.metric("absrel")                      # definition copied, rev 1
builder.param("lab.knob", 3, descriptor={     # a term no vocabulary has
    "label": "knob", "role": "definition"})
```

A key that is in no loaded vocabulary raises unless you pass its descriptor.

A vocabulary of training telemetry says `"context": "train"`; every other
vocabulary is an evaluation vocabulary, and a training definition never
carries the `id` of one of its metrics.

## Adding a parameter

1. Pick a dotted lower-case key whose first segment is its area
   (`depth.`, `sky.`, `rgb.`, …). Keys share one namespace with observations
   and `reference.<role>`.
2. Choose the **role** (§3.1) by what a difference means:

   | If a different value… | role | compare (default) |
   |---|---|---|
   | changes what the metrics mean | `definition` | `equal` |
   | identifies different inputs | `reference` (use `references`, not a parameter) | `equal` |
   | only changes which items are evaluated | `population` | `pair` |
   | is a measured property of what was scored | `domain` (an observation) | `measure` |
   | follows from other keys | `derived` (+ `derivedFrom`) | `ignore` |
   | never changes values | `provenance` | `ignore` |

   When in doubt, `definition`: a strict difference is always safe.
3. Narrow it where true:
   - `affects` — which cells it can change: `{"metrics": ["fid", "kid*"]}`,
     `{"axes": {"space": ["metric"]}}`. Unlisted means every cell.
   - `appliesWhen` — when the key matters at all, e.g. `sky.gt.missing_policy`
     only under GT sky masking. Skipped only when false on **both** sides.
   - `inertIf` — when it had no effect in this run, e.g. the depth cap when
     nothing was clipped (`{"observation": "depth.cap.clipped", "op": "eq",
     "value": 0}`). Skipped only when true on **both** sides.
   - `default` — the value meant by files that predate the key.
   - `hint` — what to do instead when it makes a cell incomparable.
4. Give it a short `label`: it is the subject of verdict phrases
   (`not comparable · depth cap differs`).

Conditions (§7) are three-valued: a missing or `null` operand is unknown, and
unknown always counts as "applies" / "not inert" — strict.

## Adding an observation

An observation is a measured property (`domain.region`, `depth.cap.clipped`).
Record it with `builder.observe(key, value)`. Give it role `domain` and one of
the five `observe` kinds (§3.3: `region`, `count`, `fraction`, `factor`,
`size`) when a difference should be reported as *comparable with a measure*
(`comparable · domain differs by 0.9 %`); give it role `provenance` when it
only feeds conditions (`inertIf`). A shape the kind does not accept is
unmeasurable and therefore not comparable.

## Adding a metric

Key the definition by the **base metric name** your metric tree writes (the
name after namespace and axis segments, as `decompose` returns it), with a
global `id`:

| Field | Guidance |
|---|---|
| `id` | `<area>.<name>`, unique across vocabularies |
| `kind` | `score`; `statistic` for per-item weights (hidden); `observation` for measured properties |
| `rev` | start at 1; increase whenever what the metric computes changes |
| `displayName`, `description`, `unit` | display |
| `direction` | `lower`, `higher` or `none` |
| `bounds` | `[min, max]`, `null` for open ends |
| `display` | `relative` (default), `complement` (bounded scores such as δ, SSIM, IoU: relative difference of `bound − value`), `absolute` with `fullAt` (dB: PSNR at 2), or `none` |
| `perItem` | `true` when per-item values are written to `per_file_metrics` |
| `pool` | `{"form": "mean" \| "rms", "weight": "<statistic base>"}` when a pooled aggregate equals the weighted mean (or RMS) of per-item values; `{"form": "none"}` otherwise (SILog, quantiles, FID) |
| `reducer` | for metrics written **without** a reduction-axis segment: the reducer that produced the aggregate from the per-item values of the same name — `image_mean` when the aggregate is exactly the mean of the per-file values (nulls skipped), `dataset` for FID/KID. Omit when the aggregate is not such a reduction (pooled quantiles such as RGB tail errors, means that leave items out such as RGB edge F1) |
| `requires` | the reference roles the value depends on; omit for "all roles" (strict). Narrow only when provable (no-reference RGB metrics read no reference: `[]`; Hautière: `["input"]`) |
| `dependsOn` | globs over keys; `["*"]` unless the metric provably ignores whole areas |

Write per-item values of scores in `per_file_metrics` under the per-item name
(the aggregate name without its reduction segment) and pooling weights in
`protocol.items.series` (`builder.series(name, values)`), aligned with
`items.ids`. Then a consumer can compare evaluations on their shared files
instead of calling them different.

## Adding a stratification

Describe binned metrics (`…image_mean.near.absrel`) with a stratification and
an axis of role `stratum` that names it:

```python
from euler_metric_naming.contract import sqrt_stratification

builder.stratification("depth.gt_range", sqrt_stratification(0.01, 80))
builder.axis("bin", "stratum", stratification="depth.gt_range")
```

Give `edges` with `labels` (or `categories`), `total` for the union bin
(the first and last edge may be `null` for an open end, e.g. `[0, 1, 5, null]`),
`variable`, `unit`, and `after` (keys applied before binning). Everything but
`label`, `description`, `unit` and null fields is its identity: two
evaluations' bins are the same only when those are equal. A named `scheme`
must be a `stratificationSchemes` entry of a loaded vocabulary.

## Adding a reducer

Reducers are keyed by the value of the axis with role `reduction` and say how
to recompute the aggregate on a subset (§3.7): `mean`, `median`, `pool`
(uses the metric's `pool`), or `none`. Reuse `euler.core`'s `image_mean`,
`image_median`, `pixel_pool`, `dataset`, or pass your own:
`builder.reducer("clip_mean", {"recompute": "mean", "over": "item", "label": "clip mean"})`.

## Adding a modality

Nothing in consumers changes. Declare the metricSet axes as usual
(`MetricNamespace(..., context="eval", axes=...)`), give each axis a role
with `builder.axis`, and describe every key, metric and reducer — from a new
vocabulary, or inline. The conformance fixtures `evals/s1.json`–`s6.json`
are a modality (`audio.eval`) no vocabulary knows; euler-view judges them with
the same code as depth.

To ship the terms for reuse, add `contract/vocabularies/<id>.json`, validate
it (`uv run pytest`), and regenerate the TS data
(`python scripts/generate_ts.py`).

## Versioning

- A vocabulary's `version` is `MAJOR.MINOR`. Adding a term is a minor bump.
- Changing a term's `role`, `compare`, `observe`, `affects`, `appliesWhen`,
  `inertIf`, or a metric's `requires`, `dependsOn`, `pool`, is a major bump.
- Changing what a metric computes MUST increase that metric's `rev` (files
  with different `rev`s are not comparable for it).
- Labels, descriptions and hints may change in a minor bump.
- A new optional field of the protocol schema is only a minor when ignoring
  it makes an older consumer stricter, never laxer (contract §6).
- Vocabulary changes never change the protocol `schemaVersion`; files carry
  their own snapshot, so old files keep meaning what they meant.
- The legacy table (`contract/legacy/protocol-1.json`) is frozen: it holds
  its own copies of the 1.0 terms and never follows later vocabularies.
