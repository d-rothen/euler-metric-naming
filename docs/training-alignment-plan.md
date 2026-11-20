# Plan: aligning training metrics with the evaluation contract

Status: in progress. This is the cross-repository plan for the work specified
in [training.md](training.md). It is written so a later session can resume
from it: the decisions are fixed, the deliverables have acceptance tests, and
the checklist at the end is kept up to date as each step lands.

## 1. Why

Production training runs (euler-view, November 2026) showed that validation
scores were logged under names that pack the dataset, masking variant,
prediction head, reduction and even the evaluator's version into the base
name (`depth.train.diag.valset_muses_no_sky_absrel`,
`…valset_muses_scope_euler_eval_minor`), that one run mixed percent and
fraction for the same quantity, and that a validation curve and the final
evaluation of the same checkpoint measured different things under similar
names (64 frames at a 200 m cap against 333 frames at a 240 m cap with GT sky
masking). Nothing could tell a reader whether two such numbers were
comparable. The evaluation contract already answers that question for
evaluations; this plan makes training runs speak it.

## 2. Decisions (approved)

1. **Validation scores use the evaluator's own names**, in per-set files, not
   a contract extension. A validation set is an evaluation in contract terms;
   its passes are judged against evaluations with the unchanged engine.
2. **The protocol of a validation set lives in its own file**, written once
   (`validation/<key>/set.json`), with a small pointer in `meta.json`, because
   euler-train rewrites `meta.json` on every logged step.
3. **Per-item values during validation are written by default** for sets of up
   to about 1000 items, opt-in above that, so passes pair with evaluations on
   shared files.
4. **The ×100 primary validation of dehaze-baseline (rds val) stays
   telemetry**, with an honest definition (unit %, no evaluation id), until the
   trainer moves it to euler-eval as its own docs plan.
5. **Unknown keys in euler-train are warned about once and reported** in
   `meta.json`; a strict mode that raises is opt-in, for CI and smoke tests.
   Values are never dropped.
6. **A clean cut, no transition.** New runs implement the new names from the
   start: no dual writing, no legacy names in new runs, no mixed old and new
   names inside one run. A run that declares training naming 2.0 is held to it
   fully. Old runs stay as they are. Because of this, the model repositories
   are changed last, after everything else is done and deployed.
7. **Sky and other slash-named validation metrics** move into validation sets
   once euler-eval's validation API supports them; until then they are not
   declared.
8. **Checkpoint selection is recorded** in `meta.json` checkpoint entries:
   `selection_metric`, `selection_value`, `selection_direction`,
   `validation_set`.

Principles carried over from the contract work:

- Adding a metric, axis, validation set or modality needs no consumer code
  change: definitions and protocols travel with the run as snapshots.
- Anything a consumer cannot interpret is reported, never treated as equal.
- New producer output stays importable by the euler-view deployed today.
- Old runs not being comparable with new ones is acceptable.

## 3. Order of work and releases

| Order | Repository | Version | Depends on |
|---|---|---|---|
| 1 | euler-metric-naming | 0.4.0 | — |
| 2 | euler-eval | 3.2.0 | euler-metric-naming 0.4.0 |
| 3 | euler-train | 2.15.0 | euler-metric-naming 0.4.0 (optional `naming` extra) |
| 4 | euler-view | next | euler-metric-naming 0.4.0 TS artifact |
| 5 | model repositories (dehaze-baseline, UniFuse, any other euler-train user) | — | all of the above, deployed |

The user publishes every release. euler-eval and euler-train pin
euler-metric-naming to a git source on the `training-naming-v2` branch while
0.4.0 is unreleased; the pin is replaced by `>=0.4` before they are merged,
once 0.4.0 is on PyPI.

## 4. Deliverables and acceptance tests

### 4.1 euler-metric-naming 0.4.0

Deliverables:

- `docs/training.md` (normative spec) and this plan; links from
  `docs/contract.md`, `docs/api.md`, `docs/vocabulary.md` and the README.
- Training naming 2.0 in `MetricNamespace`: `metrics=` (definitions shared by
  every modality namespace) and `namespace_metrics=` (per namespace key);
  `training_naming_config()` writes schema 2.0 when either is given (schema 1
  otherwise, unchanged), with `axisRoles`, `metrics`, derived
  `metricDescriptions`, and the `euler.train` terms in `sys.train`.
  `MetricDescription.to_definition()`.
- `euler_metric_naming.contract`: `training_naming_problems`,
  `training_metric_problem`, `training_metric_definition`,
  `training_contract`, `validation_set_problems`, `validation_pass_problems`,
  `validation_pass_contract`, `load_validation_pass`; the same in the TS
  artifact (`trainingNamingProblems`, …, `loadValidationPass`).
- `ProtocolBuilder.describe(key)`: snapshot a descriptor without a value, for
  keys that passes record.
- Vocabularies: `euler.train` 1.0 (telemetry terms); `euler.depth` 1.1 adds
  `depth.gt.valid_range_m` (GT validity bounds an in-loop validation applied,
  default `null`) and `depth.extra_mask` (an additional validity mask was
  applied, default `false`), both `definition` keys, so an in-loop
  validation that masks differently from the evaluator reads "not comparable"
  instead of silently equal.
- JSON Schemas `training-naming-2`, `validation-set-1`, `validation-pass-1`,
  shipped in the wheel and in the generated TS data.
- Conformance fixtures `contract/fixtures/training/`, generated where they are
  built (set documents, pass records) and hand-derived where they are expected
  (problems, verdicts), run by the Python suite and `conformance.test.ts`.

Acceptance:

- The full suite passes on Python 3.9 and the current Python; ruff is clean.
- `scripts/generate_ts.py --check` and `scripts/generate_fixtures.py --check`
  are clean; the node conformance suite passes (node ≥ 22).
- Every training fixture case gives the same problem codes and paths, the same
  definitions and the same verdicts (tier, label, codes, shared, recomputed
  values) in Python and TS.
- A schema-1 `training_naming_config()` is byte-identical to 0.3.0's.

### 4.2 euler-eval 3.2.0

Deliverables:

- `euler_eval.validation.DepthValidationSet` (dense and sparse depth GT): built
  from the GT-only dataset of `build_validation_gt_dataset` and the settings
  the CLI knows (`alignment` none / affine / auto_affine, `sky_depth` cap,
  `benchmark_depth_range`, `spatial_alignment`), plus in-loop only settings
  that are recorded honestly (`valid_depth_range`, an extra validity mask).
  `declaration()` returns the set document; a pass accumulator takes dataset
  samples and in-memory predictions and returns the pass record; accumulators
  merge across processes.
- Protocol helpers shared by the CLI and the validation set (parameters,
  observations, references, axis roles, vocabularies), moved out of `cli.py`
  without changing any CLI output except the `euler.depth` vocabulary version
  tag (1.0 to 1.1).
- `DepthValidationSet.from_protocol(eval.json)`: settings taken from an
  existing evaluation; settings the in-loop path cannot reproduce (GT sky
  masking, predicted-sky masking, sky IoU) are refused with a message.
- The euler-train payload of a CLI evaluation names the results files it wrote
  (`metadata.results_files`), instead of copying protocols into `meta.json`.
- `requirements.txt` back in sync with `pyproject.toml`.
- Docs: `docs/validation.md` (validation sets), `docs/output.md`, CHANGELOG.

Acceptance (the decisive test, `tests/test_validation_set.py`):

- Synthetic dense GT and predictions on disk. The CLI evaluates them
  (`--depth-alignment none`, a benchmark range). A `DepthValidationSet` over
  the same GT dataset scores the same predictions in memory. Loading the pass
  with `load_validation_pass` and the CLI's `eval.json` with `load_eval_file`,
  `compare_files` gives **`same protocol`** for every standard metric cell the
  pass writes (every reduction, every bin), and every pass aggregate equals the
  CLI aggregate.
- The same with the pass on a subset of the items: **`compared on N shared
  files`**, and `recompute_cell` on the shared items gives equal values on both
  sides, equal to the pass's own aggregates.
- The same with `--depth-alignment affine` against an `affine` set (metric
  space).
- A set that applies a validity range or an extra mask reads `not comparable`
  with the key named.
- The full suite passes (with euler-connector on `PYTHONPATH`).

### 4.3 euler-train 2.15.0

Deliverables:

- `run.declare_validation_set(key, document)` writes
  `validation/<key>/set.json` once and the `meta.json` pointer; re-declaring a
  key with a different document raises.
- `run.log_validation(key, record)` (or with keyword fields) appends to
  `validation/<key>/passes.jsonl`; never to `val.jsonl`.
- Naming checks for runs that declare 2.0: every key of `log()` and of passes,
  once per distinct key, through euler-metric-naming when installed; warnings
  once per key; `naming_report` in `meta.json`; `strict_naming=True` in
  `init()` raises instead. Schema-1 runs behave exactly as before.
- Checkpoint selection fields in `save_checkpoint` / `log_saved_checkpoint`.
- Stream: `validation_set` and `validation` events only when the session
  advertises `validation_sets/1`.
- `meta-schema.json`: `metric_naming` 2.0 fields, `validation_sets`,
  `naming_report`, checkpoint selection fields.
- Docs: `docs/api.md`, `docs/run-format.md`, `docs/streaming.md`, CHANGELOG.

Acceptance:

- A run that declares 2.0 and logs only declared keys has an empty
  `naming_report`; undeclared, undecomposable, undefined, evaluation-named and
  non-numeric keys each appear once with their code and count; strict mode
  raises on the first.
- A run without a declaration, or with schema 1, writes byte-identical records
  to 2.14.0 and no `naming_report`.
- Validation sets: files and pointer as specified; nothing in `val.jsonl`;
  re-declaration rules; the records load with `load_validation_pass`.
- A stream session without the capability never sends the new event types.
- The full suite passes on Python 3.9 and 3.11; `meta-schema.json` passes
  `check-jsonschema --check-metaschema`.

### 4.4 euler-view (later round)

- Store each run's definition snapshot and resolve training names through the
  contract's definition reader; "best" from the definition's direction.
- Import validation sets and passes (curves per set) and judge passes against
  evaluations with the existing verdict engine; show the verdict next to the
  curve in the run dossier.
- Advertise `validation_sets/1` in the stream session response and ingest the
  two event types.
- MCP: definitions in metric tools, direction-aware search, a read-only
  validation-versus-evaluation comparison.
- Show the naming status of 2.0 runs (`naming_report`).

### 4.5 Model repositories (last)

- dehaze-baseline: external validation sets through `DepthValidationSet`;
  telemetry with definitions; checkpoint selection by validation-set name and
  definition direction; no flattened `valset_*` names.
- UniFuse and others: definitions for their telemetry (`metrics=`).

## 5. Compatibility with the euler-view deployed today

Checked against euler-view `main` at 68c84699 (code reading, not a test run):

- Extra `metric_naming` fields (`schema_version`, `axisRoles`, `metrics`) are
  ignored: the declaration is only checked to be an object, and descriptions
  are read from `metricDescriptions`, which 2.0 still writes.
- Extra `meta.json` keys (`validation_sets`, `naming_report`) are stored with
  the rest of `meta.json` and not read.
- Extra checkpoint fields (`selection_direction`, `validation_set`) go into
  the checkpoint's metadata; `selection_metric` and `selection_value` are
  already read.
- `validation/` files are not imported. New-format validation curves are
  therefore not visible until euler-view's step lands; this is accepted
  (decision 6).
- Unknown stream event types would make the deployed ingest fail its whole
  batch (an unknown type gives no event summary, and the summary column is not
  nullable), so the new events are only sent to a session that advertises the
  capability.
- `val.jsonl` keeps its format and never carries evaluation names, so no
  series mixes two sets (the deployed tables have no uniqueness per step).

## 6. Enforcement rollout

1. euler-metric-naming ships the validators and the fixtures (this release).
2. euler-train checks 2.0 runs and reports (warn once, `naming_report`);
   strict mode is opt-in.
3. Model repositories turn strict mode on in their CI smoke runs when they
   adopt 2.0 (step 5).
4. euler-view shows the naming status and never rejects a run.

## 7. Out of scope

- Migrating or relabelling existing runs (clean split).
- RGB, rays, points-3D and sky validation sets (euler-eval's validation API
  covers depth only; decision 7).
- Any change to the contract's closed set or verdict algorithm.
- Comparing telemetry with evaluations, or across runs by verdict.
- Proving parity of sparse in-loop validation against the CLI end to end (the
  sparse set is built from the same helpers and tested on its own; a CLI parity
  test needs a sparse GT fixture on disk, listed as follow-up below).

## 8. Status checklist

Updated as each step lands. Commit hashes are on the `training-naming-v2`
branches of each repository.

- [x] Setup documents written (`docs/training.md`, this plan) — 4696dc3.
- [x] euler-metric-naming 0.4.0: training naming 2.0 in `MetricNamespace` — 33c2819.
- [x] euler-metric-naming 0.4.0: validators, definitions, validation-pass
      materialization (Python) — 33c2819.
- [x] euler-metric-naming 0.4.0: the same in the TS artifact — 33c2819.
- [x] euler-metric-naming 0.4.0: vocabularies `euler.train` 1.0, `euler.depth` 1.1 — 33c2819.
- [x] euler-metric-naming 0.4.0: JSON Schemas, conformance fixtures (Python and TS) — 33c2819.
- [x] euler-metric-naming 0.4.0: docs, CHANGELOG, version; full suite on 3.9
      (473 passed) and 3.13 with the contract extra (474 passed); node
      conformance 20 of 20; `generate_ts.py --check` and
      `generate_fixtures.py --check` clean; ruff clean.
- [ ] euler-eval 3.2.0: shared protocol helpers; `DepthValidationSet`.
- [ ] euler-eval 3.2.0: decisive test (same protocol, equal values; subset paired).
- [ ] euler-eval 3.2.0: euler-train payload pointers; requirements.txt; docs.
- [ ] euler-train 2.15.0: validation sets, naming checks, checkpoint selection, stream gate.
- [ ] euler-train 2.15.0: meta-schema, docs; full suite.
- [ ] euler-metric-naming 0.4.0 released by the user (PyPI); git pins replaced.
- [ ] euler-view (next round).
- [ ] Model repositories (last).
- [ ] Follow-up: sparse in-loop versus CLI parity test with an on-disk sparse GT fixture.
