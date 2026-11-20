# Changelog

## 0.4.0

### Added

- **Training naming 2.0** ([docs/training.md](docs/training.md)): a training
  run's `metric_naming` declaration carries `schema_version: "2.0"` and, per
  namespace, contract metric definitions (`metrics`) and axis roles
  (`axisRoles`). `MetricNamespace(metrics=..., namespace_metrics=...,
  axis_roles=...)` writes it; `sys.train` gets the new `euler.train`
  vocabulary terms. A run that declares 2.0 is held to it for every key.
- **Validation sets and passes**: a validation set is an evaluation of the
  live model on a fixed population, declared once (`set.json`: metricSet +
  contract-v2 protocol) and recorded per pass (`passes.jsonl`).
  `load_validation_pass` / `loadValidationPass` turn a pass into an
  evaluation for the unchanged verdict engine, so a validation curve is
  judged against an evaluation (`same protocol`, `compared on N shared
  files`, …).
- Validators with shared problem codes and paths, Python and TS:
  `training_naming_problems`, `training_metric_problem`,
  `metric_value_problem`, `validation_set_problems`,
  `validation_pass_problems`; definition readers `training_metric_definition`,
  `training_contract`; `validation_pass_contract`, `evaluation_metric_ids`.
- `ProtocolBuilder.describe(key)`: a descriptor snapshot without a value.
- `MetricDescription.to_definition()` / `from_definition()`.
- Vocabulary `euler.train` 1.0 (training telemetry, `"context": "train"`);
  `euler.depth` 1.1 adds `depth.gt.valid_range_m` and `depth.extra_mask`
  (defaults `null` and `false`), so an in-training validation that masks
  differently from an evaluation reads `not comparable` instead of equal.
- JSON Schemas `training-naming-2`, `validation-set-1`, `validation-pass-1`,
  also in the TS data; the vocabulary schema gains `context`.
- Conformance fixtures `contract/fixtures/training/` (declarations, keys,
  sets, passes, contracts and verdicts), run by both implementations.
- The cross-repository rollout plan,
  [docs/training-alignment-plan.md](docs/training-alignment-plan.md).

### Changed

- The conformance eval files snapshot `euler.depth@1.1` (vocabulary tags
  only; no verdict, hash or definition key changes).
- Schema-1 `training_naming_config()` output is unchanged.

## 0.3.0

### Added

- **Evaluation protocol contract, schema 2** (`euler_metric_naming.contract`,
  normative definition in [docs/contract.md](docs/contract.md)). The
  `protocol` block of a results file now carries descriptors, references,
  observations, population, stratifications, reducers, axis roles, metric
  definitions and per-item series, so any consumer judges any two evaluations
  of any modality with one generic algorithm. Stays readable by schema-1
  consumers: `protocol.parameters` keeps the v1 shape.
- `ProtocolBuilder` for producers: snapshots the vocabulary terms it uses,
  computes the population manifest, validates the closed set.
- Vocabularies `euler.core`, `euler.sky`, `euler.depth`, `euler.rgb`,
  `euler.rays`, `euler.points3d` (1.0) and an authoring guide,
  [docs/vocabulary.md](docs/vocabulary.md).
- The verdict engine (`compare`, `compare_files`): `same protocol`,
  `compared on N shared files`, `comparable · …`, `not comparable · …` per
  metric cell, with reasons, measures and hints; `recompute` /
  `recompute_cell` for values on shared files.
- RFC 8785 canonicalization, `sha256:` hashes, population manifests and
  definition keys; the three-valued condition language; the five observation
  measures and their phrases.
- The frozen schema-1 legacy adapter (`contract/legacy/protocol-1.json`):
  euler-eval ≤ 2.47 files normalize into v2 contracts.
- `load_eval_file`: whole `eval.json` files flattened as euler-view does.
- `references.<role>.identityScheme` (`euler.reference/2`; the legacy
  adapter's `euler.reference/1`): references whose identities come from
  different schemes are reported as `identity recorded in an older / another
  evaluation format` (reason code `identity_format`) unless what both record
  differs.
- `relevant_keys` / `relevantKeys`: the keys relevant to a metric cell under
  §8.4, from the engine's own code path.
- Open-ended stratification bins (`null` first / last edge) and an implicit
  per-metric `reducer` for cells without a reduction axis (RGB, depth image
  quality, FID), so such metrics pair on shared files too.
- JSON Schemas `protocol-2.schema.json` and `vocabulary-1.schema.json`
  (draft 2020-12).
- A dependency-free TS artifact for consumers (`contract/ts/`), generated
  data included, type-checked under strict TypeScript and runnable with
  Node's type stripping.
- Conformance fixtures (`contract/fixtures/`) both implementations pass,
  including 33 eval files and 49 hand-derived verdicts.
- Package data: schemas, vocabularies, the legacy table, the TS artifact and
  the fixtures ship in the wheel.

### Changed

- `jsonschema` joins the `dev` extra (tests only); the runtime stays
  dependency-free.
