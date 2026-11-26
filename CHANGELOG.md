# Changelog

## 0.5.0

### Added

- **Installed vocabularies** (`euler.vocabularies` entry points, the suite's
  extension protocol from `euler-dataset-contract` 0.23): a package declares
  `"mylab.thermal" = "my_lab.euler:VOCAB"` and `load_vocabulary("mylab.thermal")`,
  `ProtocolBuilder(..., ["euler.core", "mylab.thermal"])` use it in every
  process of that environment. The value is a mapping, a JSON path (absolute
  or relative to the declaring module) or a zero-argument callable returning
  either. See [docs/vocabulary.md](docs/vocabulary.md#installing-a-vocabulary-from-your-own-package).
- **One vocabulary registry**: shipped and installed vocabularies go through
  the same registration (validated as `vocabulary-1` documents, id equal to
  the registered name, registered once) and carry a `provider`
  (`builtin` or `entry_point` with distribution and version).
  `vocabulary_registry()`, `reload_vocabularies()`, `builtin_vocabularies()`.
  A shipped id cannot be redefined (`shadows_builtin`); conflicts, import
  failures and invalid documents are reported in `.problems`, and
  `load_vocabulary` of a rejected id raises with the reason.
  `EULER_EXTENSIONS_STRICT=1` raises instead.
- `vocabulary_problems(document)`: dependency-free `vocabulary-1` validation
  (checked against the JSON Schema in the test suite); `VocabularyError`,
  `UnknownVocabularyError` (a `KeyError`).
- **Capability report** `metric_vocabulary_report` 1.0
  (`euler_metric_naming.capabilities.capability_report`), registered in
  `euler.capabilities` as `euler-metric-naming`: naming grammar, protocol /
  training naming / validation set versions, every vocabulary with version,
  term counts and provider, the shipped schemas, and extension problems.
  `euler-suite report` and euler-view read it per environment.

### Changed

- `euler-dataset-contract>=0.23.0` is now a runtime dependency (it has no
  dependencies itself); modality names are always checked against its
  registry.
- `available_vocabularies()` and `vocabulary_documents()` include installed
  vocabularies; `vocabulary_documents(builtin_only=True)` gives the shipped
  ones (the TS artifact is generated from these, so its data is unchanged
  apart from the package version).
- Plain vocabulary documents passed to `VocabularySet` / `ProtocolBuilder` are
  validated and raise `VocabularyError` when invalid.

### Removed

- The `contract` extra (the contract is a dependency now) and its CI job.

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
