# Changelog

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
- JSON Schemas `protocol-2.schema.json` and `vocabulary-1.schema.json`
  (draft 2020-12).
- A dependency-free TS artifact for consumers (`contract/ts/`), generated
  data included, type-checked under strict TypeScript and runnable with
  Node's type stripping.
- Conformance fixtures (`contract/fixtures/`) both implementations pass,
  including 28 eval files and 41 hand-derived verdicts.
- Package data: schemas, vocabularies, the legacy table, the TS artifact and
  the fixtures ship in the wheel.

### Changed

- `jsonschema` joins the `dev` extra (tests only); the runtime stays
  dependency-free.
