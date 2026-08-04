# Contributing

Bug reports, documentation improvements, and focused API changes are welcome.
For behavior changes, please describe the producer and consumer affected: a
naming change can otherwise look harmless while breaking stored metric paths.

## Development setup

```bash
git clone https://github.com/d-rothen/euler-metric-naming.git
cd euler-metric-naming
uv sync --extra dev          # or: pip install -e ".[dev]"
```

The core package intentionally has no dependencies. To exercise the optional
modality registry integration as well, add the `contract` extra:

```bash
uv sync --extra dev --extra contract
```

## Checks

Run the test and style checks before opening a pull request:

```bash
uv run pytest
uv run ruff check .
```

The suite does not require datasets, a GPU, or network access. New public
behavior should include both a focused unit test and, when serialization is
involved, a round-trip or exact payload assertion.

To inspect the distributions that will be uploaded to PyPI:

```bash
uv build
uvx twine check dist/*
```

## Compatibility expectations

- Existing metric strings and serialized field names are public contracts.
- Axis order comes from `AxisDeclaration.position`, not dictionary key order.
- Optional axes must always leave at least one segment for the base metric.
- The dependency-free install must continue to work without
  `euler-dataset-contract`.
- Public symbols should remain importable from `euler_metric_naming` and be
  listed in `__all__`.

## Releasing

1. Update `__version__` in `euler_metric_naming/__init__.py`; package metadata
   reads it from there.
2. Update user-facing documentation when the contract changes.
3. Run the checks above and build the distributions.
4. Tag the release `v<version>` and push the tag.

The publish workflow builds the tag and uploads it to PyPI with trusted
publishing (OIDC); no API token is stored in the repository.
