"""Soft dependency on euler-dataset-contract for modality validation."""
from __future__ import annotations

try:
    from euler_dataset_contract import (  # type: ignore[import-not-found]
        get_modality_meta_fields,
    )

    _has_contract = True
except ImportError:
    _has_contract = False


def validate_modality(name: str) -> str | None:
    """Return a warning message if *name* is not a known modality, or None."""
    if not _has_contract:
        return None
    try:
        fields = get_modality_meta_fields(name)
    except (KeyError, ValueError):
        return f"unknown modality {name!r} (not in euler-dataset-contract registry)"
    if fields is None:
        return f"unknown modality {name!r} (not in euler-dataset-contract registry)"
    return None
