"""Modality validation against the euler-dataset-contract registry."""
from __future__ import annotations

from euler_dataset_contract import get_modality_meta_fields


def validate_modality(name: str) -> str | None:
    """Return a warning message if *name* is not a known modality, or None."""
    try:
        fields = get_modality_meta_fields(name)
    except (KeyError, ValueError):
        return f"unknown modality {name!r} (not in euler-dataset-contract registry)"
    if fields is None:
        return f"unknown modality {name!r} (not in euler-dataset-contract registry)"
    return None
