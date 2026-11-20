"""Hashes of ``docs/contract.md`` §9: format, population manifest, definition key."""
from __future__ import annotations

import hashlib
import unicodedata
from typing import Any, Iterable, Mapping

from ._json import is_number
from .conditions import evaluate
from .descriptors import effective_compare
from .jcs import canonicalize
from .strata import identity_projection


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_value(value: Any) -> str:
    """``"sha256:" + hex(SHA-256(UTF-8(JCS(value))))`` (§9.2)."""
    return "sha256:" + sha256_hex(canonicalize(value).encode("utf-8"))


def manifest_ids(ids: Iterable[str]) -> list[str]:
    """Item ids NFC-normalized, de-duplicated and sorted by code point (§9.3)."""
    normalized = set()
    for item in ids:
        if not isinstance(item, str):
            raise TypeError(f"item id {item!r} is not a string")
        normalized.add(unicodedata.normalize("NFC", item))
    return sorted(normalized)


def manifest_hash(ids: Iterable[str]) -> str:
    """The population manifest hash of a set of item ids (§9.3)."""
    return hash_value(manifest_ids(ids))


def definition_key_object(contract: Mapping[str, Any]) -> dict:
    """The object whose hash is the definition key (§9.4)."""
    parameters = _record(contract.get("parameters"))
    observations = _record(contract.get("observations"))
    descriptors = _record(contract.get("descriptors"))
    task = contract.get("task")

    references = {}
    for role, reference in _record(contract.get("references")).items():
        if not isinstance(reference, dict):
            continue
        view = reference.get("view")
        if isinstance(view, dict) and isinstance(view.get("source"), str):
            references[role] = {"source": view["source"]}
        else:
            references[role] = {"identity": reference.get("identity")}

    params = {}
    for key, value in parameters.items():
        descriptor = descriptors.get(key)
        descriptor = descriptor if isinstance(descriptor, dict) else None
        if effective_compare(descriptor) != "equal":
            continue
        applies = descriptor.get("appliesWhen") if descriptor else None
        if applies is not None and evaluate(applies, parameters, observations) is False:
            continue
        params[key] = value

    stratifications = {
        sid: identity_projection(stratification)
        for sid, stratification in _record(contract.get("stratifications")).items()
        if isinstance(stratification, dict)
    }

    metrics = {}
    for base, definition in _record(contract.get("metrics")).items():
        rev = definition.get("rev") if isinstance(definition, dict) else None
        if valid_rev(rev):
            metrics[base] = rev

    return {
        "task": task if isinstance(task, str) else None,
        "references": references,
        "parameters": params,
        "stratifications": stratifications,
        "metrics": metrics,
    }


def definition_key(contract: Mapping[str, Any]) -> str:
    """The definition key of a (normalized) contract (§9.4)."""
    return hash_value(definition_key_object(contract))


def valid_rev(rev: Any) -> bool:
    """A recorded ``rev``: an integral JSON number >= 1 (``2.0`` counts as 2)."""
    return is_number(rev) and float(rev).is_integer() and rev >= 1


def _record(value: Any) -> dict:
    return value if isinstance(value, dict) else {}
