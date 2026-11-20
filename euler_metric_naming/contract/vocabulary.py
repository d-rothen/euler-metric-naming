"""Vocabularies (``docs/contract.md`` §5): load, look up and snapshot terms.

A vocabulary is a JSON document shipped in ``contract/vocabularies/``. Producers
copy the terms they use into each file; consumers never need a vocabulary to
judge a file. :class:`VocabularySet` is the producer-side lookup over several
vocabularies, used by :class:`~euler_metric_naming.contract.ProtocolBuilder`.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import Any, Iterable, Optional

_PACKAGE = "euler_metric_naming.contract"


def _read_json(*parts: str) -> Any:
    node = resources.files(_PACKAGE)
    for part in parts:
        node = node / part
    return json.loads(node.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class Vocabulary:
    """One vocabulary document."""

    id: str
    version: str
    document: dict

    @property
    def tag(self) -> str:
        """``<id>@<version>``, the ``vocabulary`` field of snapshotted terms."""
        return f"{self.id}@{self.version}"

    @property
    def descriptors(self) -> dict:
        return self.document.get("descriptors", {})

    @property
    def metrics(self) -> dict:
        return self.document.get("metrics", {})

    @property
    def reducers(self) -> dict:
        return self.document.get("reducers", {})

    @property
    def stratification_schemes(self) -> dict:
        return self.document.get("stratificationSchemes", {})


def available_vocabularies() -> list[str]:
    """Ids of the vocabularies shipped with this package."""
    folder = resources.files(_PACKAGE) / "vocabularies"
    return sorted(
        entry.name[: -len(".json")]
        for entry in folder.iterdir()
        if entry.name.endswith(".json")
    )


@lru_cache(maxsize=None)
def _load(vocabulary_id: str) -> Vocabulary:
    if vocabulary_id not in available_vocabularies():
        raise KeyError(f"unknown vocabulary {vocabulary_id!r}")
    document = _read_json("vocabularies", f"{vocabulary_id}.json")
    return Vocabulary(id=document["id"], version=document["version"], document=document)


def load_vocabulary(vocabulary_id: str) -> Vocabulary:
    """Load a shipped vocabulary by id (``euler.depth``)."""
    return _load(vocabulary_id)


def vocabulary_documents() -> dict:
    """Every shipped vocabulary document, by id (deep copies)."""
    return {vid: copy.deepcopy(load_vocabulary(vid).document) for vid in available_vocabularies()}


class VocabularySet:
    """Look up terms across several vocabularies.

    A term found in more than one vocabulary is ambiguous and raises; pass
    the term explicitly instead.
    """

    def __init__(self, vocabularies: Iterable[Any]) -> None:
        loaded = []
        for item in vocabularies:
            if isinstance(item, Vocabulary):
                loaded.append(item)
            elif isinstance(item, dict):
                loaded.append(Vocabulary(id=item["id"], version=item["version"], document=item))
            else:
                loaded.append(load_vocabulary(item))
        self._vocabularies = tuple(loaded)

    @property
    def vocabularies(self) -> tuple:
        return self._vocabularies

    def _find(self, section: str, name: str) -> Optional[tuple]:
        found = [
            (vocabulary, getattr(vocabulary, section)[name])
            for vocabulary in self._vocabularies
            if name in getattr(vocabulary, section)
        ]
        if len(found) > 1:
            tags = ", ".join(vocabulary.tag for vocabulary, _ in found)
            raise ValueError(f"{section[:-1]} {name!r} is ambiguous: defined in {tags}")
        return found[0] if found else None

    def descriptor(self, key: str) -> Optional[dict]:
        """A snapshot of the descriptor of *key*, tagged with its vocabulary."""
        found = self._find("descriptors", key)
        if found is None:
            return None
        vocabulary, term = found
        return {**copy.deepcopy(term), "vocabulary": vocabulary.tag}

    def metric(self, base: str) -> Optional[dict]:
        """A snapshot of the metric definition of base name *base*."""
        found = self._find("metrics", base)
        if found is None:
            return None
        vocabulary, term = found
        return {**copy.deepcopy(term), "vocabulary": vocabulary.tag}

    def reducer(self, value: str) -> Optional[dict]:
        """A snapshot of the reducer of reduction axis value *value*."""
        found = self._find("reducers", value)
        if found is None:
            return None
        vocabulary, term = found
        return {**copy.deepcopy(term), "vocabulary": vocabulary.tag}

    def stratification_scheme(self, scheme: str) -> Optional[dict]:
        found = self._find("stratification_schemes", scheme)
        return copy.deepcopy(found[1]) if found else None
