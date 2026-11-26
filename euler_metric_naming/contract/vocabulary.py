"""Vocabularies (``docs/contract.md`` §5): one registry, shipped and installed.

A vocabulary is a ``vocabulary-1`` JSON document of terms producers snapshot
into each results file; consumers never need a vocabulary to judge a file.
:class:`VocabularySet` is the producer-side lookup over several vocabularies,
used by :class:`~euler_metric_naming.contract.ProtocolBuilder`.

Every vocabulary of an environment is in one registry and got there the same
way: the documents shipped in ``contract/vocabularies/`` and the ones other
packages install through the suite's extension protocol
(:mod:`euler_dataset_contract.extensions`), group ``euler.vocabularies``::

    [project.entry-points."euler.vocabularies"]
    "mylab.thermal" = "my_lab.euler:THERMAL_VOCABULARY"

The entry-point name is the vocabulary id. The value is a vocabulary document
(a mapping), a path to a JSON file (absolute, or relative to the directory of
the module that defines the value), or a zero-argument callable returning
either. Registration rules, the same for shipped and installed documents:

* the document must be a valid ``vocabulary-1`` document
  (:func:`~euler_metric_naming.contract.vocabulary_problems`);
* its ``id`` must be the name it is registered under;
* an id is registered once: a shipped vocabulary cannot be redefined by an
  installed one, and two installed packages offering the same id load neither.

A rejected vocabulary is reported, never hidden: it is listed in
:func:`vocabulary_registry` ``.problems`` (and so in the
``metric_vocabulary_report`` of ``euler-suite report``), and
:func:`load_vocabulary` of its id raises with the reason. With
``EULER_EXTENSIONS_STRICT=1`` building the registry raises instead.
"""
from __future__ import annotations

import copy
import json
import os
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from euler_dataset_contract.extensions import (
    ExtensionError,
    ExtensionProblem,
    load_extensions,
    strict_mode,
)

from .validate import vocabulary_problems

_PACKAGE = "euler_metric_naming.contract"
_DISTRIBUTION = "euler-metric-naming"

#: The extension-point group installed vocabularies are declared in.
VOCABULARIES_GROUP = "euler.vocabularies"

#: Codes of :class:`~euler_dataset_contract.extensions.ExtensionProblem` this
#: module adds to the protocol's own (``conflict``, ``load_failed``,
#: ``invalid``, ``api_version``).
SHADOWS_BUILTIN = "shadows_builtin"


class VocabularyError(ValueError):
    """A vocabulary document breaks a rule; ``problems`` lists every one."""

    def __init__(self, problems: Iterable[str], *, subject: str = "vocabulary") -> None:
        self.problems = list(problems)
        super().__init__(f"invalid {subject}: " + "; ".join(self.problems))


class UnknownVocabularyError(KeyError):
    """No vocabulary of that id is registered in this environment."""

    def __str__(self) -> str:  # KeyError would quote the message
        return str(self.args[0]) if self.args else ""


@dataclass(frozen=True)
class Vocabulary:
    """One vocabulary document and where it came from.

    ``provider`` is ``{"kind": "builtin", "distribution", "version"}`` for a
    shipped vocabulary, ``{"kind": "entry_point", "name", "target",
    "distribution", "version"}`` for an installed one and ``{"kind":
    "inline"}`` for a document passed to :class:`VocabularySet` directly.
    """

    id: str
    version: str
    document: dict
    provider: Mapping[str, Any] = field(
        default_factory=lambda: {"kind": "inline"}, compare=False
    )

    @property
    def tag(self) -> str:
        """``<id>@<version>``, the ``vocabulary`` field of snapshotted terms."""
        return f"{self.id}@{self.version}"

    @property
    def context(self) -> str:
        """``eval`` (evaluation metrics, the default) or ``train`` (telemetry)."""
        return self.document.get("context", "eval")

    @property
    def builtin(self) -> bool:
        return self.provider.get("kind") == "builtin"

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

    def term_counts(self) -> dict:
        return {
            "descriptors": len(self.descriptors),
            "metrics": len(self.metrics),
            "reducers": len(self.reducers),
            "stratificationSchemes": len(self.stratification_schemes),
        }

    def describe(self) -> dict:
        """The registry entry as plain data (the capability report's shape)."""
        return {
            "id": self.id,
            "version": self.version,
            "tag": self.tag,
            "context": self.context,
            "title": self.document.get("title"),
            "terms": self.term_counts(),
            "provider": dict(self.provider),
        }


@dataclass(frozen=True)
class VocabularyRegistry:
    """Every vocabulary registered in this environment, and what was rejected."""

    vocabularies: Mapping[str, Vocabulary]
    problems: tuple = ()

    def ids(self) -> list[str]:
        return sorted(self.vocabularies)

    def problem(self, vocabulary_id: str) -> Optional[ExtensionProblem]:
        for problem in self.problems:
            if problem.name == vocabulary_id:
                return problem
        return None


# -- building the registry ---------------------------------------------------------------


def _package_version() -> str:
    from .. import __version__

    return __version__


def _builtin_sources() -> list:
    folder = resources.files(_PACKAGE) / "vocabularies"
    return sorted(
        (entry for entry in folder.iterdir() if entry.name.endswith(".json")),
        key=lambda entry: entry.name,
    )


def _document(value: Any, origin: Optional[Path]) -> dict:
    """A vocabulary value (mapping, JSON path, resource, or callable) as a document."""
    if callable(value) and not isinstance(value, Mapping):
        value = value()
    if isinstance(value, Mapping):
        return copy.deepcopy(dict(value))
    if isinstance(value, (str, os.PathLike)):
        path = Path(value)
        if not path.is_absolute():
            if origin is None:
                raise ValueError(f"relative vocabulary path {str(value)!r} has no origin")
            path = origin / path
        text = path.read_text(encoding="utf-8")
    elif hasattr(value, "read_text"):  # importlib.resources Traversable
        text = value.read_text(encoding="utf-8")
    else:
        raise TypeError(
            "a vocabulary must be a mapping, a JSON file path or a zero-argument "
            f"callable returning one, not {type(value).__name__}"
        )
    document = json.loads(text)
    if not isinstance(document, dict):
        raise ValueError("the JSON file does not hold an object")
    return document


def _register(
    registry: dict, name: str, value: Any, provider: Mapping[str, Any], origin: Optional[Path]
) -> Vocabulary:
    """The one registration path, for shipped and installed vocabularies alike."""
    document = _document(value, origin)
    problems = vocabulary_problems(document)
    if problems:
        raise VocabularyError(problems, subject="vocabulary-1 document")
    if document["id"] != name:
        raise ValueError(
            f"the document's id is {document['id']!r} but it is registered as "
            f"{name!r}; name the entry point after the vocabulary id"
        )
    if name in registry:
        raise ValueError(f"vocabulary {name!r} is already registered")
    vocabulary = Vocabulary(
        id=document["id"], version=document["version"], document=document, provider=dict(provider)
    )
    registry[name] = vocabulary
    return vocabulary


def _module_dir(target: str) -> Optional[Path]:
    """The directory of the module an entry point names (it was just imported)."""
    module = sys.modules.get(target.partition(":")[0].strip())
    location = getattr(module, "__file__", None)
    return Path(location).resolve().parent if location else None


@lru_cache(maxsize=1)
def _registry() -> VocabularyRegistry:
    registry: dict = {}
    builtin = {"kind": "builtin", "distribution": _DISTRIBUTION, "version": _package_version()}
    for source in _builtin_sources():
        # A shipped document that breaks a rule is a packaging bug: raise.
        _register(registry, source.name[: -len(".json")], source, builtin, None)
    builtin_ids = set(registry)

    result = load_extensions(VOCABULARIES_GROUP, strict=False)
    problems = list(result.problems)
    for item in result.loaded:
        ext = item.extension
        who = ext.distribution or "an unknown package"
        if ext.name in builtin_ids:
            problems.append(
                ExtensionProblem(
                    VOCABULARIES_GROUP,
                    ext.name,
                    SHADOWS_BUILTIN,
                    f"{who} provides vocabulary {ext.name!r}, which {_DISTRIBUTION} "
                    "ships; a shipped vocabulary cannot be redefined, so it was not "
                    "loaded. Give it an id of your own (e.g. 'mylab.depth').",
                    (ext.distribution,),
                )
            )
            continue
        provider = {
            "kind": "entry_point",
            "name": ext.name,
            "target": ext.target,
            "distribution": ext.distribution,
            "version": ext.version,
        }
        try:
            _register(registry, ext.name, item.value, provider, _module_dir(ext.target))
        except Exception as exc:  # noqa: BLE001 - reported, never swallowed
            problems.append(
                ExtensionProblem(
                    VOCABULARIES_GROUP,
                    ext.name,
                    "invalid",
                    f"{who} provides an invalid vocabulary {ext.name!r}: {exc}",
                    (ext.distribution,),
                )
            )
    if strict_mode() and problems:
        raise ExtensionError("; ".join(problem.message for problem in problems))
    return VocabularyRegistry(dict(sorted(registry.items())), tuple(problems))


def vocabulary_registry() -> VocabularyRegistry:
    """The registry of this environment (built once per process)."""
    return _registry()


def reload_vocabularies() -> VocabularyRegistry:
    """Forget the registry, so the next lookup re-reads installed vocabularies."""
    _registry.cache_clear()
    return _registry()


# -- lookup ------------------------------------------------------------------------------


def available_vocabularies() -> list[str]:
    """Ids of every registered vocabulary: shipped and installed."""
    return vocabulary_registry().ids()


def builtin_vocabularies() -> list[str]:
    """Ids of the vocabularies shipped with this package."""
    return [vid for vid, vocab in vocabulary_registry().vocabularies.items() if vocab.builtin]


def load_vocabulary(vocabulary_id: str) -> Vocabulary:
    """The registered vocabulary *vocabulary_id* (``euler.depth``, ``mylab.thermal``)."""
    registry = vocabulary_registry()
    vocabulary = registry.vocabularies.get(vocabulary_id)
    if vocabulary is not None:
        return vocabulary
    problem = registry.problem(vocabulary_id)
    if problem is not None:
        raise UnknownVocabularyError(
            f"vocabulary {vocabulary_id!r} was not registered: {problem.message}"
        )
    raise UnknownVocabularyError(
        f"unknown vocabulary {vocabulary_id!r}; registered: "
        f"{', '.join(registry.ids()) or 'none'}. Install the package that provides "
        f"it (entry-point group {VOCABULARIES_GROUP!r})."
    )


def vocabulary_documents(*, builtin_only: bool = False) -> dict:
    """Registered vocabulary documents by id (deep copies).

    ``builtin_only=True`` gives the shipped ones, independent of what else is
    installed (the TS artifact is generated from these).
    """
    registry = vocabulary_registry()
    return {
        vid: copy.deepcopy(vocab.document)
        for vid, vocab in registry.vocabularies.items()
        if vocab.builtin or not builtin_only
    }


class VocabularySet:
    """Look up terms across several vocabularies.

    Items are registered ids, :class:`Vocabulary` objects or plain
    ``vocabulary-1`` documents (validated). A term found in more than one
    vocabulary is ambiguous and raises; pass the term explicitly instead.
    """

    def __init__(self, vocabularies: Iterable[Any]) -> None:
        loaded = []
        for item in vocabularies:
            if isinstance(item, Vocabulary):
                loaded.append(item)
            elif isinstance(item, Mapping):
                document = copy.deepcopy(dict(item))
                problems = vocabulary_problems(document)
                if problems:
                    raise VocabularyError(problems, subject="vocabulary-1 document")
                loaded.append(
                    Vocabulary(id=document["id"], version=document["version"], document=document)
                )
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
