"""Build a v2 ``protocol`` block (``docs/contract.md`` §4) the way producers should.

The builder snapshots the descriptors, metric definitions and reducers of every
vocabulary term it uses, computes the population manifest, and validates the
closed set before returning the JSON-ready block::

    builder = ProtocolBuilder("depth.dense", ["euler.core", "euler.sky", "euler.depth"])
    builder.reference("gt", identity=gt_identity, view={...})
    builder.param("depth.cap.max_m", 38)
    builder.observe("depth.cap.clipped", 0.012)
    builder.population(item_ids, unit="image")
    builder.stratification("depth.gt_range", sqrt_stratification(0.01, 32))
    builder.axis("space", "space").axis("bin", "stratum", stratification="depth.gt_range")
    builder.reducer("image_mean")
    builder.metrics_for(metric_names, "depth.eval", axes)
    protocol = builder.build()

Unknown terms raise unless their descriptor / definition is passed explicitly.
"""
from __future__ import annotations

import copy
import re
import unicodedata
from typing import Any, Iterable, Mapping, Optional, Sequence

from ..axes import AxisDeclaration, decompose
from .closed import AXIS_ROLES, REFERENCE_IDENTITY_SCHEME, SCHEMA_VERSION
from .hashing import manifest_hash
from .validate import (
    ProtocolValidationError,
    descriptor_problems,
    metric_problems,
    parameter_value_problem,
    protocol_problems,
    reducer_problems,
    stratification_problems,
)
from .vocabulary import VocabularySet

_SCHEME_RE = re.compile(r"^[^/\s]+/[0-9]+$")


class ProtocolBuilder:
    """Accumulates one evaluation's protocol and returns it with :meth:`build`.

    Parameters
    ----------
    task:
        Task id written to ``protocol.task`` (e.g. ``"depth.dense"``).
    vocabularies:
        Vocabulary ids (or loaded documents) whose terms may be used by key.
    schema_version:
        The ``schemaVersion`` to write; defaults to the package's.
    """

    def __init__(
        self,
        task: Optional[str] = None,
        vocabularies: Iterable[Any] = ("euler.core",),
        *,
        schema_version: str = SCHEMA_VERSION,
    ) -> None:
        self._vocab = VocabularySet(vocabularies)
        self._task = task
        self._schema_version = schema_version
        self._used_vocabularies: dict = {}
        self._parameters: dict = {}
        self._observations: dict = {}
        self._descriptors: dict = {}
        self._references: dict = {}
        self._population: Optional[dict] = None
        self._item_ids: Optional[list] = None
        self._stratifications: dict = {}
        self._reducers: dict = {}
        self._axes: dict = {}
        self._metrics: dict = {}
        self._series: dict = {}

    # -- terms -----------------------------------------------------------------

    def _use(self, snapshot: dict) -> dict:
        tag = snapshot.get("vocabulary")
        if isinstance(tag, str) and "@" in tag:
            vid, version = tag.rsplit("@", 1)
            self._used_vocabularies[vid] = version
        return snapshot

    def _describe(self, key: str, descriptor: Optional[Mapping[str, Any]]) -> None:
        if descriptor is not None:
            snapshot = copy.deepcopy(dict(descriptor))
            if isinstance(snapshot.get("vocabulary"), str):
                self._use(snapshot)
        else:
            snapshot = self._vocab.descriptor(key)
            if snapshot is None:
                raise KeyError(
                    f"{key!r} is not a term of {self._vocabulary_tags()}; "
                    "pass descriptor=... to describe it"
                )
            self._use(snapshot)
        problems = descriptor_problems(snapshot, f"descriptors.{key}")
        if problems:
            raise ProtocolValidationError(problems)
        self._descriptors[key] = snapshot

    def _vocabulary_tags(self) -> str:
        tags = [vocabulary.tag for vocabulary in self._vocab.vocabularies]
        return ", ".join(tags) or "no vocabulary"

    def param(
        self, key: str, value: Any, *, descriptor: Optional[Mapping[str, Any]] = None
    ) -> ProtocolBuilder:
        """Record a parameter (a scalar or a list of at most 64 scalars)."""
        if isinstance(value, tuple):
            value = list(value)
        problem = parameter_value_problem(value)
        if problem:
            raise ValueError(f"parameter {key!r} {problem}")
        if key in self._observations:
            raise ValueError(f"{key!r} is already an observation")
        self._describe(key, descriptor)
        self._parameters[key] = copy.deepcopy(value)
        return self

    def describe(
        self, key: str, *, descriptor: Optional[Mapping[str, Any]] = None
    ) -> ProtocolBuilder:
        """Snapshot the descriptor of *key* without recording a value.

        For keys whose values are recorded elsewhere, such as the parameters a
        validation pass resolves and the observations it makes
        (``docs/training.md`` §3.2). A key already described keeps its
        descriptor.
        """
        if key not in self._descriptors:
            self._describe(key, descriptor)
        return self

    def params(self, values: Mapping[str, Any]) -> ProtocolBuilder:
        """Record several vocabulary parameters at once."""
        for key, value in values.items():
            self.param(key, value)
        return self

    def observe(
        self, key: str, value: Any, *, descriptor: Optional[Mapping[str, Any]] = None
    ) -> ProtocolBuilder:
        """Record an observation (shape given by the descriptor's ``observe``)."""
        if key in self._parameters:
            raise ValueError(f"{key!r} is already a parameter")
        self._describe(key, descriptor)
        self._observations[key] = copy.deepcopy(value)
        return self

    def reference(
        self,
        role: str,
        *,
        identity: str,
        modality: Optional[str] = None,
        dataset: Optional[Mapping[str, Any]] = None,
        manifest: Optional[Mapping[str, Any]] = None,
        view: Optional[Mapping[str, Any]] = None,
        descriptor: Optional[Mapping[str, Any]] = None,
        identity_scheme: Optional[str] = REFERENCE_IDENTITY_SCHEME,
    ) -> ProtocolBuilder:
        """Record a reference input; its descriptor key is ``reference.<role>``.

        ``identity_scheme`` (``<family>/<integer>``) names how *identity* was
        computed; euler's v2 identities are ``euler.reference/2`` (default).
        ``None`` leaves it unspecified.
        """
        if not isinstance(identity, str) or not identity:
            raise ValueError("a reference needs a non-empty identity")
        if identity_scheme is not None and not _SCHEME_RE.match(identity_scheme):
            raise ValueError("identity_scheme must have the form <family>/<integer>")
        entry: dict = {}
        if modality is not None:
            entry["modality"] = modality
        if dataset is not None:
            entry["dataset"] = copy.deepcopy(dict(dataset))
        entry["identity"] = identity
        if identity_scheme is not None:
            entry["identityScheme"] = identity_scheme
        if manifest is not None:
            entry["manifest"] = copy.deepcopy(dict(manifest))
        if view is not None:
            entry["view"] = copy.deepcopy(dict(view))
        self._describe(f"reference.{role}", descriptor)
        self._references[role] = entry
        return self

    def population(
        self,
        ids: Sequence[str],
        *,
        unit: Optional[str] = None,
        accounting: Optional[Mapping[str, Any]] = None,
        id_scheme: str = "euler.item_id/1",
        items: bool = True,
    ) -> ProtocolBuilder:
        """Record the evaluated items; computes ``count`` and ``manifest``.

        With ``items=True`` (default) the ids also become ``items.ids``, the
        alignment of every :meth:`series`.
        """
        normalized = [unicodedata.normalize("NFC", item) for item in ids]
        if len(set(normalized)) != len(normalized):
            raise ValueError("item ids must be unique (after NFC normalization)")
        population: dict = {}
        if unit is not None:
            population["unit"] = unit
        population["idScheme"] = id_scheme
        population["count"] = len(normalized)
        population["manifest"] = manifest_hash(normalized)
        if accounting is not None:
            population["accounting"] = copy.deepcopy(dict(accounting))
        self._population = population
        self._item_ids = normalized if items else None
        return self

    def stratification(self, sid: str, stratification: Mapping[str, Any]) -> ProtocolBuilder:
        """Record a stratification (see :func:`sqrt_stratification`)."""
        snapshot = copy.deepcopy(dict(stratification))
        problems = stratification_problems(snapshot, f"stratifications.{sid}")
        scheme = snapshot.get("scheme")
        if isinstance(scheme, str) and self._vocab.stratification_scheme(scheme) is None:
            problems.append(f"stratifications.{sid}.scheme {scheme!r} is not a vocabulary scheme")
        if problems:
            raise ProtocolValidationError(problems)
        self._stratifications[sid] = snapshot
        return self

    def axis(
        self, name: str, role: str, *, stratification: Optional[str] = None
    ) -> ProtocolBuilder:
        """Give the metricSet axis *name* a role (§3.9)."""
        if role not in AXIS_ROLES:
            raise ValueError(f"axis role must be one of {', '.join(AXIS_ROLES)}")
        entry: dict = {"role": role}
        if role == "stratum":
            if not stratification:
                raise ValueError("a stratum axis needs its stratification id")
            entry["stratification"] = stratification
        elif stratification is not None:
            raise ValueError("only stratum axes name a stratification")
        self._axes[name] = entry
        return self

    def reducer(
        self, value: str, reducer: Optional[Mapping[str, Any]] = None
    ) -> ProtocolBuilder:
        """Record the reducer of reduction axis value *value*."""
        if reducer is not None:
            snapshot = copy.deepcopy(dict(reducer))
        else:
            snapshot = self._vocab.reducer(value)
            if snapshot is None:
                raise KeyError(
                    f"reducer {value!r} is not a term of {self._vocabulary_tags()}; "
                    "pass its definition"
                )
            self._use(snapshot)
        problems = reducer_problems(snapshot, f"reducers.{value}")
        if problems:
            raise ProtocolValidationError(problems)
        self._reducers[value] = snapshot
        return self

    def metric(
        self, base: str, definition: Optional[Mapping[str, Any]] = None
    ) -> ProtocolBuilder:
        """Record the definition of base metric *base* (vocabulary or explicit)."""
        if definition is not None:
            snapshot = copy.deepcopy(dict(definition))
            if isinstance(snapshot.get("vocabulary"), str):
                self._use(snapshot)
        else:
            snapshot = self._vocab.metric(base)
            if snapshot is None:
                raise KeyError(
                    f"metric {base!r} is not a term of {self._vocabulary_tags()}; "
                    "pass its definition"
                )
            self._use(snapshot)
        problems = metric_problems(snapshot, f"metrics.{base}")
        if problems:
            raise ProtocolValidationError(problems)
        self._metrics[base] = snapshot
        reducer = snapshot.get("reducer")
        if isinstance(reducer, str) and reducer not in self._reducers:
            term = self._vocab.reducer(reducer) or VocabularySet(["euler.core"]).reducer(reducer)
            if term is not None:  # else build() reports the dangling name
                self._use(term)
                self._reducers[reducer] = term
        return self

    def metrics_for(
        self,
        metric_names: Iterable[str],
        namespace: str,
        axes: Mapping[str, Any],
        *,
        missing: str = "error",
    ) -> ProtocolBuilder:
        """Snapshot the definition of every base metric of *metric_names*.

        *axes* are the metricSet axis declarations (``AxisDeclaration`` or
        their dicts). Reducers of the reduction axis values found are recorded
        too. ``missing="skip"`` leaves bases without a vocabulary term out
        instead of raising.
        """
        if missing not in ("error", "skip"):
            raise ValueError("missing must be 'error' or 'skip'")
        declarations = _declarations(axes)
        reduction_axes = [name for name, entry in self._axes.items() if entry["role"] == "reduction"]
        for name in metric_names:
            parsed = decompose(name, namespace, declarations)
            if parsed.metric not in self._metrics:
                snapshot = self._vocab.metric(parsed.metric)
                if snapshot is None:
                    if missing == "error":
                        raise KeyError(
                            f"metric {parsed.metric!r} (of {name!r}) is not a term of "
                            f"{self._vocabulary_tags()}"
                        )
                else:
                    self.metric(parsed.metric)
            for axis_name in reduction_axes:
                value = parsed.axes.get(axis_name)
                if value is not None and value not in self._reducers:
                    if self._vocab.reducer(value) is not None:
                        self.reducer(value)
                    elif missing == "error":
                        raise KeyError(f"reducer {value!r} is not a vocabulary term")
        return self

    def series(self, name: str, values: Sequence[Any]) -> ProtocolBuilder:
        """Record a per-item statistic/observation series aligned with the ids."""
        if self._item_ids is None:
            raise ValueError("call population(ids) before series()")
        values = list(values)
        if len(values) != len(self._item_ids):
            raise ValueError(
                f"series {name!r} has {len(values)} values for {len(self._item_ids)} items"
            )
        self._series[name] = values
        return self

    # -- output ----------------------------------------------------------------

    def build(self) -> dict:
        """Return the JSON-ready ``protocol`` block; raises when it is invalid."""
        protocol: dict = {"schemaVersion": self._schema_version}
        if self._task is not None:
            protocol["task"] = self._task
        protocol["vocabularies"] = dict(sorted(self._used_vocabularies.items()))
        protocol["parameters"] = copy.deepcopy(self._parameters)
        protocol["descriptors"] = copy.deepcopy(self._descriptors)
        if self._references:
            protocol["references"] = copy.deepcopy(self._references)
        if self._observations:
            protocol["observations"] = copy.deepcopy(self._observations)
        if self._population is not None:
            protocol["population"] = copy.deepcopy(self._population)
        if self._stratifications:
            protocol["stratifications"] = copy.deepcopy(self._stratifications)
        if self._reducers:
            protocol["reducers"] = copy.deepcopy(self._reducers)
        if self._axes:
            protocol["axes"] = copy.deepcopy(self._axes)
        if self._metrics:
            protocol["metrics"] = copy.deepcopy(self._metrics)
        if self._item_ids is not None and (self._series or self._item_ids):
            protocol["items"] = {
                "ids": list(self._item_ids),
                "series": copy.deepcopy(self._series),
            }
        problems = protocol_problems(protocol)
        if problems:
            raise ProtocolValidationError(problems)
        return protocol


def _declarations(axes: Mapping[str, Any]) -> dict:
    declarations = {}
    for name, axis in axes.items():
        if isinstance(axis, AxisDeclaration):
            declarations[name] = axis
        else:
            declarations[name] = AxisDeclaration.from_dict(dict(axis))
    return declarations
