"""MetricNamespace — the primary API for structured metric naming."""
from __future__ import annotations

import warnings
from typing import Any

from .axes import AxisDeclaration, validate_metric_name
from .descriptions import MetricDescription
from ._compat import validate_modality

_RESERVED_SCOPES = frozenset({"sys"})
_TRAIN_KINDS = ("loss", "diag", "stat")


class MetricNamespace:
    """Structured metric name builder and validator.

    Parameters
    ----------
    producer:
        Producer identifier (e.g. ``"euler_train.weather_metric"``).
    producer_version:
        Semver version string of the producer.
    modalities:
        Tuple of modality names that this producer emits metrics for
        (e.g. ``("depth", "rgb", "rays")``).
    stages:
        Optional tuple of pipeline stage names.  When provided, the
        ``loss``, ``diag``, and ``stat`` helpers accept a ``stage``
        argument.  When omitted, stage-less metric names are produced.
    context:
        Either ``"train"`` or ``"eval"``.  Determines the second
        segment of the namespace (e.g. ``depth.train`` vs ``depth.eval``).
    descriptions:
        Optional mapping of base metric name to :class:`MetricDescription`.
    """

    def __init__(
        self,
        producer: str,
        producer_version: str,
        modalities: tuple[str, ...] | list[str],
        stages: tuple[str, ...] | list[str] | None = None,
        context: str = "train",
        descriptions: dict[str, MetricDescription] | None = None,
    ) -> None:
        if context not in ("train", "eval"):
            raise ValueError(f"context must be 'train' or 'eval', got {context!r}")

        self._producer = producer
        self._producer_version = producer_version
        self._modalities = tuple(modalities)
        self._stages = tuple(stages) if stages is not None else None
        self._context = context
        self._descriptions = dict(descriptions) if descriptions else {}
        self._sys_used = False

        for mod in self._modalities:
            if mod in _RESERVED_SCOPES:
                raise ValueError(
                    f"{mod!r} is a reserved scope prefix and cannot be used as a modality"
                )
            warning = validate_modality(mod)
            if warning:
                warnings.warn(warning, stacklevel=2)

        if self._stages is not None:
            for stage in self._stages:
                _validate_segment(stage, "stage")

        for mod in self._modalities:
            _validate_segment(mod, "modality")

    @property
    def producer(self) -> str:
        return self._producer

    @property
    def producer_version(self) -> str:
        return self._producer_version

    @property
    def context(self) -> str:
        return self._context

    @property
    def modalities(self) -> tuple[str, ...]:
        return self._modalities

    @property
    def stages(self) -> tuple[str, ...] | None:
        return self._stages

    @property
    def descriptions(self) -> dict[str, MetricDescription]:
        return dict(self._descriptions)

    # -- training convenience methods ------------------------------------------

    def loss(self, modality: str, *args: str, metric: str | None = None) -> str:
        """Build a loss metric key.

        Call signatures::

            ns.loss("depth", "prior", "log_radius")   # with stage
            ns.loss("depth", metric="total")           # without stage
        """
        return self._build_kind_key("loss", modality, args, metric)

    def diag(self, modality: str, *args: str, metric: str | None = None) -> str:
        """Build a diagnostic metric key.

        Call signatures::

            ns.diag("depth", "prior", "depth_mae")    # with stage
            ns.diag("depth", metric="rmse")            # without stage
        """
        return self._build_kind_key("diag", modality, args, metric)

    def stat(self, modality: str, *args: str, metric: str | None = None) -> str:
        """Build a stat metric key.

        Call signatures::

            ns.stat("depth", "structure", "confidence")  # with stage
            ns.stat("depth", metric="valid_fraction")     # without stage
        """
        return self._build_kind_key("stat", modality, args, metric)

    def sys(self, *parts: str) -> str:
        """Build a system metric key.

        Examples::

            ns.sys("lr")                    # → "sys.train.lr"
            ns.sys("lr", "geometry_encoder") # → "sys.train.lr.geometry_encoder"
        """
        if not parts:
            raise TypeError("sys() requires at least one metric segment")
        for part in parts:
            _validate_segment(part, "sys metric")
        self._sys_used = True
        return f"sys.{self._context}." + ".".join(parts)

    # -- axis declarations -----------------------------------------------------

    def axes(self, modality: str) -> dict[str, AxisDeclaration]:
        """Return axis declarations for a modality's namespace.

        For training context, the axes are ``kind`` (required) and
        optionally ``stage``.  For eval context, override this in a
        subclass or use :meth:`axes_for_eval`.
        """
        self._check_modality(modality)
        return self._build_axes()

    def _build_axes(self) -> dict[str, AxisDeclaration]:
        axes: dict[str, AxisDeclaration] = {
            "kind": AxisDeclaration(
                position=0,
                values=_TRAIN_KINDS,
                optional=False,
                description="Metric kind",
            ),
        }
        if self._stages is not None:
            axes["stage"] = AxisDeclaration(
                position=1,
                values=self._stages,
                optional=True,
                description="Pipeline refinement stage",
            )
        return axes

    # -- envelope generation ---------------------------------------------------

    def metric_set_envelope(
        self,
        modality: str,
        *,
        source_kind: str = "computed",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate a metricSet envelope for evaluation metrics.

        Returns the envelope dict suitable for inclusion in ``eval.json``.
        """
        self._check_modality(modality)
        namespace = f"{modality}.{self._context}"

        envelope: dict[str, Any] = {
            "metricNamespace": namespace,
            "producerKey": self._producer,
            "producerVersion": self._producer_version,
            "sourceKind": source_kind,
        }

        if metadata:
            envelope["metadata"] = metadata
        else:
            envelope["metadata"] = {}

        axes = self._build_axes()
        if axes:
            envelope["axes"] = {
                name: decl.to_dict() for name, decl in axes.items()
            }

        if self._descriptions:
            envelope["metricDescriptions"] = {
                name: desc.to_dict()
                for name, desc in self._descriptions.items()
            }

        return envelope

    def training_naming_config(self) -> dict[str, Any]:
        """Generate the run-level ``metric_naming`` payload for euler-train.

        This is stored on the model run and used by euler-view for
        metric decomposition and grouping.
        """
        namespaces: dict[str, Any] = {}

        axes_dict = {
            name: decl.to_dict() for name, decl in self._build_axes().items()
        }
        desc_dict = (
            {name: desc.to_dict() for name, desc in self._descriptions.items()}
            if self._descriptions
            else None
        )

        for mod in self._modalities:
            ns_key = f"{mod}.{self._context}"
            entry: dict[str, Any] = {"axes": axes_dict}
            if desc_dict:
                entry["metricDescriptions"] = desc_dict
            namespaces[ns_key] = entry

        # Include sys.{context} when sys metrics have been used or always
        # for training context (spec: euler-metric-naming should include
        # sys.train automatically when system metrics are emitted).
        sys_key = f"sys.{self._context}"
        if self._sys_used or self._context == "train":
            namespaces[sys_key] = {"axes": {}}

        return {
            "producer_key": self._producer,
            "producer_version": self._producer_version,
            "namespaces": namespaces,
        }

    # -- internals -------------------------------------------------------------

    def _build_kind_key(
        self,
        kind: str,
        modality: str,
        args: tuple[str, ...],
        metric_kw: str | None,
    ) -> str:
        if self._context != "train":
            raise TypeError(
                f"loss/diag/stat helpers are only available in 'train' context, "
                f"not {self._context!r}"
            )

        self._check_modality(modality)

        if args and metric_kw is not None:
            raise TypeError(
                "pass (modality, stage, metric) positionally or "
                "(modality, metric=metric), not both"
            )

        if len(args) == 2:
            stage, metric_name = args[0], args[1]
        elif len(args) == 0:
            if metric_kw is None:
                raise TypeError("metric is required")
            stage = None
            metric_name = metric_kw
        else:
            raise TypeError(
                f"expected 0 or 2 extra positional args, got {len(args)}; "
                f"use ns.{kind}(modality, stage, metric) or "
                f"ns.{kind}(modality, metric=metric)"
            )

        if stage is not None:
            if self._stages is None:
                raise ValueError(
                    f"stages not declared for this namespace; "
                    f"cannot use stage={stage!r}"
                )
            if stage not in self._stages:
                raise ValueError(
                    f"unknown stage {stage!r}; "
                    f"declared stages: {list(self._stages)}"
                )
            _validate_segment(stage, "stage")

        _validate_segment(metric_name, "metric")

        namespace = f"{modality}.{self._context}"
        parts = [namespace, kind]
        if stage is not None:
            parts.append(stage)
        parts.append(metric_name)
        key = ".".join(parts)
        validate_metric_name(key)
        return key

    def _check_modality(self, modality: str) -> None:
        if modality not in self._modalities:
            raise ValueError(
                f"unknown modality {modality!r}; "
                f"declared modalities: {list(self._modalities)}"
            )


def _validate_segment(value: str, label: str) -> None:
    """Validate a single metric name segment."""
    import re

    if not re.match(r"^[a-z0-9_]+$", value):
        raise ValueError(
            f"{label} {value!r} must match [a-z0-9_]+"
        )
