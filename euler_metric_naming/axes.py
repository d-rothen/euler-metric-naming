"""Axis declarations and metric name decomposition / recomposition."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

_SEGMENT_RE = re.compile(r"^[a-z0-9_]+$")
_FIRST_SEGMENT_RE = re.compile(r"^[a-z0-9]+$")


@dataclass(frozen=True)
class AxisDeclaration:
    """One axis in a metric namespace (e.g. ``kind``, ``stage``)."""

    position: int
    values: tuple[str, ...]
    optional: bool = False
    description: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "position": self.position,
            "optional": self.optional,
            "values": list(self.values),
        }
        if self.description is not None:
            d["description"] = self.description
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AxisDeclaration:
        return cls(
            position=data["position"],
            values=tuple(data["values"]),
            optional=data.get("optional", False),
            description=data.get("description"),
        )


@dataclass(frozen=True)
class DecomposedMetric:
    """Result of decomposing a fully-qualified metric name."""

    namespace: str
    axes: dict[str, str | None]
    metric: str

    def recompose(self) -> str:
        parts = [self.namespace]
        for _axis_name, value in sorted(
            self.axes.items(),
            key=lambda kv: kv[0],
        ):
            if value is not None:
                parts.append(value)
        parts.append(self.metric)
        return ".".join(parts)


def decompose(
    metric_name: str,
    namespace: str,
    axes: dict[str, AxisDeclaration],
) -> DecomposedMetric:
    """Decompose a fully-qualified metric name into namespace, axis values, and base metric.

    Implements the algorithm from the metric naming spec section 4.2.

    Parameters
    ----------
    metric_name:
        Fully-qualified metric name (e.g. ``"depth.train.loss.prior.log_radius"``).
    namespace:
        The metric namespace (e.g. ``"depth.train"``).
    axes:
        Axis declarations for this namespace, keyed by axis name.

    Returns
    -------
    DecomposedMetric
        Decomposed result with namespace, axis values, and base metric name.

    Raises
    ------
    ValueError
        If the metric name does not belong to the namespace or if a required
        axis is missing.
    """
    prefix = namespace + "."
    if not metric_name.startswith(prefix):
        raise ValueError(
            f"metric {metric_name!r} does not belong to namespace {namespace!r}"
        )

    remainder = metric_name[len(prefix):]
    segments = remainder.split(".")
    if not segments or segments == [""]:
        raise ValueError(f"metric {metric_name!r} has no segments after namespace")

    sorted_axes = sorted(axes.items(), key=lambda kv: kv[1].position)
    axis_values: dict[str, str | None] = {}
    idx = 0

    for axis_name, axis_decl in sorted_axes:
        if idx < len(segments):
            candidate = segments[idx]
            remaining_after_consume = len(segments) - idx - 1
            if candidate in axis_decl.values and remaining_after_consume >= 1:
                axis_values[axis_name] = candidate
                idx += 1
                continue

        if axis_decl.optional:
            axis_values[axis_name] = None
        else:
            raise ValueError(
                f"required axis {axis_name!r} missing in metric {metric_name!r}: "
                f"segment {segments[idx]!r} not in {list(axis_decl.values)}"
            )

    remaining = segments[idx:]
    if not remaining:
        raise ValueError(
            f"metric {metric_name!r} has no base metric name after axis consumption"
        )

    return DecomposedMetric(
        namespace=namespace,
        axes=axis_values,
        metric=".".join(remaining),
    )


def recompose(
    namespace: str,
    axes: dict[str, AxisDeclaration],
    axis_values: dict[str, str | None],
    metric: str,
) -> str:
    """Recompose a fully-qualified metric name from its parts.

    Parameters
    ----------
    namespace:
        The metric namespace (e.g. ``"depth.train"``).
    axes:
        Axis declarations for this namespace.
    axis_values:
        Values for each axis (``None`` for skipped optional axes).
    metric:
        Base metric name.

    Returns
    -------
    str
        Fully-qualified metric name.
    """
    parts = [namespace]
    for axis_name, axis_decl in sorted(axes.items(), key=lambda kv: kv[1].position):
        value = axis_values.get(axis_name)
        if value is not None:
            if value not in axis_decl.values:
                raise ValueError(
                    f"axis {axis_name!r} value {value!r} not in {list(axis_decl.values)}"
                )
            parts.append(value)
        elif not axis_decl.optional:
            raise ValueError(f"required axis {axis_name!r} has no value")
    parts.append(metric)
    return ".".join(parts)


def validate_metric_name(name: str) -> None:
    """Validate that a metric name follows the naming convention.

    Raises
    ------
    ValueError
        If any segment violates the charset rules.
    """
    segments = name.split(".")
    if len(segments) < 2:
        raise ValueError(
            f"metric name {name!r} must have at least 2 dot-separated segments"
        )
    if not _FIRST_SEGMENT_RE.match(segments[0]):
        raise ValueError(
            f"first segment {segments[0]!r} must match [a-z0-9]+"
        )
    for seg in segments[1:]:
        if not _SEGMENT_RE.match(seg):
            raise ValueError(
                f"segment {seg!r} must match [a-z0-9_]+"
            )
