"""Comparison and filtering utilities for metric dictionaries."""
from __future__ import annotations

from fnmatch import fnmatch
from typing import Any

from .axes import AxisDeclaration, decompose


def compare_stages(
    metrics: dict[str, Any],
    ns: Any,
    modality: str,
    metric_name: str,
) -> dict[str, Any]:
    """Collect values of a base metric across all pipeline stages.

    Parameters
    ----------
    metrics:
        Flat dict of ``{fully_qualified_name: value}``.
    ns:
        A :class:`~euler_metric_naming.MetricNamespace` instance.
    modality:
        Modality to query (e.g. ``"depth"``).
    metric_name:
        Base metric name after axis stripping (e.g. ``"depth_mae"``).

    Returns
    -------
    dict[str, Any]
        Mapping of ``{stage_value: metric_value}``.  Metrics without a
        stage are keyed as ``None``.
    """
    namespace = f"{modality}.{ns.context}"
    axes = ns.axes(modality)
    prefix = namespace + "."

    result: dict[str | None, Any] = {}
    for key, value in metrics.items():
        if not key.startswith(prefix):
            continue
        try:
            dec = decompose(key, namespace, axes)
        except ValueError:
            continue
        if dec.metric == metric_name:
            stage = dec.axes.get("stage")
            result[stage] = value
    return result


def filter_kind(
    metrics: dict[str, Any],
    ns: Any,
    modality: str,
    kind: str,
) -> dict[str, Any]:
    """Return all metrics of a given kind for a modality.

    Parameters
    ----------
    metrics:
        Flat dict of ``{fully_qualified_name: value}``.
    ns:
        A :class:`~euler_metric_naming.MetricNamespace` instance.
    modality:
        Modality to query (e.g. ``"depth"``).
    kind:
        Kind to filter by (e.g. ``"loss"``, ``"diag"``, ``"stat"``).

    Returns
    -------
    dict[str, Any]
        Subset of *metrics* where the ``kind`` axis matches.
    """
    namespace = f"{modality}.{ns.context}"
    axes = ns.axes(modality)
    prefix = namespace + "."

    result: dict[str, Any] = {}
    for key, value in metrics.items():
        if not key.startswith(prefix):
            continue
        try:
            dec = decompose(key, namespace, axes)
        except ValueError:
            continue
        if dec.axes.get("kind") == kind:
            result[key] = value
    return result


def filter_glob(
    metrics: dict[str, Any],
    pattern: str,
) -> dict[str, Any]:
    """Return metrics whose fully-qualified names match a glob pattern.

    Uses ``fnmatch`` rules: ``*`` matches within a segment, ``?``
    matches a single character.  To match across dot-separated segments,
    use multiple wildcards (e.g. ``"depth.train.*.prior.*"``).

    Parameters
    ----------
    metrics:
        Flat dict of ``{fully_qualified_name: value}``.
    pattern:
        Glob pattern to match against metric names.

    Returns
    -------
    dict[str, Any]
        Subset of *metrics* matching the pattern.
    """
    return {
        key: value
        for key, value in metrics.items()
        if fnmatch(key, pattern)
    }
