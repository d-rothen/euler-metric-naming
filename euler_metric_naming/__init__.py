"""euler-metric-naming — structured metric naming for the Euler ML ecosystem."""
from __future__ import annotations

from .axes import (
    AxisDeclaration,
    DecomposedMetric,
    decompose,
    recompose,
    validate_metric_name,
)
from .descriptions import MetricDescription
from .matching import compare_stages, filter_glob, filter_kind
from .namespace import MetricNamespace

__version__ = "0.2.0"

__all__ = [
    "AxisDeclaration",
    "DecomposedMetric",
    "MetricDescription",
    "MetricNamespace",
    "__version__",
    "compare_stages",
    "decompose",
    "filter_glob",
    "filter_kind",
    "recompose",
    "validate_metric_name",
]
