"""euler-metric-naming — structured metric naming for the Euler ML ecosystem."""
from __future__ import annotations

from .namespace import MetricNamespace
from .axes import AxisDeclaration, DecomposedMetric, decompose, recompose, validate_metric_name
from .descriptions import MetricDescription
from .matching import compare_stages, filter_kind, filter_glob

__all__ = [
    "AxisDeclaration",
    "DecomposedMetric",
    "MetricDescription",
    "MetricNamespace",
    "compare_stages",
    "decompose",
    "filter_glob",
    "filter_kind",
    "recompose",
    "validate_metric_name",
]
__version__ = "0.1.0"
