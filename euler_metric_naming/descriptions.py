"""Metric description metadata."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class MetricDescription:
    """Display metadata for a single base metric.

    Keys here are base metric names (after stripping namespace and axes),
    not fully-qualified names.  The same description applies to all axis
    combinations of that metric.
    """

    is_higher_better: bool | None = None
    min_value: float | None = None
    max_value: float | None = None
    scale: str | None = None
    unit: str | None = None
    format_hint: str | None = None
    display_name: str | None = None
    description: str | None = None

    _VALID_SCALES = frozenset({"linear", "log", "percentage", "binary"})

    def __post_init__(self) -> None:
        if self.scale is not None and self.scale not in self._VALID_SCALES:
            raise ValueError(
                f"scale must be one of {sorted(self._VALID_SCALES)}, got {self.scale!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the camelCase JSON format used in metricSet envelopes."""
        d: dict[str, Any] = {}
        if self.is_higher_better is not None:
            d["isHigherBetter"] = self.is_higher_better
        if self.min_value is not None:
            d["min"] = self.min_value
        if self.max_value is not None:
            d["max"] = self.max_value
        if self.scale is not None:
            d["scale"] = self.scale
        if self.unit is not None:
            d["unit"] = self.unit
        if self.format_hint is not None:
            d["formatHint"] = self.format_hint
        if self.display_name is not None:
            d["displayName"] = self.display_name
        if self.description is not None:
            d["description"] = self.description
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MetricDescription:
        """Deserialize from camelCase JSON format."""
        return cls(
            is_higher_better=data.get("isHigherBetter"),
            min_value=data.get("min"),
            max_value=data.get("max"),
            scale=data.get("scale"),
            unit=data.get("unit"),
            format_hint=data.get("formatHint"),
            display_name=data.get("displayName"),
            description=data.get("description"),
        )
