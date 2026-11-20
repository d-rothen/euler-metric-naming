"""The three-valued condition language of ``docs/contract.md`` §7.

:func:`evaluate` returns ``True``, ``False`` or ``None`` (unknown).
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

from ._json import is_number, json_equal
from .closed import CONDITION_OPS

_FORMS = ("all", "any", "not", "param", "observation")


def evaluate(
    condition: Any,
    parameters: Optional[Mapping[str, Any]] = None,
    observations: Optional[Mapping[str, Any]] = None,
) -> Optional[bool]:
    """Evaluate *condition* against one side's parameters and observations.

    Returns ``True``, ``False`` or ``None`` for unknown. A malformed condition
    (unknown operator, wrong shape, more than one form) is unknown.
    """
    params = parameters if isinstance(parameters, Mapping) else {}
    obs = observations if isinstance(observations, Mapping) else {}
    return _evaluate(condition, params, obs)


def _evaluate(condition: Any, params: Mapping, obs: Mapping) -> Optional[bool]:
    if not isinstance(condition, dict):
        return None
    forms = [form for form in _FORMS if form in condition]
    if len(forms) != 1:
        return None
    form = forms[0]
    if form in ("all", "any"):
        members = condition[form]
        if not isinstance(members, list):
            return None
        results = [_evaluate(member, params, obs) for member in members]
        if form == "all":
            if any(result is False for result in results):
                return False
            if any(result is None for result in results):
                return None
            return True
        if any(result is True for result in results):
            return True
        if any(result is None for result in results):
            return None
        return False
    if form == "not":
        result = _evaluate(condition["not"], params, obs)
        return None if result is None else not result
    key = condition[form]
    op = condition.get("op")
    if not isinstance(key, str) or op not in CONDITION_OPS:
        return None
    source = params if form == "param" else obs
    present = key in source and source[key] is not None
    if op == "exists":
        return present
    if not present or "value" not in condition:
        return None
    operand = source[key]
    value = condition["value"]
    if op == "eq":
        return json_equal(operand, value)
    if op == "ne":
        return not json_equal(operand, value)
    if op == "in":
        if not isinstance(value, list):
            return None
        return any(json_equal(operand, item) for item in value)
    if not (is_number(operand) and is_number(value)):
        return None
    a, b = float(operand), float(value)
    if op == "lt":
        return a < b
    if op == "le":
        return a <= b
    if op == "gt":
        return a > b
    return a >= b


def is_valid_condition(condition: Any) -> bool:
    """Structural check used by producers: well-formed per §7."""
    return not condition_problems(condition)


def condition_problems(condition: Any, path: str = "condition") -> list[str]:
    """Return the structural problems of a condition (empty when well-formed)."""
    if not isinstance(condition, dict):
        return [f"{path} must be an object"]
    forms = [form for form in _FORMS if form in condition]
    if len(forms) != 1:
        return [f"{path} must have exactly one of {', '.join(_FORMS)}"]
    form = forms[0]
    if form in ("all", "any"):
        members = condition[form]
        if not isinstance(members, list):
            return [f"{path}.{form} must be a list"]
        problems: list[str] = []
        for index, member in enumerate(members):
            problems.extend(condition_problems(member, f"{path}.{form}[{index}]"))
        return problems
    if form == "not":
        return condition_problems(condition["not"], f"{path}.not")
    problems = []
    if not isinstance(condition[form], str) or not condition[form]:
        problems.append(f"{path}.{form} must be a non-empty string")
    op = condition.get("op")
    if op not in CONDITION_OPS:
        problems.append(f"{path}.op must be one of {', '.join(CONDITION_OPS)}")
    elif op != "exists" and "value" not in condition:
        problems.append(f"{path}.value is required for op {op!r}")
    elif op == "in" and not isinstance(condition.get("value"), list):
        problems.append(f"{path}.value must be a list for op 'in'")
    return problems
