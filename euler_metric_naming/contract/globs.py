"""Globs of ``docs/contract.md`` §4.2.

``*`` matches any run of characters, dots included; every other character is
literal. Matching is case-sensitive and anchored at both ends. This is not
:mod:`fnmatch`: ``?`` and ``[`` are literal.
"""
from __future__ import annotations

from typing import Iterable


def glob_match(pattern: str, name: str) -> bool:
    """Return whether *name* matches the glob *pattern*."""
    parts = pattern.split("*")
    if len(parts) == 1:
        return pattern == name
    first, last = parts[0], parts[-1]
    if len(name) < len(first) + len(last):
        return False
    if not name.startswith(first) or not name.endswith(last):
        return False
    position = len(first)
    end = len(name) - len(last)
    for middle in parts[1:-1]:
        if not middle:
            continue
        index = name.find(middle, position, end)
        if index < 0:
            return False
        position = index + len(middle)
    return True


def glob_any(patterns: Iterable[str], name: str) -> bool:
    """Return whether *name* matches any of *patterns* (strings only)."""
    return any(isinstance(p, str) and glob_match(p, name) for p in patterns)
