"""The closed set of the evaluation protocol contract (``docs/contract.md`` §3).

Everything a consumer must understand to judge an evaluation is listed here.
Changing any of these tuples is a major change of the protocol schema. The TS
artifact receives the same values through ``scripts/generate_ts.py``.
"""
from __future__ import annotations

#: The protocol schema version this package writes.
SCHEMA_VERSION = "2.0"

#: Majors a consumer of this package can judge: 2 natively, 1 through the
#: legacy adapter (§10).
SUPPORTED_MAJORS = (1, 2)

#: §3.1 descriptor roles and the compare rule each defaults to.
ROLE_DEFAULT_COMPARE = {
    "reference": "equal",
    "definition": "equal",
    "population": "pair",
    "domain": "measure",
    "derived": "ignore",
    "provenance": "ignore",
}
ROLES = tuple(ROLE_DEFAULT_COMPARE)

#: §3.2 compare rules, most strict first.
COMPARE_RULES = ("equal", "measure", "pair", "ignore")

#: §3.3 observation kinds.
OBSERVE_KINDS = ("region", "count", "fraction", "factor", "size")

#: §3.4 display scales.
DISPLAY_SCALES = ("relative", "complement", "absolute", "none")

#: §3.5 metric directions.
DIRECTIONS = ("lower", "higher", "none")

#: §3.6 metric kinds.
METRIC_KINDS = ("score", "statistic", "observation")

#: §3.7 reducer recompute operations.
RECOMPUTE_OPS = ("mean", "median", "pool", "none")

#: §3.8 pool forms.
POOL_FORMS = ("mean", "rms", "none")

#: §3.9 axis roles.
AXIS_ROLES = ("space", "reduction", "stratum", "variant", "family", "stage", "kind")

#: §3.10 condition combinators and comparison operators.
CONDITION_COMBINATORS = ("all", "any", "not")
CONDITION_OPS = ("exists", "eq", "ne", "lt", "le", "gt", "ge", "in")

#: §3.11 verdict tiers, least severe first.
TIERS = ("same", "paired", "comparable", "incomparable")

#: Reason codes of §8, in no particular order.
REASON_CODES = (
    "schema",
    "reference",
    "revision",
    "not_recorded",
    "stratification",
    "parameter",
    "domain",
    "population_disjoint",
    "population_paired",
    "population_partial",
    "population_unknown",
)

#: The item id scheme of §4.4.
ITEM_ID_SCHEME = "euler.item_id/1"

#: Separator of item id path segments (§4.4).
ITEM_ID_SEPARATOR = "~"

#: Fields left out of a stratification's identity projection (§4.6).
STRATIFICATION_DISPLAY_FIELDS = ("label", "description", "unit")

#: v1 compatibility limits on ``protocol.parameters`` (§2).
MAX_PARAMETERS = 512
MAX_KEY_LENGTH = 255
MAX_LIST_LENGTH = 64
MAX_STRING_LENGTH = 1024


def closed_sets() -> dict:
    """Return the closed set as one JSON-ready object (used by generators)."""
    return {
        "schemaVersion": SCHEMA_VERSION,
        "supportedMajors": list(SUPPORTED_MAJORS),
        "roles": list(ROLES),
        "roleDefaultCompare": dict(ROLE_DEFAULT_COMPARE),
        "compareRules": list(COMPARE_RULES),
        "observeKinds": list(OBSERVE_KINDS),
        "displayScales": list(DISPLAY_SCALES),
        "directions": list(DIRECTIONS),
        "metricKinds": list(METRIC_KINDS),
        "recomputeOps": list(RECOMPUTE_OPS),
        "poolForms": list(POOL_FORMS),
        "axisRoles": list(AXIS_ROLES),
        "conditionCombinators": list(CONDITION_COMBINATORS),
        "conditionOps": list(CONDITION_OPS),
        "tiers": list(TIERS),
        "reasonCodes": list(REASON_CODES),
        "itemIdScheme": ITEM_ID_SCHEME,
        "itemIdSeparator": ITEM_ID_SEPARATOR,
        "parameterLimits": {
            "maxParameters": MAX_PARAMETERS,
            "maxKeyLength": MAX_KEY_LENGTH,
            "maxListLength": MAX_LIST_LENGTH,
            "maxStringLength": MAX_STRING_LENGTH,
        },
    }
