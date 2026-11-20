"""The evaluation protocol contract, schema 2 (``docs/contract.md``).

Producers build the ``protocol`` block of a results file with
:class:`ProtocolBuilder`; consumers normalize any file with :func:`normalize`
(or :func:`load_eval_file`) and judge one metric cell of two evaluations with
:func:`compare` / :func:`compare_files`. Everything is dependency-free; the
same contract ships as a TS artifact in ``contract/ts/``.
"""
from __future__ import annotations

from .builder import ProtocolBuilder
from .closed import SCHEMA_VERSION, closed_sets
from .conditions import evaluate as evaluate_condition
from .descriptors import effective_compare
from .evalfile import (
    EvalFile,
    file_evidence,
    flatten_metrics,
    flatten_per_file,
    load_eval_file,
)
from .globs import glob_match
from .hashing import (
    definition_key,
    definition_key_object,
    hash_value,
    manifest_hash,
    manifest_ids,
)
from .jcs import CanonicalizationError, canonicalize
from .legacy import legacy_table, normalize_legacy
from .measures import format_factor, format_percent, format_sig3, measure, phrase
from .normalize import is_supported, normalize, schema_major
from .strata import identity_projection, sqrt_equal_width_edges, sqrt_stratification
from .validate import ProtocolValidationError, check_protocol, protocol_problems
from .verdict import (
    Reason,
    Verdict,
    compare,
    compare_files,
    per_item_name,
    recompute,
    recompute_cell,
)
from .vocabulary import (
    Vocabulary,
    VocabularySet,
    available_vocabularies,
    load_vocabulary,
    vocabulary_documents,
)

__all__ = [
    "SCHEMA_VERSION",
    "CanonicalizationError",
    "EvalFile",
    "ProtocolBuilder",
    "ProtocolValidationError",
    "Reason",
    "Verdict",
    "Vocabulary",
    "VocabularySet",
    "available_vocabularies",
    "canonicalize",
    "check_protocol",
    "closed_sets",
    "compare",
    "compare_files",
    "definition_key",
    "definition_key_object",
    "effective_compare",
    "evaluate_condition",
    "file_evidence",
    "flatten_metrics",
    "flatten_per_file",
    "format_factor",
    "format_percent",
    "format_sig3",
    "glob_match",
    "hash_value",
    "identity_projection",
    "is_supported",
    "legacy_table",
    "load_eval_file",
    "load_vocabulary",
    "manifest_hash",
    "manifest_ids",
    "measure",
    "normalize",
    "normalize_legacy",
    "per_item_name",
    "phrase",
    "protocol_problems",
    "recompute",
    "recompute_cell",
    "schema_major",
    "sqrt_equal_width_edges",
    "sqrt_stratification",
    "vocabulary_documents",
]
