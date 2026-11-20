"""The evaluation protocol contract, schema 2 (``docs/contract.md``).

Producers build the ``protocol`` block of a results file with
:class:`ProtocolBuilder`; consumers normalize any file with :func:`normalize`
(or :func:`load_eval_file`) and judge one metric cell of two evaluations with
:func:`compare` / :func:`compare_files`. Training runs (``docs/training.md``)
declare their telemetry with metric definitions and record validation sets,
whose passes :func:`load_validation_pass` turns into evaluations for the same
engine. Everything is dependency-free; the same contract ships as a TS
artifact in ``contract/ts/``.
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
from .training import (
    RESERVED_RECORD_FIELDS,
    TRAINING_SCHEMA_VERSION,
    VALIDATION_SET_KIND,
    VALIDATION_SET_SCHEMA_VERSION,
    Problem,
    evaluation_metric_ids,
    is_training_naming_v2,
    load_validation_pass,
    metric_value_problem,
    training_contract,
    training_metric_definition,
    training_metric_problem,
    training_naming_problems,
    validation_pass_contract,
    validation_pass_ids,
    validation_pass_problems,
    validation_set_problems,
)
from .validate import ProtocolValidationError, check_protocol, protocol_problems
from .verdict import (
    Reason,
    Verdict,
    compare,
    compare_files,
    per_item_name,
    recompute,
    recompute_cell,
    relevant_keys,
)
from .vocabulary import (
    Vocabulary,
    VocabularySet,
    available_vocabularies,
    load_vocabulary,
    vocabulary_documents,
)

__all__ = [
    "RESERVED_RECORD_FIELDS",
    "SCHEMA_VERSION",
    "TRAINING_SCHEMA_VERSION",
    "VALIDATION_SET_KIND",
    "VALIDATION_SET_SCHEMA_VERSION",
    "CanonicalizationError",
    "EvalFile",
    "Problem",
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
    "evaluation_metric_ids",
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
    "is_training_naming_v2",
    "legacy_table",
    "load_eval_file",
    "load_validation_pass",
    "load_vocabulary",
    "manifest_hash",
    "manifest_ids",
    "measure",
    "metric_value_problem",
    "normalize",
    "normalize_legacy",
    "per_item_name",
    "phrase",
    "protocol_problems",
    "recompute",
    "recompute_cell",
    "relevant_keys",
    "schema_major",
    "sqrt_equal_width_edges",
    "sqrt_stratification",
    "training_contract",
    "training_metric_definition",
    "training_metric_problem",
    "training_naming_problems",
    "validation_pass_contract",
    "validation_pass_ids",
    "validation_pass_problems",
    "validation_set_problems",
    "vocabulary_documents",
]
