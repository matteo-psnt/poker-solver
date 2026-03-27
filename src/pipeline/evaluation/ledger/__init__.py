"""The evaluation ledger.

Split three ways: what makes two scores comparable (`tiers`), how one score is
written down (`records`), and how they are read back (`queries`).
"""

from src.pipeline.evaluation.ledger.queries import (
    curve_series,
    latest_record_for_run,
    read_records,
    rebuild_ledger,
)
from src.pipeline.evaluation.ledger.records import (
    DEFAULT_LEDGER_PATH,
    LEDGER_SCHEMA_VERSION,
    RunProvenance,
    append_record,
    build_record,
    eval_slug,
    load_payload,
    payload_pointer,
    record_evaluation,
    record_instant,
    write_payload,
    write_record,
)
from src.pipeline.evaluation.ledger.tiers import (
    CONDITIONAL_TIER_KNOBS,
    TIER_KNOBS,
    build_exact_br_knobs_from_params,
    build_lbr_knobs,
    build_lbr_knobs_from_params,
    tier_key,
    tier_label,
    tier_mismatches,
)

__all__ = (
    "CONDITIONAL_TIER_KNOBS",
    "DEFAULT_LEDGER_PATH",
    "LEDGER_SCHEMA_VERSION",
    "TIER_KNOBS",
    "RunProvenance",
    "append_record",
    "build_exact_br_knobs_from_params",
    "build_lbr_knobs",
    "build_lbr_knobs_from_params",
    "build_record",
    "curve_series",
    "eval_slug",
    "latest_record_for_run",
    "load_payload",
    "payload_pointer",
    "read_records",
    "rebuild_ledger",
    "record_evaluation",
    "record_instant",
    "tier_key",
    "tier_label",
    "tier_mismatches",
    "write_payload",
    "write_record",
)
