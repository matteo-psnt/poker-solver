"""Service-layer APIs for training and evaluation orchestration.

The one seam between transports (headless CLI, the Azure task wrapper, the
interactive menu) and the pipeline internals. A transport composes calls
from here; it never reaches past them, which is what keeps a cloud run and a
local run doing the same thing.

The flat ``src.pipeline.services`` namespace is the public API. Submodules
group by concern only:

``runs``
    Readers over the run directory — what exists, what state it is in.
``static_training``
    Start a run over the static tree, or continue one to an absolute target.
``bucketing``
    Produce a card abstraction, or measure one at several bucket counts.
``scoring``
    Score a run; one entrypoint per estimator, plus the record orchestrator.
``experiments``
    Read the experiment record — curves, baselines, arm-vs-control attribution.
"""

from src.pipeline.services.bucketing import precompute_abstraction, sweep_bucket_counts
from src.pipeline.services.experiments import (
    CONTROL_ARM,
    DEFAULT_BASELINE_PATH,
    ArmResult,
    Baseline,
    CurveOutput,
    CurvePoint,
    ExperimentReport,
    RunDigest,
    experiment_report,
    exploitability_curve,
    load_baseline,
    promote_baseline,
    run_digest,
)
from src.pipeline.services.runs import (
    RunSummary,
    checkpoint_iteration_of,
    describe_runs,
    list_runs,
    load_run_metadata,
)
from src.pipeline.services.scoring import (
    BLUEPRINT_MATCH_ESTIMATOR_LABEL,
    EXACT_BR_ESTIMATOR_LABEL,
    LBR_ESTIMATOR_LABEL,
    EvaluationOutput,
    evaluate_and_record,
    evaluate_blueprint_match,
    evaluate_run_exact_br,
    evaluate_run_lbr,
    evaluate_run_resolver_gate,
    record_blueprint_match,
)
from src.pipeline.services.static_training import StaticTrainingOutput, train_static
from src.pipeline.services.vector_blueprint import (
    VectorBlueprintOutput,
    train_vector_blueprint,
)
from src.pipeline.training.run_tracker import ExperimentTag

__all__ = [
    "BLUEPRINT_MATCH_ESTIMATOR_LABEL",
    "CONTROL_ARM",
    "DEFAULT_BASELINE_PATH",
    "EXACT_BR_ESTIMATOR_LABEL",
    "LBR_ESTIMATOR_LABEL",
    "ArmResult",
    "Baseline",
    "CurveOutput",
    "CurvePoint",
    "EvaluationOutput",
    "ExperimentReport",
    "ExperimentTag",
    "RunDigest",
    "RunSummary",
    "StaticTrainingOutput",
    "VectorBlueprintOutput",
    "checkpoint_iteration_of",
    "describe_runs",
    "evaluate_and_record",
    "evaluate_blueprint_match",
    "evaluate_run_exact_br",
    "evaluate_run_lbr",
    "evaluate_run_resolver_gate",
    "experiment_report",
    "exploitability_curve",
    "list_runs",
    "load_baseline",
    "load_run_metadata",
    "precompute_abstraction",
    "promote_baseline",
    "record_blueprint_match",
    "run_digest",
    "sweep_bucket_counts",
    "train_static",
    "train_vector_blueprint",
]
