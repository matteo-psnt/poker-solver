"""Paired A/B over solver knobs, with the determinism preconditions enforced.

## Why this exists

Knob decisions in this repo have repeatedly been made from numbers that could not
support them. Two failure modes did the damage:

1. **Multi-worker training is Hogwild**, so two runs of the same config differ.
   Between-run variance here is large enough to swamp real effects — three
   same-spec 10M runs once scored LBR 1213 / 1758 / 2094. A single-run comparison
   across arms is therefore uninterpretable, no matter how careful the arms are.
2. **Sampling-based evaluators add their own variance** on top, so a difference
   between arms mixes the knob, the training noise, and the scoring noise.

Both are avoidable, and the fix is cheap:

- **Single-worker training at a fixed seed is bit-identical.** Not
  approximately — the same config twice produces byte-identical arrays.
- **`exact_br` has zero evaluation variance** (±0.00): a deterministic exact best
  response over one fixed sampled board plan.

Compose those and an arm-vs-control difference is attributable to the knob and
nothing else. This module makes that the only way to run the comparison: workers
is not a parameter, the seed is mandatory, and :func:`run_ab` can verify the
determinism precondition empirically before trusting any result.

## What it does NOT give you

The arms are exact, but they are one point in a large space. Nothing here tells
you the result transfers to production scale, a different abstraction, or a
different iteration count — an exact answer to a small question is still an
answer to a small question. `exact_br` also scores the *board-restricted* game
(a fixed set of sampled flops), so its absolute value is not full-HUNL
exploitability; compare within a run of this harness, never across board plans.

## Usage

    result = run_ab(
        "quick_test",
        arms=[Arm("prune@110", {"solver__enable_pruning": True,
                                "solver__pruning_threshold": 110.0})],
        iterations=200_000,
        seed=42,
        verify_determinism=True,
    )
    print(format_ab_table(result))
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.pipeline.services.evaluation import evaluate_and_record
from src.pipeline.services.static_training import train_static
from src.shared.config import DEFAULT_RUNS_DIR

logger = logging.getLogger(__name__)

# Hard-wired, not a parameter. Workers write the shared table without
# synchronisation (Hogwild), so two runs of one config diverge and an
# arm-vs-control difference stops being attributable to the knob. This is still
# true on the static backend — the table is shared there too; only the
# addressing changed. Exposing this as an option would make the harness's one
# guarantee optional.
AB_NUM_WORKERS = 1

# exact_br is the only estimator with zero evaluation variance, which is what
# lets a single run per arm be conclusive. lbr is a sampling estimator: valid,
# but it needs replicates the harness does not run.
AB_METHOD = "exact_br"

CONTROL_NAME = "control"


@dataclass(frozen=True)
class Arm:
    """One configuration under test.

    ``overrides`` uses the config loader's ``__`` nesting separator, matching
    ``--set`` on the CLI (e.g. ``{"solver__pruning_threshold": 110.0}``).
    """

    name: str
    overrides: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ArmResult:
    """A trained-and-scored arm.

    ``coverage`` is the fraction of the statically-enumerated table training
    actually reached. It replaces the old "infosets discovered" count, which was
    a dynamic-backend artifact: rows all exist from the start now, so the
    question is which ones got VISITED, not which got created. For a knob that
    trades exploration for speed this is the diagnostic that matters.
    """

    name: str
    run_id: str
    overrides: dict[str, Any]
    iterations: int
    touched_rows: int
    num_rows: int
    coverage: float
    runtime_seconds: float
    exploitability_mbb: float

    @property
    def is_control(self) -> bool:
        return not self.overrides


@dataclass(frozen=True)
class AbResult:
    """A complete comparison. ``arms[0]`` is always the control."""

    config_name: str
    iterations: int
    seed: int
    arms: list[ArmResult]
    determinism_verified: bool

    @property
    def control(self) -> ArmResult:
        return self.arms[0]


class DeterminismError(RuntimeError):
    """Raised when two identical runs disagree, voiding the comparison.

    Not a warning: if the harness's own precondition fails, every number it would
    go on to report is meaningless, and reporting them anyway is how a false
    result gets into the record.
    """


def _train_and_score(
    config_name: str,
    arm: Arm,
    *,
    iterations: int,
    seed: int,
    runs_dir: Path,
) -> ArmResult:
    """Train one arm single-worker at ``seed``, then score it with exact_br."""
    logger.info("[ab] training arm %r (%d iters, seed %d)", arm.name, iterations, seed)
    started = time.monotonic()
    out = train_static(
        config_name,
        num_workers=AB_NUM_WORKERS,
        num_iterations=iterations,
        seed=seed,
        config_overrides=dict(arm.overrides) or None,
        runs_dir=runs_dir,
    )
    elapsed = time.monotonic() - started

    run_dir = runs_dir / out.run_id
    logger.info("[ab] scoring arm %r (%s)", arm.name, out.run_id)
    payload = evaluate_and_record(run_dir, method=AB_METHOD)
    exploitability = float(payload["results"]["exploitability_mbb"])

    return ArmResult(
        name=arm.name,
        run_id=out.run_id,
        overrides=dict(arm.overrides),
        iterations=out.iterations,
        touched_rows=out.touched_rows,
        num_rows=out.num_rows,
        coverage=out.coverage,
        runtime_seconds=elapsed,
        exploitability_mbb=exploitability,
    )


def run_ab(
    config_name: str,
    arms: list[Arm],
    *,
    iterations: int,
    seed: int,
    runs_dir: Path = Path(DEFAULT_RUNS_DIR),
    verify_determinism: bool = False,
) -> AbResult:
    """Train and score a control plus ``arms``, all single-worker at one seed.

    The control is an arm with no overrides and is always run first, so every
    comparison is against a baseline produced by this same harness rather than a
    number carried over from elsewhere.

    ``verify_determinism`` trains and scores the control a *second* time and
    requires an exact match. It roughly doubles the control's cost and is worth
    it the first time a config, machine, or code version is used: it converts the
    harness's central assumption from an assertion into a measurement. Raises
    :class:`DeterminismError` on mismatch.
    """
    if not arms:
        raise ValueError("run_ab needs at least one arm to compare against the control")
    if any(not arm.overrides for arm in arms):
        raise ValueError(
            "An arm with no overrides is the control, which run_ab creates itself. "
            "Give every arm at least one override."
        )
    duplicate_names = {a.name for a in arms if [x.name for x in arms].count(a.name) > 1}
    if duplicate_names:
        raise ValueError(f"Arm names must be unique; repeated: {sorted(duplicate_names)}")

    results = [
        _train_and_score(
            config_name, Arm(CONTROL_NAME), iterations=iterations, seed=seed, runs_dir=runs_dir
        )
    ]

    verified = False
    if verify_determinism:
        replica = _train_and_score(
            config_name,
            Arm(f"{CONTROL_NAME}-replica"),
            iterations=iterations,
            seed=seed,
            runs_dir=runs_dir,
        )
        if replica.exploitability_mbb != results[0].exploitability_mbb:
            raise DeterminismError(
                "Two identical runs disagree "
                f"({results[0].exploitability_mbb} vs {replica.exploitability_mbb} mbb/g), "
                "so an arm-vs-control difference cannot be attributed to the knob. "
                "Every number this comparison would report is void. Check that "
                "training is genuinely single-worker and that the seed reaches the "
                "solver."
            )
        logger.info(
            "[ab] determinism verified: two identical runs both scored %.2f mbb/g",
            replica.exploitability_mbb,
        )
        verified = True

    results.extend(
        _train_and_score(config_name, arm, iterations=iterations, seed=seed, runs_dir=runs_dir)
        for arm in arms
    )

    return AbResult(
        config_name=config_name,
        iterations=iterations,
        seed=seed,
        arms=results,
        determinism_verified=verified,
    )


def format_ab_table(result: AbResult) -> str:
    """Render a comparison as a fixed-width table, control first."""
    control = result.control
    lines = [
        f"A/B  config={result.config_name}  iters={result.iterations:,}  seed={result.seed}"
        f"  workers={AB_NUM_WORKERS}  method={AB_METHOD}"
        f"  determinism={'verified' if result.determinism_verified else 'assumed'}",
        "",
        f"{'arm':<24} {'wall':>9} {'coverage':>10} {'exploitability':>15} {'vs control':>12}",
        "-" * 74,
    ]
    for arm in result.arms:
        if arm is control:
            delta = "—"
        else:
            pct = 100.0 * (arm.exploitability_mbb - control.exploitability_mbb)
            pct /= control.exploitability_mbb
            delta = f"{pct:+.1f}%"
        lines.append(
            f"{arm.name:<24} {arm.runtime_seconds:>8.1f}s {arm.coverage:>9.1%}"
            f" {arm.exploitability_mbb:>15.2f} {delta:>12}"
        )
    lines.append(f"{'':<24} {'':>9} {'of ' + f'{control.num_rows:,}' + ' rows':>10}")

    lines += [
        "",
        "Lower exploitability is better. exact_br scores the BOARD-RESTRICTED game,",
        "so compare within this table only — the absolute value is not full-HUNL",
        "exploitability and does not transfer across board plans or abstractions.",
    ]
    if not result.determinism_verified:
        lines.append(
            "Determinism was ASSUMED, not measured. Re-run with verify_determinism "
            "the first time a config, machine, or code version is used."
        )
    return "\n".join(lines)
