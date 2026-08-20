"""Knob tiers: which evaluations are comparable at all.

A score is only meaningful against another taken with the same instrument.
These define "same instrument", which is why `compare` and `report` refuse
mismatched tiers by design."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.pipeline.evaluation.estimators.lbr.config import LBRConfig

logger = logging.getLogger(__name__)

TIER_KNOBS = ("scorer", "opponent", "include_off_tree")

CONDITIONAL_TIER_KNOBS = (
    "runouts",
    "resolver_iterations",
    "lookahead_depth",
    "lookahead_top_k",
    "num_flops",
    "num_turns",
    "num_rivers",
)


def knob_hash(knobs: dict[str, Any]) -> str:
    """Short digest of a knob set, used by `records` to name a result file."""
    digest = hashlib.sha256(json.dumps(knobs, sort_keys=True).encode()).hexdigest()
    return digest[:8]


def build_lbr_knobs_from_params(
    *,
    scorer: str,
    opponent: str,
    hands: int,
    runouts: int,
    include_off_tree: bool,
    base_seed: Any,
    resolver_iterations: int | None = None,
    lookahead_depth: int | None = None,
    lookahead_top_k: int | None = None,
) -> dict[str, Any]:
    """Canonical LBR knob tier, built from explicit values.

    ``base_seed`` is the seed the deals were actually drawn from (LBR resolves a random
    seed when none is passed and reports it back), which is the value paired comparison
    must match on. Tier-specific knobs are included only when they apply, so a
    blueprint+myopic eval and a deployed+lookahead eval never collide on knob shape.
    """
    knobs: dict[str, Any] = {
        "scorer": scorer,
        "opponent": opponent,
        "hands": hands,
        "runouts": runouts,
        "include_off_tree": bool(include_off_tree),
        "base_seed": base_seed,
    }
    if opponent == "deployed":
        knobs["resolver_iterations"] = resolver_iterations
    if scorer == "lookahead":
        knobs["lookahead_depth"] = lookahead_depth
        knobs["lookahead_top_k"] = lookahead_top_k
    return knobs


def build_lbr_knobs(config: LBRConfig, results: dict[str, Any]) -> dict[str, Any]:
    """Canonical LBR knob tier for an eval that ran under ``config``.

    Deriving the tier from the same :class:`LBRConfig` the eval consumed makes
    "every transport records identical tiers" structural — the guardrail in
    :func:`tier_mismatches` only works if all surfaces agree on exactly what
    "same tier" means. ``base_seed`` and the deployed resolver's pinned
    ``resolver_iterations`` come from the effective ``results`` because both are
    resolved during the eval, not fixed by the config object.
    """
    return build_lbr_knobs_from_params(
        scorer=config.scorer,
        opponent=config.opponent,
        hands=config.num_hands,
        runouts=config.equity_runouts,
        include_off_tree=config.include_off_tree,
        base_seed=results.get("base_seed"),
        resolver_iterations=results.get("resolver_iterations"),
        lookahead_depth=config.lookahead_depth,
        lookahead_top_k=config.lookahead_top_k,
    )


def build_exact_br_knobs_from_params(
    *, num_flops: int, num_turns: int, num_rivers: int, board_seed: int
) -> dict[str, Any]:
    """Canonical exact-BR knob tier: the board plan IS the comparison tier.

    ``base_seed`` mirrors ``board_seed`` so the pairing guard applies unchanged:
    two exact-BR evals are comparable iff they scored the same sampled boards.
    Values are deterministic points — within a matched tier a difference is
    exact, with no paired samples or p-value involved.
    """
    return {
        "num_flops": num_flops,
        "num_turns": num_turns,
        "num_rivers": num_rivers,
        "base_seed": board_seed,
    }


def build_resolver_match_knobs(results: dict[str, Any]) -> dict[str, Any]:
    """Canonical resolver-gate knob tier, read back off the RESULTS.

    Off the results rather than off the caller's arguments on purpose: both
    knobs may be `None` at the call site meaning "whatever the run's config
    says", and two arms that resolved to different values must not share a tier
    just because both were spelled as a default.

    `leaf_continuation_fraction` is IN the tier because a different leaf
    valuation is a different game to score -- the same reason the resolver's
    `max_depth` change was recorded as a tier break. Pairing two arms across it
    is the comparison being made, and `compare`/`report` refusing to do it
    silently is correct.
    """
    return {
        "num_deals": results["num_deals"],
        "base_seed": results["seed"],
        "leaf_continuation_fraction": results["leaf_continuation_fraction"],
        "resolver_max_iterations": results["resolver_max_iterations"],
    }


def tier_key(record: dict[str, Any]) -> tuple[Any, ...]:
    """Identity of the comparison tier a row belongs to.

    The same rule :func:`tier_mismatches` enforces pairwise, expressed as a
    groupable key. Both must cover the SAME knobs or they contradict each other:
    without the conditional ones a depth-2 and a depth-4 lookahead eval hash into
    one tier and get plotted on a single axis, which is exactly the silent
    instrument-mixing a tier is supposed to prevent. Same for exact_br rows scored
    over different board budgets.
    """
    knobs = record.get("knobs") or {}
    return (
        record.get("method"),
        *(knobs.get(k) for k in TIER_KNOBS),
        *(knobs.get(k) for k in CONDITIONAL_TIER_KNOBS),
        knobs.get("base_seed"),
    )


def tier_label(record: dict[str, Any]) -> str:
    """Human-readable one-line description of a row's tier.

    Must name every knob :func:`tier_key` splits on, or two genuinely different
    tiers render as identical strings -- and the operator sees "also recorded, not
    mixed in: <the same text>" with no way to tell what ``--tier 1`` would select.
    Conditional knobs are shown only when present, so a myopic row is not padded
    with lookahead fields that do not apply to it.
    """
    knobs = record.get("knobs") or {}
    parts = [str(record.get("method") or "?")]
    parts += [f"{k}={knobs[k]}" for k in TIER_KNOBS if knobs.get(k) is not None]
    parts += [f"{k}={knobs[k]}" for k in CONDITIONAL_TIER_KNOBS if knobs.get(k) is not None]
    if knobs.get("base_seed") is not None:
        parts.append(f"seed={knobs['base_seed']}")
    return " ".join(parts)


def tier_mismatches(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    """Return human-readable reasons two ledger rows must not be paired (empty if OK).

    Enforces the two rules that were previously discipline-only: a *shared, non-null
    base seed* (paired common-random-numbers requires hand-for-hand identical deals)
    and *identical comparison-tier knobs* (never mix scorer/opponent/off-tree). Equal
    hand counts are required too, since paired stats need equal-length sequences.

    Also refuses to pair across ``method``, and to pair rows whose payloads carry no
    per-hand samples: two ``exact_br`` rows used to pass every check vacuously (their
    knobs have no scorer/opponent/off-tree keys, so ``None == None``) and then fail
    downstream with a bare ``KeyError: 'pair_samples_mbb'``.
    """
    reasons: list[str] = []
    ka, kb = a.get("knobs", {}), b.get("knobs", {})

    method_a, method_b = a.get("method"), b.get("method")
    if method_a != method_b:
        reasons.append(
            f"method differs ({method_a!r} vs {method_b!r}): these are different "
            "estimators, not two measurements of the same thing."
        )
    elif method_a == "exact_br":
        reasons.append(
            "exact_br rows carry no per-hand samples, so there is nothing to pair. "
            "Compare their exploitability_mbb directly — within a matched board tier "
            "the difference is exact and needs no p-value."
        )

    for knob in ("card_abstraction_hash", "action_config_hash"):
        if a.get(knob) != b.get(knob):
            reasons.append(  # noqa: PERF401 - multi-line message reads worse as a genexp
                f"{knob} differs ({a.get(knob)!r} vs {b.get(knob)!r}): the two runs are "
                "bucketed differently, so their exploitability numbers are not on one scale."
            )

    seed_a, seed_b = ka.get("base_seed"), kb.get("base_seed")
    if seed_a is None or seed_b is None:
        reasons.append(
            "base_seed missing on one side: paired CRN comparison needs both evals run "
            "with the same explicit --seed so hand i is the same deal in both."
        )
    elif seed_a != seed_b:
        reasons.append(
            f"base_seed differs ({seed_a} vs {seed_b}): the deals are not paired, so the "
            "variance cancellation behind the p-value does not hold."
        )

    for knob in TIER_KNOBS:
        if ka.get(knob) != kb.get(knob):
            reasons.append(  # noqa: PERF401 - multi-line message reads worse as a genexp
                f"{knob} differs ({ka.get(knob)!r} vs {kb.get(knob)!r}): mixing tiers "
                "compares two different exploiters/strategies, not two runs."
            )

    for knob in CONDITIONAL_TIER_KNOBS:
        if (knob in ka or knob in kb) and ka.get(knob) != kb.get(knob):
            reasons.append(  # noqa: PERF401 - multi-line message reads worse as a genexp
                f"{knob} differs ({ka.get(knob)!r} vs {kb.get(knob)!r}): the exploiter "
                "searched to a different depth/width, so the two numbers are not comparable."
            )

    na = a.get("results", {}).get("num_hands")
    nb = b.get("results", {}).get("num_hands")
    if na != nb:
        reasons.append(f"num_hands differs ({na} vs {nb}): paired samples must be equal-length.")

    return reasons
