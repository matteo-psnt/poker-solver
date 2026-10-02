"""Knob tiers: which evaluations are comparable at all.

A score is only meaningful against another taken with the same instrument.
These define "same instrument", which is why `ledger` refuses mismatched
tiers by design."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.pipeline.evaluation.estimators.lbr.config import LBRConfig

logger = logging.getLogger(__name__)

TIER_KNOBS = ("scorer", "opponent", "include_off_tree")

# Read off the RECORD, not off `knobs`: they say what GAME was measured, and two
# numbers from different games are not two measurements of one thing. A null
# splits from a value on purpose -- `eval_tree_fingerprint` was added after the
# limp fix changed the tree under a fixed action config, so a row without one
# was measured on a tree nothing can now identify.
IDENTITY_KNOBS = ("card_abstraction_hash", "action_config_hash", "eval_tree_fingerprint")

CONDITIONAL_TIER_KNOBS = (
    "runouts",
    "resolver_iterations",
    "lookahead_depth",
    "lookahead_top_k",
    "num_flops",
    "num_turns",
    "num_rivers",
    # exact_br responder/strategy variants. Present on a row only when set, so
    # every row written before them pairs exactly as it did.
    "in_abstraction",
    "policy_threshold",
    "purify",
    "conditional_chance",
    # WHICH strategy of the checkpoint was scored -- the average, the current
    # iterate, or a late window of the average. Same arrays, different policy.
    "policy_iterate",
    "avg_window_from",
    "avg_gamma",
    # A mixture is a different STRATEGY, not a different estimator, so it splits
    # the tier exactly as a reweighted average does.
    "mix_run",
    "mix_at",
    "mix_weight",
    # blueprint_match: WHO was played, and at which rung. A chip edge is a
    # statement about one pairing, and head-to-head results are intransitive,
    # so two edges over different opponents are not two readings of one thing.
    "run_b",
    "rung_b",
    # ...and HOW each side was served. The seat purifies at 0.20 while every gate
    # matched raw tables, so a fielded-configuration row must not pair with the
    # raw one that picked the rung. Conditional, so rows written before this
    # pair exactly as they did.
    "policy_threshold_a",
    "policy_threshold_b",
    # settling: the rung gap the drift was measured over IS the instrument.
    "from_iteration",
    "rung_gap",
    # How many deals the estimate rests on. Both match builders have always put
    # this in `knobs` and `tier_key` never read it, so a 2,000-deal probe and a
    # 20,000-deal run of the SAME pairing hashed to one tier -- measured 09-21,
    # digest 182b114e for both -- and every reader here pools UNWEIGHTED, which
    # averages a +-153 mbb row into +-49 ones as though they were equals.
    "num_deals",
    # Resolver knobs. These decide what the resolver PLAYS, so two rows across
    # any of them measure different strategies -- `policy_blend_alpha` most of
    # all (alpha=0 scored +2140 where the shipped alpha scored -781.6). Both
    # builders recorded them in `knobs`/`results` while neither guard listed
    # them here, so every resolver A/B paired and grouped as one instrument.
    "leaf_continuation_fraction",
    "resolver_max_iterations",
    "resolver_blend_alpha",
    "resolver_root_prior_weight",
    "resolver_min_street",
    "root_prior_weight",
    "leaf_rollouts",
    "allin_runouts",
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
    resolver_blend_alpha: float | None = None,
    resolver_root_prior_weight: float | None = None,
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
        # What the deployed row PLAYS, not just how long it searched.
        knobs["resolver_blend_alpha"] = resolver_blend_alpha
        knobs["resolver_root_prior_weight"] = resolver_root_prior_weight
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
        resolver_blend_alpha=results.get("resolver_blend_alpha"),
        resolver_root_prior_weight=results.get("resolver_root_prior_weight"),
        lookahead_depth=config.lookahead_depth,
        lookahead_top_k=config.lookahead_top_k,
    )


def build_exact_br_knobs_from_params(
    *,
    num_flops: int,
    num_turns: int,
    num_rivers: int,
    board_seed: int,
    in_abstraction: bool = False,
    policy_threshold: float = 0.0,
    purify: bool = False,
    conditional_chance: bool = False,
    policy_iterate: str = "average",
    avg_window_from: int | None = None,
    avg_gamma: float | None = None,
    mix_run: str | None = None,
    mix_at: int | None = None,
    mix_weight: float = 0.5,
    deployed: bool = False,
    resolver_iterations: int | None = None,
    resolver_prior_weight: float | None = None,
) -> dict[str, Any]:
    """Canonical exact-BR knob tier: the board plan IS the comparison tier.

    ``base_seed`` mirrors ``board_seed`` so the pairing guard applies unchanged:
    two exact-BR evals are comparable iff they scored the same sampled boards.
    Values are deterministic points — within a matched tier a difference is
    exact, with no paired samples or p-value involved. A bucket-constrained
    responder, a transformed blueprint or conditional chance is a different
    number, so those enter the tier -- but only when set, so existing rows
    keep their tier.
    """
    knobs: dict[str, Any] = {
        "num_flops": num_flops,
        "num_turns": num_turns,
        "num_rivers": num_rivers,
        "base_seed": board_seed,
    }
    if in_abstraction:
        knobs["in_abstraction"] = True
    if purify:
        knobs["purify"] = True
    elif policy_threshold > 0.0:
        knobs["policy_threshold"] = policy_threshold
    if conditional_chance:
        knobs["conditional_chance"] = True
    if policy_iterate != "average":
        knobs["policy_iterate"] = policy_iterate
    if avg_window_from is not None:
        knobs["avg_window_from"] = int(avg_window_from)
    if avg_gamma is not None:
        knobs["avg_gamma"] = float(avg_gamma)
    if mix_run is not None:
        knobs["mix_run"] = str(mix_run)
        knobs["mix_at"] = mix_at
        knobs["mix_weight"] = float(mix_weight)
    if deployed:
        # blueprint+resolver is a DIFFERENT STRATEGY, not a knob on the same
        # one, so it has to enter the tier or a deployed row pairs with a
        # blueprint row and the difference reads as noise. Measured 08-31:
        # the two scored 3076.4 and 2409.8 under identical recorded knobs.
        knobs["deployed"] = True
        knobs["resolver_iterations"] = resolver_iterations
        knobs["resolver_prior_weight"] = resolver_prior_weight
    return knobs


def build_resolver_match_knobs(results: dict[str, Any]) -> dict[str, Any]:
    """Canonical resolver-gate knob tier, read back off the RESULTS.

    Off the results rather than off the caller's arguments on purpose: both
    knobs may be `None` at the call site meaning "whatever the run's config
    says", and two arms that resolved to different values must not share a tier
    just because both were spelled as a default.

    `leaf_continuation_fraction` is IN the tier because a different leaf
    valuation is a different game to score -- the same reason the resolver's
    `max_depth` change was recorded as a tier break. Pairing two arms across it
    is the comparison being made, and `ledger` refusing to do it
    silently is correct.
    """
    return {
        "num_deals": results["num_deals"],
        "base_seed": results["seed"],
        "leaf_continuation_fraction": results["leaf_continuation_fraction"],
        "resolver_max_iterations": results["resolver_max_iterations"],
        "allin_runouts": results["allin_runouts"],
        # A prior on the root strategy changes what is PLAYED, exactly as the
        # leaf valuation does, so it splits the tier for the same reason.
        "root_prior_weight": results.get("root_prior_weight", 0.0),
        # Sampled runouts ARE the leaf valuation, so they define the game being
        # solved exactly as `leaf_continuation_fraction` does.
        "leaf_rollouts": results.get("leaf_rollouts", 8),
        # How much of the resolver's answer is actually PLAYED. The largest
        # single lever on this number, and it was absent from the tier.
        "resolver_blend_alpha": results.get("resolver_blend_alpha"),
    }


def build_blueprint_match_knobs(results: dict[str, Any]) -> dict[str, Any]:
    """Canonical blueprint-match knob tier, read back off the RESULTS.

    The OPPONENT is the instrument here: a chip edge over one arm says nothing
    about the edge over another, so `run_b` and the rung of it that played are
    tier knobs, not annotations. Head-to-head results are also intransitive,
    which is exactly why pairing two rows measured against different opponents
    has to be refused rather than averaged.
    """
    knobs: dict[str, Any] = {
        "run_b": results["run_b"],
        "rung_b": results.get("rung_b"),
        "num_deals": results["num_deals"],
        "base_seed": results["seed"],
    }
    # Emitted only when a side was actually purified, so every raw-vs-raw row
    # ever written keeps the tier digest it already has.
    for side in ("a", "b"):
        threshold = float(results.get(f"policy_threshold_{side}") or 0.0)
        if threshold > 0.0:
            knobs[f"policy_threshold_{side}"] = threshold
    # A DEPLOYED side A is a different player, so it must not pair with the bare
    # match that chose the rung -- and at an unpinned iteration count it is not
    # comparable to another deployed arm either. `opponent` is already a tier knob
    # (lbr's), so naming it that is what splits the tier. Conditional, so every
    # bare row keeps the digest it has.
    if results.get("opponent") == "deployed":
        knobs["opponent"] = "deployed"
        knobs["resolver_max_iterations"] = results.get("resolver_max_iterations")
        knobs["root_prior_weight"] = results.get("root_prior_weight")
        knobs["resolver_blend_alpha"] = results.get("resolver_blend_alpha")
        knobs["resolver_min_street"] = results.get("resolver_min_street")
    return knobs


def build_settling_knobs(results: dict[str, Any]) -> dict[str, Any]:
    """Canonical settling knob tier.

    The RUNG GAP is the instrument. Drift measured over 400 rungs and drift over
    2,000 are not two readings of one quantity -- the average strategy mechanically
    moves less the wider the base it is averaged over -- so a curve is only a curve
    at a fixed gap.
    """
    return {
        "from_iteration": results.get("from_iteration"),
        "rung_gap": results.get("rung_gap"),
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
        *(record.get(k) for k in IDENTITY_KNOBS),
        *(knobs.get(k) for k in TIER_KNOBS),
        *(knobs.get(k) for k in CONDITIONAL_TIER_KNOBS),
        knobs.get("base_seed"),
    )


def tier_digest(record: dict[str, Any]) -> str:
    """:func:`tier_key`, hashed, for storing beside a row.

    Here rather than at either call site because there are two -- the importer
    and the live writer -- and a tier computed two ways is two answers to which
    evals may be compared. A five-column version of this pairing reported
    -100.0 mbb where the truth was -60.0.
    """
    return hashlib.sha256(
        json.dumps(list(tier_key(record)), sort_keys=True, default=str).encode()
    ).hexdigest()[:32]


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
    # Truncated: these are 16-hex digests and the label is read in a table row.
    parts += [f"{k}={str(record[k])[:8]}" for k in IDENTITY_KNOBS if record.get(k) is not None]
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

    for knob in IDENTITY_KNOBS:
        if a.get(knob) != b.get(knob):
            reasons.append(  # noqa: PERF401 - multi-line message reads worse as a genexp
                f"{knob} differs ({a.get(knob)!r} vs {b.get(knob)!r}): the two rows measured "
                "different games, so their exploitability numbers are not on one scale."
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
                f"{knob} differs ({ka.get(knob)!r} vs {kb.get(knob)!r}): the two rows were "
                "measured with a different instrument setting, so they are not comparable."
            )

    na = a.get("results", {}).get("num_hands")
    nb = b.get("results", {}).get("num_hands")
    if na != nb:
        reasons.append(f"num_hands differs ({na} vs {nb}): paired samples must be equal-length.")

    return reasons
