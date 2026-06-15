"""Every norm in ``coupling`` is a Gram contraction; these build the tensor.

The module's whole efficiency argument is that ``||T - P_C||`` never forms ``T``.
That is exactly the class of shortcut the showdown-orientation miss came from —
the arithmetic is invisible to zero-sum and monotonicity checks alike — so each
identity is pinned here against a brute-force tensor on shapes small enough to
build one, and the card-incidence shortcut against the blocking matrix
``bucket_game.derive`` actually spends.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from src.core.game.state import Street
from src.engine.solver.vector import bucket_game, coupling
from tests.engine.solver.vector.contexts import ordered_context

COUNTS = {Street.PREFLOP: 169, Street.FLOP: 6, Street.TURN: 5, Street.RIVER: 4}


def _rows(rng: np.random.Generator, boards: int, cells: int) -> sparse.csr_matrix:
    """Random per-board distributions — the ``v_c`` the identities act on."""
    dense = rng.random((boards, cells))
    return sparse.csr_matrix(dense / dense.sum(axis=1, keepdims=True))


def _brute_coupling(dense: np.ndarray, labels: np.ndarray | None) -> tuple[float, float]:
    """``(||T - P_C||^2, ||P_C||^2)`` with every tensor built explicitly."""
    outer = np.einsum("ci,cj->cij", dense, dense)
    t = outer.mean(axis=0)
    if labels is None:
        labels = np.zeros(dense.shape[0], dtype=np.int64)
    p = np.zeros_like(t)
    for label in np.unique(labels):
        members = labels == label
        mean = dense[members].mean(axis=0)
        p += members.mean() * np.outer(mean, mean)
    return float(((t - p) ** 2).sum()), float((p**2).sum())


def _brute_dispersion(dense: np.ndarray, labels: np.ndarray | None) -> tuple[float, float]:
    if labels is None:
        labels = np.zeros(dense.shape[0], dtype=np.int64)
    residual = 0.0
    for label in np.unique(labels):
        members = labels == label
        mean = dense[members].mean(axis=0)
        residual += float(((dense[members] - mean) ** 2).sum())
    return residual / dense.shape[0], float((dense.mean(axis=0) ** 2).sum())


@pytest.mark.parametrize("labels_of", [None, "split", "singleton"])
def test_coupling_norms_match_the_explicit_tensor(labels_of: str | None) -> None:
    rng = np.random.default_rng(11)
    rows = _rows(rng, boards=9, cells=7)
    dense = np.asarray(rows.todense())
    labels = {
        None: None,
        "split": np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
        "singleton": np.arange(9),
    }[labels_of]

    gram = np.asarray((rows @ rows.T).todense())
    got = coupling._coupling_errors(gram, labels)
    expected = _brute_coupling(dense, labels)
    assert got[0] == pytest.approx(expected[0], rel=1e-9, abs=1e-15)
    assert got[1] == pytest.approx(expected[1], rel=1e-9, abs=1e-15)


@pytest.mark.parametrize("labels_of", [None, "split"])
def test_dispersion_norms_match_the_explicit_tensor(labels_of: str | None) -> None:
    rng = np.random.default_rng(12)
    rows = _rows(rng, boards=9, cells=7)
    dense = np.asarray(rows.todense())
    labels = None if labels_of is None else np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

    gram = np.asarray((rows @ rows.T).todense())
    got = coupling._dispersion_errors(gram, labels)
    expected = _brute_dispersion(dense, labels)
    assert got[0] == pytest.approx(expected[0], rel=1e-9, abs=1e-15)
    assert got[1] == pytest.approx(expected[1], rel=1e-9, abs=1e-15)


def test_one_class_recovers_nothing_and_singletons_recover_everything() -> None:
    """The dial's two endpoints, which fix its orientation.

    ``C=1`` IS the board-free game, so it must recover exactly zero; one class
    per board reproduces ``T``, so it must recover exactly one. A sign or
    transposition slip shows up here before it can look like a real curve.
    """
    rng = np.random.default_rng(13)
    rows = _rows(rng, boards=12, cells=5)
    for kind in ("coupling", "dispersion"):
        gap = coupling.measure("probe", kind, rows, [1, 12])
        assert gap.recovered[1] == pytest.approx(0.0, abs=1e-12)
        assert gap.recovered[12] == pytest.approx(1.0, abs=1e-9)
        assert gap.relative > 0.0


def test_identical_boards_have_no_error_to_recover() -> None:
    """A universe where the board says nothing must price at the floor.

    ``relative`` is a square root of a difference of same-scale quantities, so a
    true zero lands near 1e-8 rather than 1e-16 — the cancellation is exact in
    double precision and the sqrt halves the exponent. 1e-6 is the honest noise
    floor for this statistic; any real coupling measured below it is not a
    reading.
    """
    single = _rows(np.random.default_rng(14), boards=1, cells=5)
    repeated = sparse.vstack([single] * 8, format="csr")
    for kind in ("coupling", "dispersion"):
        assert coupling.measure("flat", kind, repeated, [1, 4]).relative < 1e-6


def test_compatible_row_matches_the_blocking_matrix_derive_spends() -> None:
    """The 52-wide card-incidence shortcut against the ``(H, H)`` ground truth."""
    rng = np.random.default_rng(15)
    context = ordered_context(rng, COUNTS, num_cards=14)
    street = Street.RIVER
    count = COUNTS[street]

    gate = np.zeros((context.num_hands, count))
    gate[np.arange(context.num_hands), context.buckets_for(street)] = 1.0
    expected = gate.T @ (~context.blocks).astype(float) @ gate / context.num_hands**2

    buckets = context.buckets_for(street)
    got = coupling._compatible_row(context, buckets, count).toarray().reshape(count, count)
    assert got == pytest.approx(expected, abs=1e-12)


def test_transition_rows_average_to_the_matrix_derive_builds() -> None:
    """Averaging the per-board rows must reproduce ``derive``'s marginal exactly.

    This is what makes ``recovered[1] == 0`` mean "the board-free game": the
    ``C=1`` model has to be the game ``bucket_game`` actually ships, not merely
    something adjacent to it.
    """
    rng = np.random.default_rng(16)
    contexts = [ordered_context(rng, COUNTS, num_cards=14) for _ in range(6)]
    step = (Street.TURN, Street.RIVER)
    shape = (COUNTS[step[0]], COUNTS[step[1]])

    stacked = coupling.accumulate(contexts, COUNTS)[f"transition:{step[0].name}->{step[1].name}"]
    joint = np.asarray(stacked.todense()).mean(axis=0).reshape(shape)
    totals = joint.sum(axis=1, keepdims=True)
    marginal = np.divide(joint, totals, out=np.zeros_like(joint), where=totals > 0)

    expected = bucket_game.derive(contexts, COUNTS).transitions[step]
    assert marginal == pytest.approx(expected, abs=1e-12)


class TestBoardRelativeRelabelling:
    """The candidate abstraction change, pinned before it is measured on.

    `board_relative` is a claim about bucket IDENTITY, and every way of getting
    it wrong still produces a plausible-looking number: an off-by-one keeps the
    ordering, a sort on the wrong axis keeps the count. So the properties are
    checked directly rather than inferred from the gap moving.
    """

    def test_it_is_dense_and_zero_based(self) -> None:
        buckets = np.array([7, 7, 42, 500, 42, 0])
        assert coupling.board_relative(buckets).tolist() == [1, 1, 2, 3, 2, 0]

    def test_it_preserves_the_strength_order(self) -> None:
        """The artifact numbers buckets by strength, which is what makes a rank
        meaningful — so the relabelling must be monotone in the original id."""
        rng = np.random.default_rng(21)
        buckets = rng.integers(0, 600, size=400)
        ranked = coupling.board_relative(buckets)
        order = np.argsort(buckets, kind="stable")
        assert np.all(np.diff(ranked[order]) >= 0)

    def test_it_leaves_an_already_dense_board_untouched(self) -> None:
        dense = np.array([0, 1, 2, 3, 2, 1])
        assert coupling.board_relative(dense).tolist() == dense.tolist()

    def test_relabelling_changes_what_is_measured(self) -> None:
        """Guards the arm's power: if the two agreed, the comparison is vacuous.

        Needs MORE buckets than a board has hands, which is the production
        situation and the whole reason the relabelling exists — ~100 of 600
        river buckets live on any one board. `ordered_context` spreads ranks
        across every bucket it is given, so at `COUNTS` occupancy is dense and
        the relabelling is correctly a no-op.
        """
        sparse_counts = {
            Street.PREFLOP: 169,
            Street.FLOP: 100,
            Street.TURN: 200,
            Street.RIVER: 400,
        }
        rng = np.random.default_rng(22)
        contexts = [ordered_context(rng, sparse_counts, num_cards=14) for _ in range(6)]
        plain = coupling.accumulate(contexts, sparse_counts, relabel=False)
        ranked = coupling.accumulate(contexts, sparse_counts, relabel=True)
        name = "transition:TURN->RIVER"
        assert (plain[name] - ranked[name]).nnz > 0

    def test_a_dense_board_is_left_alone_end_to_end(self) -> None:
        """The complement: where every bucket is occupied there is nothing to
        renumber, so the two accumulations must agree exactly.

        Every street needs FEWER buckets than the board has hands — `COUNTS`
        would not do, because its 169 preflop classes against 36 live hands are
        themselves sparse.
        """
        dense_counts = {
            Street.PREFLOP: 7,
            Street.FLOP: 6,
            Street.TURN: 5,
            Street.RIVER: 4,
        }
        rng = np.random.default_rng(23)
        contexts = [ordered_context(rng, dense_counts, num_cards=14) for _ in range(4)]
        plain = coupling.accumulate(contexts, dense_counts, relabel=False)
        ranked = coupling.accumulate(contexts, dense_counts, relabel=True)
        for name, matrix in plain.items():
            assert (matrix - ranked[name]).nnz == 0, name
