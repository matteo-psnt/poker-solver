"""How much the board-free game loses by forgetting which board it is on.

``bucket_game.derive`` averages every constant over boards: one transition
matrix per street step, one compatibility matrix per street. The kernel then
advances each player through that average separately
(``bucket_kernel._advance``, called once per seat). Both players see the same
card, so their bucket moves are correlated, and averaging first then applying
twice is exactly the step that drops the correlation.

This module prices that step *before* any kernel is written, because the price
is a property of the abstraction rather than of a trained strategy.

Two defects, two shapes
    ``transition`` is a COUPLING error. Let ``b_c`` be the joint distribution of
    ``(from bucket, to bucket)`` for one hand drawn uniformly on board ``c``.
    Two hands on the SAME board give ``T = E_c[b_c (x) b_c]``; the board-free
    game substitutes ``P = b_bar (x) b_bar``, which is two hands on
    INDEPENDENTLY DRAWN boards. That substitution is the dropped correlation,
    exactly.

    ``compatible`` is an AVERAGING error, one order lower: the kernel spends
    ``E_c[compat_c]`` where the truth is per board, so its price is the
    dispersion ``E_c||compat_c - compat_bar||^2``.

    Reporting both is the point. Fixing the coupling and assuming card removal
    follows is the mistake the OCHS episode is on file for.

The dial
    Conditioning chance on a public class ``k`` gives
    ``P_C = sum_k p_k (b_bar_k (x) b_bar_k)`` — today's game at ``C=1``, and
    ``T`` exactly when every board is its own class. ``recovered`` is the
    fraction of the error a partition closes, so ``recovered[1] == 0.0`` always
    and the curve's shape over ``C`` is the whole verdict.

Why nothing here forms a tensor
    ``T`` has ``(n_from * n_to)^2`` entries — 9e8 at the flop/turn shapes and
    3e10 at turn/river. Every norm is a contraction of ``<b_c, b_c'>``, so the
    measurement runs off the ``(boards, boards)`` Gram matrix.
    ``test_coupling.py`` pins each identity against a brute-force tensor on
    shapes small enough to build one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans

from src.core.game.state import Street
from src.engine.solver.vector.bucket_game import STREET_STEPS

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from src.engine.solver.vector.hand_context import HandContext

DTYPE = np.float64

# Streets whose compatibility matrix is worth pricing. Preflop is board-free by
# definition -- every board shares one preflop matrix, so its dispersion is
# identically zero and measuring it only adds a row of noise to the report.
COMPAT_STREETS: tuple[Street, ...] = (Street.FLOP, Street.TURN, Street.RIVER)

# Streets a board-relative relabelling may touch. NOT preflop: its 169 canonical
# classes are a function of the hand alone, identical on every board, so ranking
# them within a board would INJECT the board dependence the relabelling exists to
# remove -- and would make the preflop row of a report answer a different
# question from the rest of it.
RELABEL_STREETS: tuple[Street, ...] = (Street.FLOP, Street.TURN, Street.RIVER)


@dataclass(frozen=True, slots=True)
class ErrorGap:
    """One averaged constant's price, and what conditioning on classes buys back.

    ``relative`` is the error as a fraction of the averaged constant's own norm:
    scale-free, and zero exactly when the board carries no information.
    ``recovered`` maps class count to the fraction of the squared error a
    partition closes.
    """

    name: str
    kind: str  # "coupling" (transitions) or "dispersion" (compatibility)
    boards: int
    cells: int
    relative: float
    recovered: dict[int, float]


def _gram(vectors: sparse.csr_matrix) -> np.ndarray:
    """``H[c, c'] = <v_c, v_c'>``, the only contraction the norms below need."""
    return np.asarray((vectors @ vectors.T).todense(), dtype=DTYPE)


def _membership(labels: np.ndarray, boards: int) -> tuple[np.ndarray, np.ndarray]:
    """``(averaging operator, class weights)``: row ``k`` averages its members."""
    classes = np.unique(labels)
    operator = np.zeros((classes.shape[0], boards), dtype=DTYPE)
    for index, label in enumerate(classes):
        members = labels == label
        operator[index, members] = 1.0 / float(members.sum())
    weights = np.array([(labels == label).mean() for label in classes], dtype=DTYPE)
    return operator, weights


def _blocks(
    gram: np.ndarray, labels: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(<v_bar_k, v_c>, <v_bar_k, v_bar_l>, weights)``, with None meaning one class."""
    if labels is None:
        labels = np.zeros(gram.shape[0], dtype=np.int64)
    operator, weights = _membership(labels, gram.shape[0])
    class_to_board = operator @ gram
    return class_to_board, class_to_board @ operator.T, weights


def _coupling_errors(gram: np.ndarray, labels: np.ndarray | None) -> tuple[float, float]:
    """``(||T - P_C||^2, ||P_C||^2)`` using ``<a (x) a, b (x) b> = <a, b>^2``.

    ``labels`` of None is the ``C=1`` case, where ``P_C`` collapses to the
    single global outer product the board-free kernel actually uses.
    """
    class_to_board, class_to_class, weights = _blocks(gram, labels)
    t_norm = float((gram**2).mean())  # ||T||^2
    cross = float(weights @ (class_to_board**2).mean(axis=1))  # <T, P_C>
    p_norm = float(weights @ (class_to_class**2) @ weights)  # ||P_C||^2
    return max(t_norm - 2.0 * cross + p_norm, 0.0), p_norm


def _dispersion_errors(gram: np.ndarray, labels: np.ndarray | None) -> tuple[float, float]:
    """``(E_c||v_c - v_bar_k(c)||^2, ||v_bar||^2)`` — the first-order analogue."""
    _, class_to_class, weights = _blocks(gram, labels)
    own = float(np.diag(gram).mean())  # E_c <v_c, v_c>
    explained = float(weights @ np.diag(class_to_class))
    return max(own - explained, 0.0), float(gram.mean())


def _embed(gram: np.ndarray, dimensions: int) -> np.ndarray:
    """Coordinates whose Euclidean distances match ``||v_c - v_c'||``.

    Classical MDS on the Gram matrix. Clustering happens here rather than on
    ``v_c`` because the feature axis is up to 180,000 wide while the Gram is
    ``(boards, boards)`` — and the distances are identical either way.
    """
    size = gram.shape[0]
    centering = np.eye(size, dtype=DTYPE) - 1.0 / size
    centered = centering @ gram @ centering
    values, vectors = np.linalg.eigh(centered)
    keep = np.argsort(values)[::-1][:dimensions]
    return vectors[:, keep] * np.sqrt(np.clip(values[keep], 0.0, None))


def _partition(embedding: np.ndarray, classes: int, seed: int) -> np.ndarray:
    """k-means labels, the proxy that minimises within-class scatter of ``v_c``.

    Not literally the objective either error above measures — the coupling one
    weights by an outer product — but both are driven by
    ``E_c||v_c - v_bar_k||^2``, and the residual reported back is always exact.
    """
    if classes <= 1:
        return np.zeros(embedding.shape[0], dtype=np.int64)
    if classes >= embedding.shape[0]:
        return np.arange(embedding.shape[0], dtype=np.int64)
    fitted = KMeans(n_clusters=classes, n_init=4, random_state=seed).fit(embedding)
    return np.asarray(fitted.labels_, dtype=np.int64)


def measure(
    name: str,
    kind: str,
    per_board: sparse.csr_matrix,
    class_counts: Sequence[int],
    *,
    seed: int = 0,
    dimensions: int = 32,
) -> ErrorGap:
    """Price one averaged constant and sweep the class dial over it."""
    errors = _coupling_errors if kind == "coupling" else _dispersion_errors
    gram = _gram(per_board)
    total, reference = errors(gram, None)
    embedding = _embed(gram, min(dimensions, gram.shape[0]))

    recovered: dict[int, float] = {}
    for count in sorted(set(class_counts)):
        residual, _ = errors(gram, _partition(embedding, count, seed))
        recovered[count] = 0.0 if total <= 0.0 else float(1.0 - residual / total)

    return ErrorGap(
        name=name,
        kind=kind,
        boards=int(per_board.shape[0]),
        cells=int(per_board.shape[1]),
        relative=float(np.sqrt(total / reference)) if reference > 0.0 else 0.0,
        recovered=recovered,
    )


def _one_hot(buckets: np.ndarray, count: int) -> sparse.csr_matrix:
    """``(hands, buckets)`` indicator as a sparse gate."""
    hands = buckets.shape[0]
    return sparse.csr_matrix(
        (np.ones(hands, dtype=DTYPE), (np.arange(hands), buckets)), shape=(hands, count)
    )


def _transition_row(
    from_buckets: np.ndarray, to_buckets: np.ndarray, shape: tuple[int, int]
) -> sparse.csr_matrix:
    """One board's joint ``P(from bucket, to bucket)`` for a uniformly drawn hand."""
    from_count, to_count = shape
    flat = from_buckets * to_count + to_buckets
    hands = flat.shape[0]
    counts = np.bincount(flat, minlength=from_count * to_count).astype(DTYPE)
    return sparse.csr_matrix(counts / hands)


def _compatible_row(context: HandContext, buckets: np.ndarray, count: int) -> sparse.csr_matrix:
    """One board's ``P(both hands live, bucket pair)`` for two hands drawn uniformly.

    Card removal is counted off the 52-wide card incidence rather than the
    ``(H, H)`` blocking matrix: two distinct holdings share at most one card, so
    ``compatible = |b1||b2| - S + D`` with ``S`` the shared-card pair count and
    ``D`` the holdings a bucket pair has in common. ``test_coupling.py`` pins it
    against ``gate.T @ ~blocks @ gate``, which is what ``derive`` spends.
    """
    gate = _one_hot(buckets, count)
    hands = buckets.shape[0]

    incidence = sparse.csr_matrix(
        (
            np.ones(2 * hands, dtype=DTYPE),
            (context.hand_cards.reshape(-1), np.repeat(np.arange(hands), 2)),
        ),
        shape=(52, hands),
    )
    per_card = incidence @ gate  # (52, buckets)
    shared = np.asarray((per_card.T @ per_card).todense(), dtype=DTYPE)
    occupancy = np.asarray(gate.sum(axis=0), dtype=DTYPE).ravel()
    both = np.asarray((gate.T @ gate).todense(), dtype=DTYPE)  # holdings in both buckets

    compatible = np.outer(occupancy, occupancy) - shared + both
    return sparse.csr_matrix(compatible.ravel() / float(hands * hands))


def board_relative(buckets: np.ndarray) -> np.ndarray:
    """Renumber a board's occupied buckets densely, strongest last, from zero.

    A bucket is a slice of *(hand, board)* space: only ~100 of 600 river buckets
    exist on any one board, so an artifact bucket id says as much about WHICH
    board this is as about how strong the hand is. That identity is information
    the board-free game has thrown away and cannot get back, and no small class
    count encodes "which 100 of 600".

    Renumbering to within-board rank removes it. Bucket ids become comparable
    across boards, and what a bucket means is strength RELATIVE to the range this
    board actually produces -- a different abstraction, not a cheaper encoding of
    the same one, and it is the artifact's own strength order that makes the rank
    well defined.
    """
    return np.unique(buckets, return_inverse=True)[1]


def accumulate(
    contexts: Iterable[HandContext],
    buckets_per_street: dict[Street, int],
    *,
    relabel: bool = False,
) -> dict[str, sparse.csr_matrix]:
    """Stream a universe into one flattened row per board, per averaged constant.

    Consumes the iterable once, exactly as ``bucket_game.derive`` does, and holds
    only the flattened rows — a board's transition row has at most one nonzero
    per hand, so thousands of boards stay in tens of megabytes where the
    contexts themselves would be tens of gigabytes.
    """
    steps = [
        (step, (buckets_per_street[step[0]], buckets_per_street[step[1]])) for step in STREET_STEPS
    ]
    rows: dict[str, list[sparse.csr_matrix]] = {
        **{f"transition:{a.name}->{b.name}": [] for a, b in STREET_STEPS},
        **{f"compatible:{street.name}": [] for street in COMPAT_STREETS},
    }

    for context in contexts:
        held = {
            street: (
                board_relative(context.buckets_for(street))
                if relabel and street in RELABEL_STREETS
                else context.buckets_for(street)
            )
            for street in (Street.PREFLOP, Street.FLOP, Street.TURN, Street.RIVER)
        }
        for step, shape in steps:
            rows[f"transition:{step[0].name}->{step[1].name}"].append(
                _transition_row(held[step[0]], held[step[1]], shape)
            )
        for street in COMPAT_STREETS:
            rows[f"compatible:{street.name}"].append(
                _compatible_row(context, held[street], buckets_per_street[street])
            )

    return {name: sparse.vstack(parts, format="csr") for name, parts in rows.items()}


def measure_all(
    contexts: Iterable[HandContext],
    buckets_per_street: dict[Street, int],
    class_counts: Sequence[int],
    *,
    seed: int = 0,
    relabel: bool = False,
) -> list[ErrorGap]:
    """Price every averaged constant of the board-free game in one pass."""
    stacked = accumulate(contexts, buckets_per_street, relabel=relabel)
    return [
        measure(
            name,
            "coupling" if name.startswith("transition") else "dispersion",
            matrix,
            class_counts,
            seed=seed,
        )
        for name, matrix in stacked.items()
    ]


__all__: Sequence[str] = (
    "COMPAT_STREETS",
    "RELABEL_STREETS",
    "ErrorGap",
    "accumulate",
    "board_relative",
    "measure",
    "measure_all",
)
