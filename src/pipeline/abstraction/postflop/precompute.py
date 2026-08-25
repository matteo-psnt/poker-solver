"""Full-coverage precomputation pipeline for combo-level abstraction.

For every canonical board on every street:

1. Compute exact equity -- and, on flop/turn, the equity-realization histogram
   -- for every canonical hand class (range-vs-range engine).
2. Bucket all (board, class) pairs per street. Flop/turn cluster over
   realization-distribution CDFs, which is potential-aware: Euclidean distance
   between 1D CDFs is the Cramer distance, an EMD-family metric, so draws
   separate from made hands of equal equity. River is weighted 1D k-means over
   scalar equity, no potential remaining.
3. Store dense bucket matrices keyed by canonical board ID.

Every legal postflop state resolves to a bucket computed on its OWN board --
there is no board clustering, no representative sampling, and no fallback.

sklearn is imported at its two call sites rather than module scope, against this
project's own rule: ``sklearn.cluster`` costs ~0.5s and pulls scipy behind it,
and this module is reachable from ``src.pipeline.services``, the facade every
reader command imports. ``tests/interfaces/test_import_weight.py`` fails if it
comes back.
"""

import logging
import math
import multiprocessing as mp
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.core.game.state import Card, Street
from src.pipeline.abstraction.config import PrecomputeConfig
from src.pipeline.abstraction.postflop.board_enumeration import (
    EXPECTED_CANONICAL_COUNTS,
    CanonicalBoardEnumerator,
)
from src.pipeline.abstraction.postflop.bucketer import (
    METADATA_FILENAME,
    N_HAND_COLUMNS,
    POSTFLOP_STREETS,
    STORAGE_VERSION,
    DenseBucketer,
    bucket_dtype,
    build_hand_column_index,
)
from src.pipeline.abstraction.postflop.canonical_hands import enumerate_hand_classes
from src.pipeline.abstraction.postflop.quality import compute_street_quality
from src.pipeline.abstraction.preflop.opponent_clusters import opponent_cluster_assignment
from src.pipeline.abstraction.utils.equity import RangeEquityEngine
from src.shared import records
from src.shared.log import progress_bars_enabled

logger = logging.getLogger(__name__)

_HAND_ID_TO_COL = build_hand_column_index()

# Equity quantization grid for the weighted 1D k-means fit (river). 2^16 bins
# keep the fit exact to ~1.5e-5 equity while bounding its input size.
_KMEANS_EQUITY_BINS = 65536

# Row cap for fitting the flop/turn histogram k-means; assignment always
# covers every row (in chunks). 2M weighted rows pin down <=1k centroids.
_KMEANS_FIT_SAMPLE = 2_000_000
_KMEANS_ASSIGN_CHUNK = 4_000_000

_MAX_CHUNK_BOARDS = 512


def _worker_compute_board_chunk(
    args: tuple[
        Sequence[tuple[int, tuple[Card, ...]]],
        int | None,
        int,
        int | None,
        tuple[np.ndarray, int] | None,
    ],
) -> list[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]]:
    """
    Compute per-class equities (and optional per-class feature vectors) for a
    chunk of boards.

    ``args`` is (boards, flop_runouts, seed, histogram_bins, ochs). ``ochs`` is
    None for the equity-realization histogram, or (cluster_of_class,
    num_clusters) to fill the feature slot with OCHS instead.

    Returns one (row, columns, equities, multiplicities, features) tuple per
    board; features is None when histogram_bins is None.
    """
    boards, flop_runouts, seed, histogram_bins, ochs_clusters = args
    engine = RangeEquityEngine(max_runouts=flop_runouts, seed=seed)

    # OCHS fills the same per-class feature slot the realization histogram
    # normally occupies, so everything downstream — the matrices, the L2 k-means,
    # the storage layout — is reused unchanged. Only the meaning of the vector
    # differs: win rate per opponent cluster, rather than probability per equity
    # bin.
    results = []
    for row, board in boards:
        table = engine.board_equities(board, histogram_bins=histogram_bins)
        classes = enumerate_hand_classes(board)

        ochs_lookup = None
        if ochs_clusters is not None:
            combos, ochs = engine.board_ochs(board, ochs_clusters[0], ochs_clusters[1])
            ochs_lookup = {frozenset((a.mask, b.mask)): ochs[i] for i, (a, b) in enumerate(combos)}

        cols = np.empty(len(classes), dtype=np.int32)
        equities = np.empty(len(classes), dtype=np.float32)
        multiplicities = np.empty(len(classes), dtype=np.uint8)
        features = (
            np.empty((len(classes), histogram_bins), dtype=np.float16)
            if histogram_bins is not None
            else None
        )
        for k, hand_class in enumerate(classes):
            cols[k] = _HAND_ID_TO_COL[hand_class.canonical.hand_id]
            equities[k] = table.equity(hand_class.representative)
            multiplicities[k] = hand_class.multiplicity
            if features is None:
                continue
            if ochs_lookup is None:
                features[k] = table.histogram(hand_class.representative)
            else:
                card_a, card_b = hand_class.representative
                features[k] = ochs_lookup[frozenset((card_a.mask, card_b.mask))]

        results.append((row, cols, equities, multiplicities, features))

    return results


def street_runouts(street: Street, flop_runouts: int | None = None) -> int:
    """What one street's equity pass costs, in runouts enumerated.

    Boards times runouts per board, which is what the pass actually walks:
    1,755 flops of 1,176 runouts each is ~70% of a build, 16,432 turns of 48 is
    ~27%, and 134,459 rivers of one is the remaining ~5%. Counting boards alone
    inverts that, and counting streets flattens it.

    Approximate on purpose, and only ever a WEIGHT: it ignores the k-means that
    follows each pass, and hand classes per board are treated as equal across
    streets. It decides how a bar is divided, never what is computed.
    """
    cards = {Street.FLOP: 3, Street.TURN: 4, Street.RIVER: 5}[street]
    exact = math.comb(52 - cards, 5 - cards)
    boards = EXPECTED_CANONICAL_COUNTS[street]
    return boards * (min(flop_runouts, exact) if street == Street.FLOP and flop_runouts else exact)


class PostflopPrecomputer:
    """
    Precomputes full-coverage combo abstraction tables.

    This is the main entry point for generating abstraction artifacts.
    """

    def __init__(self, config: PrecomputeConfig):
        self.config = config

        # Per-street outputs, filled by precompute_street.
        self._board_ids: dict[Street, np.ndarray] = {}
        self._bucket_matrices: dict[Street, np.ndarray] = {}
        self._num_buckets: dict[Street, int] = {}
        self._quality: dict[Street, dict] = {}

    def precompute_street(
        self,
        street: Street,
        board_limit: int | None = None,
        on_fraction: Callable[[float], None] | None = None,
    ) -> None:
        """
        Precompute buckets for every canonical board on a street.

        Args:
            street: Which street to precompute.
            board_limit: Optional cap on the number of canonical boards
                (lowest board IDs first). Test hook — production runs cover
                every board.
            on_fraction: How much of this street's equity pass is done, in
                [0, 1]. The bucketing that follows it is not counted; it is a
                k-means over matrices that are already in memory.
        """
        board_ids, equity_matrix, weight_matrix, hist_matrix = self.compute_street_matrices(
            street, board_limit=board_limit, on_fraction=on_fraction
        )
        logger.info(f"Bucketing {street.name} into {self.config.num_buckets[street]} buckets...")
        self.bucket_street(street, board_ids, equity_matrix, weight_matrix, hist_matrix)

        quality = self._quality[street]
        logger.info(
            f"Completed {street.name}: {quality['class_count']:,} classes "
            f"({quality['combo_count']:,} combos) into {quality['num_buckets']} buckets, "
            f"variance explained {quality['variance_explained']:.4f}"
        )

    def compute_street_matrices(
        self,
        street: Street,
        board_limit: int | None = None,
        on_fraction: Callable[[float], None] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        """
        Compute per-board equity/weight (and flop/turn histogram) matrices.

        Exposed separately from bucketing so bucket-count sweeps can reuse one
        expensive equity pass across many bucketing configurations.

        ``on_fraction`` is called with how much of this street's pass has
        finished, in [0, 1] -- the only thing a caller outside the process can
        watch, since nothing reaches disk until the whole build succeeds.
        """
        logger.info(f"Enumerating canonical boards for {street.name}...")
        enumerator = CanonicalBoardEnumerator(street)
        enumerator.enumerate()
        board_infos = sorted(enumerator.iterate(), key=lambda info: info.board_id)
        if board_limit is not None:
            board_infos = board_infos[:board_limit]

        n_boards = len(board_infos)
        board_ids = np.array([info.board_id for info in board_infos], dtype=np.int64)
        logger.info(f"Computing exact equities for {n_boards} canonical {street.name} boards...")

        # The river has no future cards, so a realization histogram is degenerate
        # there. Under OCHS the same slot carries win rates per opponent cluster
        # instead, which is the whole point: it restores a multi-dimensional
        # river feature where scalar equity had collapsed to one saturated number.
        use_ochs = street == Street.RIVER and self.config.river_feature == "ochs"
        if street == Street.RIVER:
            histogram_bins = self.config.ochs_clusters if use_ochs else None
        else:
            histogram_bins = self.config.equity_histogram_bins

        ochs_args = None
        if use_ochs:
            ochs_args = (
                opponent_cluster_assignment(
                    num_clusters=self.config.ochs_clusters, seed=self.config.seed
                ),
                self.config.ochs_clusters,
            )

        equity_matrix = np.full((n_boards, N_HAND_COLUMNS), np.nan, dtype=np.float32)
        weight_matrix = np.zeros((n_boards, N_HAND_COLUMNS), dtype=np.uint8)
        hist_matrix = (
            np.zeros((n_boards, N_HAND_COLUMNS, histogram_bins), dtype=np.float16)
            if histogram_bins is not None
            else None
        )

        num_workers = self.config.num_workers or mp.cpu_count()
        flop_runouts = self.config.flop_runouts if street == Street.FLOP else None

        boards = [(row, info.representative) for row, info in enumerate(board_infos)]
        chunk_size = min(_MAX_CHUNK_BOARDS, max(1, n_boards // (num_workers * 8)))
        chunks = [boards[i : i + chunk_size] for i in range(0, n_boards, chunk_size)]

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    _worker_compute_board_chunk,
                    (chunk, flop_runouts, self.config.seed, histogram_bins, ochs_args),
                )
                for chunk in chunks
            ]
            for finished, future in enumerate(
                tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"Computing {street.name} equities",
                    disable=not progress_bars_enabled(),
                ),
                start=1,
            ):
                # The same count the bar in this terminal draws, for the caller
                # who is not in this terminal. A node's is drawn to a log nobody
                # reads, and it was the only place this number existed.
                if on_fraction is not None:
                    on_fraction(finished / len(futures))
                for row, cols, equities, multiplicities, histograms in future.result():
                    equity_matrix[row, cols] = equities
                    weight_matrix[row, cols] = multiplicities
                    if hist_matrix is not None:
                        hist_matrix[row, cols] = histograms

        return board_ids, equity_matrix, weight_matrix, hist_matrix

    def bucket_street(
        self,
        street: Street,
        board_ids: np.ndarray,
        equity_matrix: np.ndarray,
        weight_matrix: np.ndarray,
        hist_matrix: np.ndarray | None,
        num_buckets: int | None = None,
    ) -> dict:
        """
        Bucket one street's matrices and store the result.

        Returns the street's quality metrics (also kept internally for save()).
        """
        target_buckets = num_buckets if num_buckets is not None else self.config.num_buckets[street]

        valid = ~np.isnan(equity_matrix)
        values = equity_matrix[valid].astype(np.float64)
        weights = weight_matrix[valid].astype(np.float64)
        if values.size == 0:
            raise ValueError(f"No equity data computed for {street.name}")

        is_ochs = street == Street.RIVER and self.config.river_feature == "ochs"

        if hist_matrix is None:
            bucket_flat, actual_buckets = self._bucket_scalar(values, weights, target_buckets)
            quality_extra: dict = {"bucketing": "scalar_equity"}
        elif is_ochs:
            # OCHS vectors are already the feature. Taking a CDF of them would be
            # meaningless — the components are win rates against distinct
            # opponent clusters, not a distribution over ordered bins.
            features = hist_matrix[valid].astype(np.float64)
            bucket_flat, actual_buckets, dispersion = self._bucket_histograms(
                features, weights, target_buckets, order_by="mean"
            )
            quality_extra = {
                "bucketing": "ochs",
                "ochs_clusters": int(hist_matrix.shape[-1]),
                "within_bucket_ochs_rmse": dispersion,
            }
        else:
            features = self._cdf_features(hist_matrix, valid)
            bucket_flat, actual_buckets, hist_dispersion = self._bucket_histograms(
                features, weights, target_buckets, order_by="cdf"
            )
            quality_extra = {
                "bucketing": "equity_histogram_cdf",
                "histogram_bins": int(hist_matrix.shape[-1]),
                "within_bucket_cdf_rmse": hist_dispersion,
            }

        dtype = bucket_dtype(actual_buckets)
        matrix = np.full(equity_matrix.shape, np.iinfo(dtype).max, dtype=dtype)
        matrix[valid] = bucket_flat

        quality = compute_street_quality(
            equities=values,
            buckets=bucket_flat,
            weights=weights,
            num_buckets=actual_buckets,
        )
        quality.update(quality_extra)

        self._board_ids[street] = board_ids
        self._bucket_matrices[street] = matrix
        self._num_buckets[street] = actual_buckets
        self._quality[street] = quality
        return quality

    def _bucket_scalar(
        self, values: np.ndarray, weights: np.ndarray, num_buckets: int
    ) -> tuple[np.ndarray, int]:
        """Weighted 1D k-means over scalar equities (river)."""
        # Weighted k-means on the quantized equity histogram: exact up to the
        # grid resolution, independent of how many (board, class) pairs exist.
        quantized = np.clip(
            (values * (_KMEANS_EQUITY_BINS - 1)).astype(np.int64), 0, _KMEANS_EQUITY_BINS - 1
        )
        histogram = np.bincount(quantized, weights=weights, minlength=_KMEANS_EQUITY_BINS)
        occupied_bins = np.nonzero(histogram)[0]
        points = (occupied_bins / (_KMEANS_EQUITY_BINS - 1)).reshape(-1, 1)

        # See the sibling fit below: 0.85s to import, and only precompute needs
        # it. Every reader of a built abstraction imports this module.
        from sklearn.cluster import KMeans  # noqa: PLC0415 -- 0.85s import, fit-only

        kmeans = KMeans(
            n_clusters=min(num_buckets, len(occupied_bins)),
            max_iter=self.config.kmeans_max_iter,
            n_init=self.config.kmeans_n_init,
            random_state=self.config.seed,
        )
        kmeans.fit(points, sample_weight=histogram[occupied_bins])

        # Bucket 0 = lowest equity; assignment by nearest center = boundary search.
        centers = np.unique(kmeans.cluster_centers_.ravel())
        boundaries = (centers[1:] + centers[:-1]) / 2
        return np.searchsorted(boundaries, values), len(centers)

    @staticmethod
    def _cdf_features(hist_matrix: np.ndarray, valid: np.ndarray) -> np.ndarray:
        """
        Realization-distribution CDFs for all valid cells.

        The last CDF entry is identically 1 and is dropped. Euclidean distance
        between these vectors is the (discretized) Cramér distance between the
        underlying distributions.
        """
        cdf = np.cumsum(hist_matrix[valid].astype(np.float32), axis=-1)
        return cdf[:, :-1]

    def _bucket_histograms(
        self,
        features: np.ndarray,
        weights: np.ndarray,
        num_buckets: int,
        order_by: str = "cdf",
    ) -> tuple[np.ndarray, int, float]:
        """
        K-means over multi-dimensional per-hand features.

        ``order_by`` selects how centroids are ranked into bucket ids: ``"cdf"``
        reads them as realization CDFs (flop/turn), ``"mean"`` as OCHS win-rate
        vectors (river).

        Fits on a weighted subsample, assigns every row in chunks, and orders
        buckets by ascending centroid-implied mean equity so bucket IDs stay
        comparable across streets and configs.
        """
        n = features.shape[0]
        rng = np.random.default_rng(self.config.seed)
        if n > _KMEANS_FIT_SAMPLE:
            fit_idx = rng.choice(n, size=_KMEANS_FIT_SAMPLE, replace=False)
        else:
            fit_idx = np.arange(n)

        # Measured 0.85s to import `sklearn.cluster`. This module is imported by
        # everything that READS a card abstraction -- training, evaluation, the
        # play server -- and only `precompute` ever fits one. That asymmetry is
        # the whole reason the import is here.
        from sklearn.cluster import KMeans  # noqa: PLC0415 -- 0.85s import, fit-only

        kmeans = KMeans(
            n_clusters=min(num_buckets, len(fit_idx)),
            max_iter=self.config.kmeans_max_iter,
            n_init=self.config.kmeans_n_init,
            random_state=self.config.seed,
        )
        kmeans.fit(features[fit_idx], sample_weight=weights[fit_idx])

        labels = np.empty(n, dtype=np.int64)
        for start in range(0, n, _KMEANS_ASSIGN_CHUNK):
            chunk = slice(start, min(start + _KMEANS_ASSIGN_CHUNK, n))
            labels[chunk] = kmeans.predict(features[chunk])

        centers = kmeans.cluster_centers_
        if order_by == "mean":
            # OCHS centroids: components are win rates against opponent clusters,
            # so their plain mean is the natural strength summary. The CDF formula
            # below would read them as bin probabilities and order buckets by a
            # quantity that does not exist.
            center_means = centers.mean(axis=1)
        else:
            # Mean equity implied by a centroid CDF c: sum of bin probabilities
            # times bin centers, with p = diff([0, c, 1]).
            n_bins = centers.shape[1] + 1
            bin_centers = (np.arange(n_bins) + 0.5) / n_bins
            full_cdf = np.hstack([centers, np.ones((centers.shape[0], 1))])
            probabilities = np.diff(full_cdf, axis=1, prepend=0.0)
            center_means = probabilities @ bin_centers

        order = np.argsort(center_means, kind="stable")
        relabel = np.empty_like(order)
        relabel[order] = np.arange(order.size)
        labels = relabel[labels]

        # Weighted RMS Cramér distance to the assigned centroid (fit sample).
        fit_labels = labels[fit_idx]
        ordered_centers = centers[order]
        distances_sq = ((features[fit_idx] - ordered_centers[fit_labels]) ** 2).sum(axis=1)
        dispersion = float(
            np.sqrt(np.average(distances_sq, weights=weights[fit_idx]) / centers.shape[1])
        )

        return labels, centers.shape[0], round(dispersion, 6)

    def precompute_all(
        self,
        streets: list[Street] | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> DenseBucketer:
        """Precompute all (or the given) postflop streets and return the bucketer.

        ``on_progress`` is called with (runouts done, runouts to do) as the
        build proceeds. Nothing is written to the output directory until
        :meth:`save`, so this is the only way the work is observable from
        outside the process at all — which is why it is counted in RUNOUTS and
        not in streets. Streets are three, and unequal: a canonical flop is
        1,176 runouts against a river's one, so the flop is most of the build
        and "2 of 3 streets" said the opposite.
        """
        if streets is None:
            streets = list(POSTFLOP_STREETS)

        weights = [street_runouts(street, self.config.flop_runouts) for street in streets]
        total = sum(weights)
        behind = 0
        for street, weight in zip(streets, weights, strict=True):
            within = None
            if on_progress is not None:

                def within(fraction: float, done: int = behind, of: int = weight) -> None:
                    on_progress(done + int(fraction * of), total)

            self.precompute_street(street, on_fraction=within)
            behind += weight
            if on_progress is not None:
                on_progress(behind, total)

        return self.build_bucketer()

    def build_bucketer(self) -> DenseBucketer:
        """Assemble the runtime bucketer from precomputed matrices."""
        return DenseBucketer(
            num_buckets_by_street=self._num_buckets,
            board_ids_by_street=self._board_ids,
            buckets_by_street=self._bucket_matrices,
            hand_id_to_col=_HAND_ID_TO_COL,
        )

    def save(self, path: Path) -> None:
        """
        Save the abstraction artifact.

        Creates ``metadata.json`` plus mmap-friendly ``.npy`` arrays per street
        (see ``bucketer.py`` for the storage layout).
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self.build_bucketer().save_arrays(path)

        streets = {
            street.name: {
                "num_buckets": self._num_buckets[street],
                "num_boards": int(self._board_ids[street].size),
                "quality": self._quality[street],
            }
            for street in POSTFLOP_STREETS
            if street in self._num_buckets
        }
        metadata = {
            "storage_version": STORAGE_VERSION,
            "config": self.config.model_dump(),
            "config_hash": self.config.get_config_hash(),
            "num_preflop_buckets": 169,
            "streets": streets,
        }
        records.write_snapshot(
            path / METADATA_FILENAME, metadata, records.REGISTRY[METADATA_FILENAME]
        )

        logger.info(f"Saved abstraction to {path}")

    @classmethod
    def load(cls, path: Path) -> DenseBucketer:
        """Load a precomputed abstraction artifact."""
        return DenseBucketer.load(Path(path))
