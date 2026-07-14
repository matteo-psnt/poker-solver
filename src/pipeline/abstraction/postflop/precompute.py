"""
Full-coverage precomputation pipeline for combo-level abstraction.

For every canonical board on every street:
1. Compute exact equity — and, on flop/turn, the equity-realization
   histogram — for every canonical hand class (range-vs-range engine)
2. Bucket all (board, class) pairs per street:
   - flop/turn: k-means over realization-distribution CDFs (potential-aware —
     Euclidean distance between 1D CDFs is the Cramér distance, an
     EMD-family metric, so draws separate from made hands of equal equity)
   - river: weighted 1D k-means over scalar equity (no potential remains)
3. Store dense bucket matrices keyed by canonical board ID

Every legal postflop state resolves to a bucket computed on its own board —
there is no board clustering, no representative sampling, and no fallback.
"""

import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

from src.core.game.state import Card, Street
from src.pipeline.abstraction.config import PrecomputeConfig
from src.pipeline.abstraction.postflop.board_enumeration import CanonicalBoardEnumerator
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
from src.pipeline.abstraction.utils.equity import RangeEquityEngine

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
    args: tuple[list[tuple[int, tuple[Card, ...]]], int | None, int, int | None],
) -> list[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]]:
    """
    Compute per-class equities (and optional realization histograms) for a
    chunk of boards.

    Returns one (row, columns, equities, multiplicities, histograms) tuple per
    board; histograms is None when histogram_bins is None.
    """
    boards, flop_runouts, seed, histogram_bins = args
    engine = RangeEquityEngine(max_runouts=flop_runouts, seed=seed)

    results = []
    for row, board in boards:
        table = engine.board_equities(board, histogram_bins=histogram_bins)
        classes = enumerate_hand_classes(board)

        cols = np.empty(len(classes), dtype=np.int32)
        equities = np.empty(len(classes), dtype=np.float32)
        multiplicities = np.empty(len(classes), dtype=np.uint8)
        histograms = (
            np.empty((len(classes), histogram_bins), dtype=np.float16)
            if histogram_bins is not None
            else None
        )
        for k, hand_class in enumerate(classes):
            cols[k] = _HAND_ID_TO_COL[hand_class.canonical.hand_id]
            equities[k] = table.equity(hand_class.representative)
            multiplicities[k] = hand_class.multiplicity
            if histograms is not None:
                histograms[k] = table.histogram(hand_class.representative)

        results.append((row, cols, equities, multiplicities, histograms))

    return results


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

    def precompute_street(self, street: Street, board_limit: int | None = None) -> None:
        """
        Precompute buckets for every canonical board on a street.

        Args:
            street: Which street to precompute.
            board_limit: Optional cap on the number of canonical boards
                (lowest board IDs first). Test hook — production runs cover
                every board.
        """
        board_ids, equity_matrix, weight_matrix, hist_matrix = self.compute_street_matrices(
            street, board_limit=board_limit
        )
        print(f"Bucketing {street.name} into {self.config.num_buckets[street]} buckets...")
        self.bucket_street(street, board_ids, equity_matrix, weight_matrix, hist_matrix)

        quality = self._quality[street]
        print(
            f"Completed {street.name}: {quality['class_count']:,} classes "
            f"({quality['combo_count']:,} combos) into {quality['num_buckets']} buckets, "
            f"variance explained {quality['variance_explained']:.4f}"
        )

    def compute_street_matrices(
        self, street: Street, board_limit: int | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        """
        Compute per-board equity/weight (and flop/turn histogram) matrices.

        Exposed separately from bucketing so bucket-count sweeps can reuse one
        expensive equity pass across many bucketing configurations.
        """
        print(f"Enumerating canonical boards for {street.name}...")
        enumerator = CanonicalBoardEnumerator(street)
        enumerator.enumerate()
        board_infos = sorted(enumerator.iterate(), key=lambda info: info.board_id)
        if board_limit is not None:
            board_infos = board_infos[:board_limit]

        n_boards = len(board_infos)
        board_ids = np.array([info.board_id for info in board_infos], dtype=np.int64)
        print(f"Computing exact equities for {n_boards} canonical {street.name} boards...")

        histogram_bins = self.config.equity_histogram_bins if street != Street.RIVER else None

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
                    (chunk, flop_runouts, self.config.seed, histogram_bins),
                )
                for chunk in chunks
            ]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Computing {street.name} equities",
            ):
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

        if hist_matrix is None:
            bucket_flat, actual_buckets = self._bucket_scalar(values, weights, target_buckets)
            quality_extra: dict = {"bucketing": "scalar_equity"}
        else:
            features = self._cdf_features(hist_matrix, valid)
            bucket_flat, actual_buckets, hist_dispersion = self._bucket_histograms(
                features, weights, target_buckets
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
    ) -> tuple[np.ndarray, int, float]:
        """
        K-means over realization-CDF features (flop/turn).

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

        # Mean equity implied by a centroid CDF c: sum of bin probabilities
        # times bin centers, with p = diff([0, c, 1]).
        centers = kmeans.cluster_centers_
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

    def precompute_all(self, streets: list[Street] | None = None) -> DenseBucketer:
        """Precompute all (or the given) postflop streets and return the bucketer."""
        if streets is None:
            streets = list(POSTFLOP_STREETS)

        for street in streets:
            self.precompute_street(street)

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
        with open(path / METADATA_FILENAME, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved abstraction to {path}")

    @classmethod
    def load(cls, path: Path) -> DenseBucketer:
        """Load a precomputed abstraction artifact."""
        return DenseBucketer.load(Path(path))
