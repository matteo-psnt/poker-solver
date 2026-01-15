"""
Full-coverage precomputation pipeline for combo-level abstraction.

For every canonical board on every street:
1. Compute exact equity for every canonical hand class (range-vs-range engine)
2. Bucket all (board, class) equities per street with weighted 1D k-means
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

# Equity quantization grid for the weighted 1D k-means fit. 2^16 bins keep the
# fit exact to ~1.5e-5 equity while bounding its input size.
_KMEANS_EQUITY_BINS = 65536

_MAX_CHUNK_BOARDS = 512


def _worker_compute_board_chunk(
    args: tuple[list[tuple[int, tuple[Card, ...]]], int | None, int],
) -> list[tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Compute per-class equities for a chunk of boards.

    Returns one (row, columns, equities, multiplicities) tuple per board.
    """
    boards, flop_runouts, seed = args
    engine = RangeEquityEngine(max_runouts=flop_runouts, seed=seed)

    results = []
    for row, board in boards:
        table = engine.board_equities(board)
        classes = enumerate_hand_classes(board)

        cols = np.empty(len(classes), dtype=np.int32)
        equities = np.empty(len(classes), dtype=np.float32)
        multiplicities = np.empty(len(classes), dtype=np.uint8)
        for k, hand_class in enumerate(classes):
            cols[k] = _HAND_ID_TO_COL[hand_class.canonical.hand_id]
            equities[k] = table.equity(hand_class.representative)
            multiplicities[k] = hand_class.multiplicity

        results.append((row, cols, equities, multiplicities))

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
        print(f"Starting precomputation for {street.name}")

        print("Step 1: Enumerating canonical boards...")
        enumerator = CanonicalBoardEnumerator(street)
        enumerator.enumerate()
        board_infos = sorted(enumerator.iterate(), key=lambda info: info.board_id)
        if board_limit is not None:
            board_infos = board_infos[:board_limit]

        n_boards = len(board_infos)
        board_ids = np.array([info.board_id for info in board_infos], dtype=np.int64)
        print(f"Covering {n_boards} canonical boards for {street.name}")

        print("Step 2: Computing exact equities for every board...")
        equity_matrix = np.full((n_boards, N_HAND_COLUMNS), np.nan, dtype=np.float32)
        weight_matrix = np.zeros((n_boards, N_HAND_COLUMNS), dtype=np.uint8)

        num_workers = self.config.num_workers or mp.cpu_count()
        flop_runouts = self.config.flop_runouts if street == Street.FLOP else None

        boards = [(row, info.representative) for row, info in enumerate(board_infos)]
        chunk_size = min(_MAX_CHUNK_BOARDS, max(1, n_boards // (num_workers * 8)))
        chunks = [boards[i : i + chunk_size] for i in range(0, n_boards, chunk_size)]

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    _worker_compute_board_chunk, (chunk, flop_runouts, self.config.seed)
                )
                for chunk in chunks
            ]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Computing {street.name} equities",
            ):
                for row, cols, equities, multiplicities in future.result():
                    equity_matrix[row, cols] = equities
                    weight_matrix[row, cols] = multiplicities

        print(f"Step 3: Bucketing into {self.config.num_buckets[street]} buckets...")
        self._bucket_street(street, board_ids, equity_matrix, weight_matrix)

        quality = self._quality[street]
        print(
            f"Completed {street.name}: {quality['class_count']:,} classes "
            f"({quality['combo_count']:,} combos) into {quality['num_buckets']} buckets, "
            f"variance explained {quality['variance_explained']:.4f}"
        )

    def _bucket_street(
        self,
        street: Street,
        board_ids: np.ndarray,
        equity_matrix: np.ndarray,
        weight_matrix: np.ndarray,
    ) -> None:
        """Weighted 1D k-means over all (board, class) equities on a street."""
        valid = ~np.isnan(equity_matrix)
        values = equity_matrix[valid].astype(np.float64)
        weights = weight_matrix[valid].astype(np.float64)

        if values.size == 0:
            raise ValueError(f"No equity data computed for {street.name}")

        # Weighted k-means on the quantized equity histogram: exact up to the
        # grid resolution, independent of how many (board, class) pairs exist.
        quantized = np.clip(
            (values * (_KMEANS_EQUITY_BINS - 1)).astype(np.int64), 0, _KMEANS_EQUITY_BINS - 1
        )
        histogram = np.bincount(quantized, weights=weights, minlength=_KMEANS_EQUITY_BINS)
        occupied_bins = np.nonzero(histogram)[0]
        points = (occupied_bins / (_KMEANS_EQUITY_BINS - 1)).reshape(-1, 1)

        kmeans = KMeans(
            n_clusters=min(self.config.num_buckets[street], len(occupied_bins)),
            max_iter=self.config.kmeans_max_iter,
            n_init=self.config.kmeans_n_init,
            random_state=self.config.seed,
        )
        kmeans.fit(points, sample_weight=histogram[occupied_bins])

        # Bucket 0 = lowest equity; assignment by nearest center = boundary search.
        centers = np.unique(kmeans.cluster_centers_.ravel())
        boundaries = (centers[1:] + centers[:-1]) / 2
        num_buckets = len(centers)

        dtype = bucket_dtype(num_buckets)
        matrix = np.full(equity_matrix.shape, np.iinfo(dtype).max, dtype=dtype)
        bucket_flat = np.searchsorted(boundaries, values)
        matrix[valid] = bucket_flat

        self._board_ids[street] = board_ids
        self._bucket_matrices[street] = matrix
        self._num_buckets[street] = num_buckets
        self._quality[street] = compute_street_quality(
            equities=values,
            buckets=bucket_flat,
            weights=weights,
            num_buckets=num_buckets,
        )

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
