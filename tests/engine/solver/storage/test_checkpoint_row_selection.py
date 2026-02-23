"""Row-selected checkpoint reads must return exactly the requested rows, in order.

Workers own ~1/N of the rows and now read only those from zarr, because loading
the full arrays into every worker cost N x the whole checkpoint in private memory
(~1.9 GB each at 18.9M keys, ~30 GB across 16) for data discarded immediately.

The rows come back *positionally* -- result row k is row_ids[k] -- and the caller
writes them straight into shared memory at its own new ids. A misordering or an
off-by-one would therefore not fail loudly; it would silently attach one
infoset's learned values to a different infoset. Hence comparing against a full
read rather than merely checking shapes.
"""

from pathlib import Path

import numpy as np
import pytest

from src.engine.solver.storage.array_specs import ARRAY_SPECS
from src.engine.solver.storage.helpers import load_checkpoint_arrays, load_checkpoint_rows

GOLDEN_RUN = Path(__file__).parents[3] / "fixtures" / "golden_run"


@pytest.fixture(scope="module")
def full_arrays():
    return load_checkpoint_arrays(GOLDEN_RUN)


@pytest.mark.parametrize("num_workers", [2, 4, 16])
def test_selected_rows_match_the_full_arrays(full_arrays, num_workers):
    """Every shard's rows must equal the same rows of a full read."""
    total = full_arrays["action_counts"].shape[0]
    for worker_id in range(num_workers):
        row_ids = np.arange(worker_id, total, num_workers, dtype=np.int32)
        selected, max_actions = load_checkpoint_rows(GOLDEN_RUN, row_ids)

        assert max_actions == full_arrays["regrets"].shape[1]
        for spec in ARRAY_SPECS:
            expected = full_arrays[spec.checkpoint_key][row_ids]
            np.testing.assert_array_equal(
                selected[spec.checkpoint_key],
                expected,
                err_msg=f"{spec.checkpoint_key} misaligned for worker {worker_id}",
            )


def test_scattered_selection_preserves_order(full_arrays):
    """Ownership is hash-scattered, so rows are sparse and non-contiguous."""
    row_ids = np.array([0, 7, 8, 91, 512, 1000], dtype=np.int32)
    selected, _ = load_checkpoint_rows(GOLDEN_RUN, row_ids)

    for spec in ARRAY_SPECS:
        for k, row in enumerate(row_ids.tolist()):
            np.testing.assert_array_equal(
                selected[spec.checkpoint_key][k],
                full_arrays[spec.checkpoint_key][row],
                err_msg=f"{spec.checkpoint_key} row {k} is not source row {row}",
            )


def test_single_row_selection(full_arrays):
    selected, _ = load_checkpoint_rows(GOLDEN_RUN, np.array([3], dtype=np.int32))
    np.testing.assert_array_equal(selected["regrets"][0], full_arrays["regrets"][3])
