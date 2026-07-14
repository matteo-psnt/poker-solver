"""
Abstraction quality metrics.

Quantifies how much strategic information the bucketing preserves, computed
from the exact per-class equities at precompute time (combo-weighted):

- ``equity_std``: spread of equity across all combos on the street. This is
  the information available to be preserved.
- ``within_bucket_std``: root-mean-square equity spread inside buckets. This
  is the information the abstraction destroys — hands this far apart in
  equity are forced to share a strategy.
- ``variance_explained``: 1 - within-variance / total variance (weighted R²).
  1.0 means buckets separate equity perfectly; 0.0 means they carry no
  equity information.
"""

from __future__ import annotations

import numpy as np


def compute_street_quality(
    equities: np.ndarray,
    buckets: np.ndarray,
    weights: np.ndarray,
    num_buckets: int,
) -> dict:
    """
    Compute quality metrics for one street.

    Args:
        equities: Flat array of per-class equities.
        buckets: Flat array of bucket assignments (same length).
        weights: Flat array of class multiplicities (combo counts).
        num_buckets: Total buckets on the street.

    Returns:
        JSON-serializable metrics dict.
    """
    weights = weights.astype(np.float64)
    total_weight = float(weights.sum())

    mean = float(np.average(equities, weights=weights))
    total_var = float(np.average((equities - mean) ** 2, weights=weights))

    bucket_weight = np.bincount(buckets, weights=weights, minlength=num_buckets)
    bucket_eq_sum = np.bincount(buckets, weights=weights * equities, minlength=num_buckets)
    bucket_eq_sq_sum = np.bincount(buckets, weights=weights * equities**2, minlength=num_buckets)

    occupied = bucket_weight > 0
    bucket_mean = np.zeros(num_buckets)
    bucket_mean[occupied] = bucket_eq_sum[occupied] / bucket_weight[occupied]
    bucket_var = np.zeros(num_buckets)
    bucket_var[occupied] = (
        bucket_eq_sq_sum[occupied] / bucket_weight[occupied] - bucket_mean[occupied] ** 2
    )
    # Clamp tiny negative values from floating-point cancellation.
    bucket_var = np.maximum(bucket_var, 0.0)

    within_var = float((bucket_weight * bucket_var).sum() / total_weight)
    variance_explained = 1.0 - within_var / total_var if total_var > 0 else 1.0

    occupied_weights = bucket_weight[occupied]
    return {
        "num_buckets": int(num_buckets),
        "occupied_buckets": int(occupied.sum()),
        "combo_count": int(total_weight),
        "class_count": len(equities),
        "equity_std": round(float(np.sqrt(total_var)), 6),
        "within_bucket_std": round(float(np.sqrt(within_var)), 6),
        "variance_explained": round(variance_explained, 6),
        "bucket_combos_min": int(occupied_weights.min()) if occupied.any() else 0,
        "bucket_combos_median": float(np.median(occupied_weights)) if occupied.any() else 0.0,
        "bucket_combos_max": int(occupied_weights.max()) if occupied.any() else 0,
    }
