"""Shared numeric guards."""

# Smallest denominator worth dividing by when normalizing a non-negative vector
# (regrets, strategy mass, combo weights) into a distribution. At or below this,
# callers treat the total as no mass and fall back to uniform.
#
# Strictly a divide-by-zero guard: it decides whether a division is meaningful,
# never replaces a value. Code that needs values floored above zero wants its own
# constant — the two can diverge (see `_LIKELIHOOD_FLOOR` in range_inference).
NORMALIZE_EPS = 1e-12
