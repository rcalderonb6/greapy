"""General-purpose numerical utilities."""

import numpy as np


def is_monotonic_increasing(a, strict=False):
    """Check if an array is monotonically increasing.

    Uses ``numpy.ismonotonic`` on NumPy 2.0+ and falls back to
    ``numpy.diff`` on older versions.

    Args:
        a: Input array to check.
        strict: If True, require strictly increasing (each element greater
            than the previous). Default is False (non-decreasing).

    Returns:
        True if the array satisfies the monotonicity condition.
    """
    a = np.asarray(a)

    if a.size <= 1:
        return True

    if hasattr(np, "ismonotonic"):
        return np.ismonotonic(a, increasing=True, strict=strict)
    else:
        return bool(np.all(np.diff(a) > 0) if strict else np.all(np.diff(a) >= 0))
