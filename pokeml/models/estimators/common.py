import numpy as np


def _normalize_eval_set(eval_set):
    if eval_set is None:
        return None

    if isinstance(eval_set, tuple) and len(eval_set) == 2:
        return [eval_set]

    if isinstance(eval_set, list):
        return eval_set

    raise ValueError(
        "eval_set must be None, a tuple (X, y), or a list of tuples [(X1, y1), ...]."
    )


def validate_iterations(iterations, total):
    """Require ordered, distinct tree-prefix lengths within fitted model."""
    values = list(iterations)
    if not values or any(not isinstance(i, (int, np.integer)) or isinstance(i, bool) for i in values):
        raise ValueError("Iterations must be nonempty integers")
    if values != sorted(set(values)) or values[0] < 1 or values[-1] > total:
        raise ValueError("Iterations must increase uniquely within fitted tree count")
    return values
