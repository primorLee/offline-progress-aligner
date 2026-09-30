"""Endpoint-constrained monotonic DTW; ties prefer diagonal, then up, then left."""
import numpy as np


def hard_dtw(cost):
    cost = np.asarray(cost, dtype=np.float64)
    if cost.ndim != 2 or min(cost.shape) == 0 or not np.isfinite(cost).all():
        raise ValueError("DTW needs a finite, nonempty [N,M] cost matrix")
    n, m = cost.shape
    accumulated = np.full((n + 1, m + 1), np.inf)
    accumulated[0, 0] = 0
    moves = np.zeros((n, m), dtype=np.uint8)
    for i in range(n):
        for j in range(m):
            previous = [accumulated[i, j], accumulated[i, j + 1], accumulated[i + 1, j]]
            k = int(np.argmin(previous))
            moves[i, j] = k
            accumulated[i + 1, j + 1] = cost[i, j] + previous[k]
    i, j = n - 1, m - 1
    path = [(i, j)]
    while i or j:
        k = moves[i, j]
        if k == 0:
            i, j = i - 1, j - 1
        elif k == 1:
            i -= 1
        else:
            j -= 1
        path.append((i, j))
    path.reverse()
    sums = np.zeros(n)
    counts = np.zeros(n)
    for i, j in path:
        sums[i] += j
        counts[i] += 1
    return sums / counts, path, float(np.mean([cost[i, j] for i, j in path]))


def alignment(a, b):
    similarity = np.clip(a @ b.T, -1, 1)
    mapping, path, cost = hard_dtw(1 - similarity)
    return mapping, similarity, path, cost
