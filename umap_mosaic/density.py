"""Simulate extra points from the shape a scatter already has.

A real scatter is often too thin to paint. Twenty-five thousand earthquakes trace
the plate boundaries in a line one or two points wide, and a picture drawn on that
has almost no dots to be drawn with. The fix is not to invent structure but to
sample more densely from the structure that is there: pick an existing point, and
put a new one a short distance away.

That is kernel density estimation used backwards — sampling from the KDE rather
than evaluating it — and the only real choice is the bandwidth. A fixed one blurs
tight filaments while barely filling sparse ground, so the jitter here is scaled
per point by the distance to its own k-th neighbour: a dot in a dense clump moves
a little, a dot on a sparse fringe moves more, and the outline stays put.

The points this returns are **not data**. They carry a flag saying so, the CSV
keeps that flag, and nothing should be measured from them — they exist to give the
painter enough dots to work with.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def local_spacing(points: np.ndarray, neighbours: int = 6) -> np.ndarray:
    """Distance from each point to its k-th nearest neighbour."""
    k = min(neighbours, len(points) - 1)
    if k < 1:
        return np.full(len(points), 1.0)
    distances, _ = cKDTree(points).query(points, k=k + 1, workers=-1)
    spacing = distances[:, -1]
    # A duplicated coordinate gives a zero radius, which would freeze its copies.
    typical = float(np.median(spacing[spacing > 0])) if (spacing > 0).any() else 1.0
    return np.where(spacing > 0, spacing, typical)


def densify(
    x: np.ndarray,
    y: np.ndarray,
    factor: float,
    seed: int = 42,
    neighbours: int = 6,
    spread: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return x, y and a boolean `simulated` flag, with `factor`x as many points.

    `factor` 1 is a no-op. `spread` scales the jitter against the local spacing;
    much above 1 and thin structure starts to smear into its surroundings.
    """
    real = np.column_stack([np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)])
    wanted = int(round(len(real) * (factor - 1)))
    if wanted <= 0 or len(real) < 2:
        return real[:, 0], real[:, 1], np.zeros(len(real), dtype=bool)

    radius = local_spacing(real, neighbours) * spread
    rng = np.random.default_rng(seed)
    parents = rng.integers(0, len(real), wanted)
    made = real[parents] + rng.normal(size=(wanted, 2)) * radius[parents, None]

    points = np.vstack([real, made])
    simulated = np.zeros(len(points), dtype=bool)
    simulated[len(real):] = True
    return points[:, 0], points[:, 1], simulated
