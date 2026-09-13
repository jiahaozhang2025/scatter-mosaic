"""Pack every face onto the best ground the scatter has, without clustering anything.

Two earlier approaches both went wrong in the same place. Fitting a portrait to each
cluster silhouette let the cluster's shape and size decide the face, so ragged
clusters mangled it and small ones dropped their member entirely. Spreading equal
squares out by farthest-point fixed that, but went too far the other way: each face
sat alone in the middle of its own territory, so a cluster big enough for three
portraits still got one, and the leftovers were pushed onto thin ground where they
are barely visible.

This packs instead. Squares are scored on how well the cloud supports them, taken
best-first, and a square that can sit flush against one already placed gets a bonus,
so faces tile into the roomy parts of the map. A group stops accepting neighbours
once it reaches `group_cap`, which keeps one dense corner from swallowing the whole
roster while still letting a large region hold several portraits.

The spots are then handed out by proximity — each cell belongs to the nearest face —
so the regions come from the faces rather than the faces coming from a clustering.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_closing, binary_dilation


def _integral(plate: np.ndarray) -> np.ndarray:
    return np.pad(plate.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))


def _windows(table: np.ndarray, side: int) -> np.ndarray:
    """Sum inside every `side x side` window, from a summed-area table."""
    return (
        table[side:, side:] - table[:-side, side:] - table[side:, :-side] + table[:-side, :-side]
    )


# Progressively looser ground rules, used only when the strict ones cannot seat
# everybody. The last is what lets an outlying wisp host a face rather than losing
# its member altogether.
STAGES = ((1.00, 1.00), (0.92, 0.55), (0.80, 0.22))


def _pack(
    rows: np.ndarray,
    cols: np.ndarray,
    support: np.ndarray,
    side: int,
    pad: int,
    count: int,
    adjacency: float,
    group_cap: int,
) -> tuple[list[int], list[int]]:
    """Greedy best-first packing with a bonus for sitting flush against a neighbour.

    Equal axis-aligned squares overlap exactly when their centres are within one
    step on both axes, so a placement can knock out its neighbours with two
    comparisons rather than stamping a grid.
    """
    step = side + pad
    tolerance = max(1.0, side * 0.14)
    cx = cols.astype(np.float64) + side / 2
    cy = rows.astype(np.float64) + side / 2

    alive = np.ones(len(cx), dtype=bool)
    picked: list[int] = []
    group_of: list[int] = []
    group_size: dict[int, int] = {}

    while len(picked) < count and alive.any():
        bonus = np.zeros(len(cx))
        joins = np.full(len(cx), -1, dtype=np.int32)
        for slot, placed in enumerate(picked):
            if group_size[group_of[slot]] >= group_cap:
                continue
            dx = np.abs(cx - cx[placed])
            dy = np.abs(cy - cy[placed])
            flush = (
                ((np.abs(dx - step) <= tolerance) & (dy <= tolerance))
                | ((np.abs(dy - step) <= tolerance) & (dx <= tolerance))
            )
            fresh = flush & (joins < 0)
            bonus[fresh] = 1.0
            joins[fresh] = slot

        index = int(np.argmax(np.where(alive, support + adjacency * bonus, -np.inf)))
        group = group_of[joins[index]] if joins[index] >= 0 else len(group_size)
        group_size[group] = group_size.get(group, 0) + 1
        group_of.append(group)
        picked.append(index)
        alive &= ~((np.abs(cx - cx[index]) < step) & (np.abs(cy - cy[index]) < step))
    return picked, group_of


def allocate_faces(
    x: np.ndarray,
    y: np.ndarray,
    count: int,
    grid: int = 256,
    size: float = 0.055,
    min_fill: float = 0.85,
    min_density: float = 0.55,
    adjacency: float = 0.6,
    group_cap: int = 4,
    margin: float = 0.0,
    shrink: float = 0.88,
    min_size: float = 0.016,
    seed: int = 42,
) -> tuple[list[dict], float]:
    """`count` equal, non-overlapping squares packed onto the cloud, one per face.

    `x` and `y` are normalized to [0, 1] with y pointing down; `size` is the square
    edge in the same units. Two tests say whether a square sits on usable ground:
    `min_fill` is the share of it the cloud covers (rejects holes) and
    `min_density` rejects thin haze.

    `min_density` is *relative*: it is a fraction of the typical density of the
    covered parts of this cloud. An absolute count would be meaningless across
    datasets — the same threshold that separates core from fringe in a 40,000-point
    embedding rejects every position in a 2,000-point one.

    `adjacency` is how much a square wants to sit flush against one already placed,
    and `group_cap` is how many may end up flush together. If `count` squares will
    not fit at `size`, the size drops by `shrink` and the search repeats, so the
    answer is the largest equal size at which every face gets a home. Returns the
    squares in left-to-right order plus the size used.
    """
    rng = np.random.default_rng(seed)
    px = np.clip((x * (grid - 1)).astype(int), 0, grid - 1)
    py = np.clip((y * (grid - 1)).astype(int), 0, grid - 1)
    total = np.zeros((grid, grid), dtype=np.int32)
    np.add.at(total, (py, px), 1)

    # Close pinholes between neighbouring cells so "solid" means the region is
    # covered, not that every single pixel happened to catch a point.
    solid = binary_closing(binary_dilation(total > 0, np.ones((3, 3))), np.ones((3, 3)), iterations=2)
    solid_table, total_table = _integral(solid), _integral(total)
    pad = int(round(margin * grid))

    best: list[dict] = []
    best_size = 0.0

    # Stage first, size second. A smaller face on ground that genuinely supports it
    # beats a larger one sitting on haze, which is what the other nesting produced:
    # the search would accept a relaxed stage before it ever tried shrinking.
    for fill_scale, density_scale in STAGES:
        current = float(size)
        while current >= min_size:
            side = int(round(current * grid))
            if side < 4:
                break
            fill = _windows(solid_table, side) / float(side * side)
            cells = _windows(total_table, side)
            density = cells / float(side * side)
            covered = fill >= min_fill * fill_scale
            if not covered.any():
                current *= shrink
                continue
            reference = float(np.median(density[covered])) or 1e-9
            eligible = np.argwhere(covered & (density >= min_density * density_scale * reference))
            if len(eligible):
                rows, cols = eligible[:, 0], eligible[:, 1]
                reach = density[rows, cols] / max(float(density[rows, cols].max()), 1e-9)
                support = 0.75 * reach + 0.25 * fill[rows, cols] + rng.uniform(0, 0.01, len(rows))
                picked, groups = _pack(rows, cols, support, side, pad, count, adjacency, group_cap)
                if len(picked) > len(best):
                    best = [
                        {
                            "box": [
                                cols[i] / grid, rows[i] / grid,
                                (cols[i] + side) / grid, (rows[i] + side) / grid,
                            ],
                            "size": round(side / grid, 4),
                            "group": int(groups[slot]),
                            "fill": round(float(fill[rows[i], cols[i]]), 3),
                            "density": round(float(density[rows[i], cols[i]]), 2),
                            "cells": int(cells[rows[i], cols[i]]),
                        }
                        for slot, i in enumerate(picked)
                    ]
                    best_size = side / grid
                if len(picked) >= count:
                    break
            current *= shrink
        if len(best) >= count:
            break

    # Left to right, then top to bottom, so the numbering reads off the artwork.
    best.sort(key=lambda patch: (patch["box"][0], patch["box"][1]))
    for slot, patch in enumerate(best):
        patch["region"] = slot
    return best, best_size


def assign_regions(x: np.ndarray, y: np.ndarray, patches: list[dict]) -> np.ndarray:
    """Give every cell to the nearest face square — the spots follow the faces.

    This replaces clustering outright: the regions exist because the faces do, so
    no face can be squeezed out by a region that turned out too small to hold it.
    """
    if not patches:
        return np.zeros(len(x), dtype=np.int16)
    centres = np.array(
        [[(p["box"][0] + p["box"][2]) / 2, (p["box"][1] + p["box"][3]) / 2] for p in patches]
    )
    dx = x[:, None] - centres[None, :, 0]
    dy = y[:, None] - centres[None, :, 1]
    return np.argmin(dx * dx + dy * dy, axis=1).astype(np.int16)
