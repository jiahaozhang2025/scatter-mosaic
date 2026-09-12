"""Post-hoc compaction of a UMAP layout.

UMAP spends a lot of canvas on empty space between clusters, which leaves each
portrait small. This slides whole clusters inward until they nearly touch,
keeping every cluster's internal shape untouched and holding a guaranteed gap
between neighbours.

The transform is rigid per cluster: it changes where clusters sit relative to
one another, not the local structure UMAP actually encodes. It is a layout
choice for the artwork, not a claim about the data.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, binary_fill_holes, label


def normalize_isotropic(
    x: np.ndarray, y: np.ndarray, pad: float = 0.04
) -> tuple[np.ndarray, np.ndarray]:
    """Map a layout into [0, 1] squared with y pointing down, equal scale on both axes.

    Scaling each axis to its own range would stretch the cloud to whatever aspect
    the canvas happens to be, which turns a square face patch into a rectangle and
    makes the poster and the viewer disagree. One scale for both, content centred.
    """
    span = max(float(np.ptp(x)), float(np.ptp(y)), 1e-9)
    mid_x = (float(x.min()) + float(x.max())) / 2
    mid_y = (float(y.min()) + float(y.max())) / 2
    nx = 0.5 + (np.asarray(x, dtype=np.float64) - mid_x) / span
    ny = 0.5 - (np.asarray(y, dtype=np.float64) - mid_y) / span
    return pad + (1 - 2 * pad) * nx, pad + (1 - 2 * pad) * ny


def _disc(radius: int) -> np.ndarray:
    if radius <= 0:
        return np.ones((1, 1), dtype=bool)
    size = 2 * radius + 1
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    return (yy * yy + xx * xx) <= radius * radius + 0.5


class _Tile:
    """A cluster's occupancy footprint plus where it currently sits on the grid."""

    __slots__ = ("mask", "row", "col", "row0", "col0")

    def __init__(self, mask: np.ndarray, row: int, col: int):
        self.mask = mask
        self.row = self.row0 = row
        self.col = self.col0 = col


def _stamp(occupancy: np.ndarray, tile: _Tile, row: int, col: int, delta: int) -> None:
    h, w = tile.mask.shape
    occupancy[row:row + h, col:col + w] += tile.mask * delta


def _free(occupancy: np.ndarray, tile: _Tile, row: int, col: int) -> bool:
    h, w = tile.mask.shape
    if row < 0 or col < 0 or row + h > occupancy.shape[0] or col + w > occupancy.shape[1]:
        return False
    return not np.any(occupancy[row:row + h, col:col + w][tile.mask])


def compact_clusters(
    x: np.ndarray,
    y: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
    strength: float = 1.0,
    gap: float = 0.012,
    grid: int = 420,
    iterations: int = 220,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Slide each cluster toward the layout centre until it nearly touches its neighbours.

    `gap` is the minimum clearance between clusters as a fraction of the layout
    diagonal. `strength` scales the resulting displacement, so 0 leaves the
    layout untouched and 1 applies the full packing. Returns new coordinates
    plus a small report.
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0 or n_clusters < 2:
        return x.astype(np.float64), y.astype(np.float64), {"moved": 0.0, "shrink": 1.0}

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    span_x, span_y = max(np.ptp(x), 1e-9), max(np.ptp(y), 1e-9)
    # A square-ish grid cell in data units keeps the packing isotropic.
    cell = max(span_x, span_y) / grid
    pad_px = max(1, int(round(gap * np.hypot(span_x, span_y) / cell / 2)))
    margin = pad_px + 4

    cols = int(span_x / cell) + 2 * margin + 2
    rows = int(span_y / cell) + 2 * margin + 2
    col_of = ((x - x.min()) / cell).astype(np.int32) + margin
    row_of = ((y - y.min()) / cell).astype(np.int32) + margin

    footprint = _disc(pad_px)
    tiles: dict[int, _Tile] = {}
    for cluster in range(n_clusters):
        hit = labels == cluster
        if not hit.any():
            continue
        plate = np.zeros((rows, cols), dtype=bool)
        plate[row_of[hit], col_of[hit]] = True
        plate = binary_fill_holes(binary_dilation(plate, footprint))
        # Stray outliers would otherwise block the whole cluster from moving.
        parts, count = label(plate)
        if count > 1:
            sizes = np.bincount(parts.ravel())
            sizes[0] = 0
            plate = np.isin(parts, np.flatnonzero(sizes >= sizes.max() * 0.02))
        ys, xs = np.where(plate)
        r0, c0 = int(ys.min()), int(xs.min())
        tiles[cluster] = _Tile(plate[r0:ys.max() + 1, c0:xs.max() + 1], r0, c0)

    occupancy = np.zeros((rows, cols), dtype=np.int16)
    for tile in tiles.values():
        _stamp(occupancy, tile, tile.row, tile.col, 1)

    angles = np.deg2rad([0, 20, -20, 42, -42, 64, -64, 85, -85])
    weight = {c: float(tile.mask.sum()) for c, tile in tiles.items()}
    # Each cluster carries its own step: halved whenever it is boxed in, so big
    # moves happen early and stragglers still creep the last few pixels.
    step = {c: max(2.0, grid / 40) for c in tiles}

    def centre(tile: _Tile) -> tuple[float, float]:
        return tile.row + tile.mask.shape[0] / 2, tile.col + tile.mask.shape[1] / 2

    for _ in range(iterations):
        total = sum(weight.values())
        target_r = sum(weight[c] * centre(tiles[c])[0] for c in tiles) / total
        target_c = sum(weight[c] * centre(tiles[c])[1] for c in tiles) / total
        order = sorted(tiles, key=lambda c: -np.hypot(*np.subtract(centre(tiles[c]), (target_r, target_c))))
        moved = False
        for cluster in order:
            tile = tiles[cluster]
            cy, cx = centre(tile)
            dr, dc = target_r - cy, target_c - cx
            distance = np.hypot(dr, dc)
            if distance < 0.5:
                continue
            base = np.arctan2(dr, dc)
            reach = min(step[cluster], distance)
            _stamp(occupancy, tile, tile.row, tile.col, -1)
            landed = False
            for angle in angles:
                row = int(round(tile.row + reach * np.sin(base + angle)))
                col = int(round(tile.col + reach * np.cos(base + angle)))
                if (row, col) != (tile.row, tile.col) and _free(occupancy, tile, row, col):
                    tile.row, tile.col = row, col
                    landed = moved = True
                    break
            _stamp(occupancy, tile, tile.row, tile.col, 1)
            step[cluster] = min(step[cluster] * 1.3, grid / 40) if landed else max(1.0, step[cluster] / 2)
        if not moved and all(s <= 1.0 for s in step.values()):
            break

    out_x = x.copy()
    out_y = y.copy()
    shifts = []
    for cluster, tile in tiles.items():
        dx = (tile.col - tile.col0) * cell * strength
        dy = (tile.row - tile.row0) * cell * strength
        hit = labels == cluster
        out_x[hit] += dx
        out_y[hit] += dy
        shifts.append(np.hypot(dx, dy))

    before = span_x * span_y
    after = max(np.ptp(out_x), 1e-9) * max(np.ptp(out_y), 1e-9)
    return out_x, out_y, {
        "moved": float(np.mean(shifts)) if shifts else 0.0,
        "shrink": float(np.sqrt(after / before)),
        "gap_px": pad_px * 2,
    }
