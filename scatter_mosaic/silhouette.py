"""Cluster silhouettes and the warp that fits an image onto one.

Used only by the `cluster` placement, which traces a smooth shape around each
group of points and bends an image onto it. The `scatter` placement needs none
of this — it uses plain squares.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from scipy.ndimage import (
    binary_closing,
    binary_fill_holes,
    gaussian_filter,
    gaussian_filter1d,
    label,
)


def remove_tiny_components(mask: np.ndarray, minimum: int = 18) -> np.ndarray:
    components, count = label(mask)
    if count == 0:
        return mask
    sizes = np.bincount(components.ravel())
    keep = sizes >= minimum
    keep[0] = False
    return keep[components]


def make_masks(
    x: np.ndarray,
    y: np.ndarray,
    clusters: np.ndarray,
    n_clusters: int,
    grid_w: int,
    grid_h: int,
    sigma: float | None = None,
) -> list[np.ndarray]:
    """One filled, non-overlapping silhouette per cluster on a `grid_h x grid_w` raster."""
    if sigma is None:
        sigma = max(3.0, 5.0 * grid_w / 1100.0)
    px = np.clip((x * (grid_w - 1)).astype(int), 0, grid_w - 1)
    py = np.clip((y * (grid_h - 1)).astype(int), 0, grid_h - 1)
    densities = np.zeros((n_clusters, grid_h, grid_w), dtype=np.float32)
    support = np.zeros_like(densities, dtype=bool)
    for cluster in range(n_clusters):
        hit = clusters == cluster
        histogram = np.zeros((grid_h, grid_w), dtype=np.float32)
        np.add.at(histogram, (py[hit], px[hit]), 1)
        density = gaussian_filter(histogram, sigma=sigma)
        scale = np.quantile(density[density > 0], .985) if np.any(density > 0) else 1.0
        densities[cluster] = density / max(scale, 1e-6)
        local = density > max(density.max() * .012, 0.00008)
        support[cluster] = binary_fill_holes(binary_closing(local, iterations=2))

    score = np.where(support, densities, -1)
    owner = score.argmax(axis=0)
    occupied = score.max(axis=0) >= 0
    masks = []
    for cluster in range(n_clusters):
        mask = remove_tiny_components((owner == cluster) & occupied)
        mask = binary_fill_holes(binary_closing(mask, iterations=2))
        masks.append(mask)
    return masks


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _bilinear(source: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    height, width = source.shape[:2]
    fx = np.clip(u, 0.0, 1.0) * (width - 1)
    fy = np.clip(v, 0.0, 1.0) * (height - 1)
    x0 = np.floor(fx).astype(np.int32)
    y0 = np.floor(fy).astype(np.int32)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    ax = (fx - x0)[..., None]
    ay = (fy - y0)[..., None]
    top = source[y0, x0] * (1 - ax) + source[y0, x1] * ax
    bottom = source[y1, x0] * (1 - ax) + source[y1, x1] * ax
    return top * (1 - ay) + bottom * ay


def _cover_span(width: int, height: int, focus: tuple[float, float]) -> tuple[float, float, float, float]:
    """Where a square portrait lands when it is scaled to cover a `width x height` box."""
    aspect = width / max(height, 1)
    span_u = 1.0 if aspect >= 1 else aspect
    span_v = 1.0 / aspect if aspect >= 1 else 1.0
    u0 = float(np.clip(focus[0] - span_u / 2, 0.0, 1.0 - span_u))
    v0 = float(np.clip(focus[1] - span_v / 2, 0.0, 1.0 - span_v))
    return u0, span_u, v0, span_v


def _axis_warp(
    lower: np.ndarray, upper: np.ndarray, extent: int, morph: float, limit: float
) -> tuple[np.ndarray, np.ndarray]:
    """Per-slice centre and half-width that ease from the whole box toward the silhouette."""
    half = extent / 2.0
    span = np.maximum((upper - lower) / 2.0, 1.0)
    centre = (upper + lower) / 2.0
    # Geometric easing keeps a narrow slice from collapsing the face to a sliver.
    eased = half ** (1 - morph) * span ** morph
    eased = np.clip(eased, half / limit, half * limit)
    return (1 - morph) * half + morph * centre, eased


def morph_into_mask(
    portrait: Image.Image,
    mask: Image.Image,
    morph: float = 0.45,
    focus: tuple[float, float] = (0.5, 0.46),
    limit: float = 1.9,
) -> Image.Image:
    """Warp `portrait` so it leans into the shape in `mask`, returned as RGBA.

    The warp bends the portrait along the silhouette's spine and lets it bulge or
    narrow with the shape, but the per-slice scale is clamped to `limit` so a long
    thin cluster stretches a face instead of smearing it into an unreadable band.
    `morph` 0 is a plain center-crop, 1 pushes every slice as far as the clamp allows.
    Alpha comes from the mask, so the result drops onto the spot layer with soft edges.
    """
    alpha = np.asarray(mask.convert("L"), dtype=np.float32) / 255.0
    height, width = alpha.shape
    solid = alpha > 0.5
    if not solid.any():
        solid = alpha > 0.0

    columns = np.arange(width, dtype=np.float32)
    rows = np.arange(height, dtype=np.float32)
    row_has = solid.any(axis=1)
    col_has = solid.any(axis=0)

    left = np.where(solid, columns[None, :], np.inf).min(axis=1)
    right = np.where(solid, columns[None, :], -np.inf).max(axis=1)
    top = np.where(solid, rows[:, None], np.inf).min(axis=0)
    bottom = np.where(solid, rows[:, None], -np.inf).max(axis=0)
    left = np.where(row_has, left, 0.0).astype(np.float32)
    right = np.where(row_has, right, width - 1.0).astype(np.float32)
    top = np.where(col_has, top, 0.0).astype(np.float32)
    bottom = np.where(col_has, bottom, height - 1.0).astype(np.float32)

    # Smooth the extents so a ragged silhouette edge does not shear the face.
    left = gaussian_filter1d(left, sigma=max(2.0, height / 22), mode="nearest")
    right = gaussian_filter1d(right, sigma=max(2.0, height / 22), mode="nearest")
    top = gaussian_filter1d(top, sigma=max(2.0, width / 22), mode="nearest")
    bottom = gaussian_filter1d(bottom, sigma=max(2.0, width / 22), mode="nearest")

    factor = float(np.clip(morph, 0.0, 1.0))
    clamp = 1.0 + (max(limit, 1.0) - 1.0) * factor
    cx, hx = _axis_warp(left, right, width, factor, clamp)
    cy, hy = _axis_warp(top, bottom, height, factor, clamp)

    # Undo the warp: ask where each output pixel sits inside the straightened box.
    straight_x = width / 2 + (columns[None, :] - cx[:, None]) * (width / 2) / hx[:, None]
    straight_y = height / 2 + (rows[:, None] - cy[None, :]) * (height / 2) / hy[None, :]

    u0, span_u, v0, span_v = _cover_span(width, height, focus)
    u = u0 + np.clip(straight_x / max(width - 1, 1), 0.0, 1.0) * span_u
    v = v0 + np.clip(straight_y / max(height - 1, 1), 0.0, 1.0) * span_v

    side = int(min(1400, max(256, max(width, height))))
    source = np.asarray(
        portrait.resize((side, side), Image.Resampling.LANCZOS), dtype=np.float32
    )
    warped = _bilinear(source, u, v)
    if warped.shape[-1] == 4:
        # A cut-out portrait carries its own alpha; the silhouette only bounds it.
        alpha = alpha * np.clip(warped[..., 3] / 255.0, 0.0, 1.0)

    rgba = np.empty((height, width, 4), dtype=np.uint8)
    rgba[..., :3] = np.clip(warped[..., :3], 0, 255).astype(np.uint8)
    rgba[..., 3] = np.clip(alpha * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(rgba)
