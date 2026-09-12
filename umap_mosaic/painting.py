"""Colour the spots themselves so the image emerges from the scatter.

The veil style lays a morphed photo *over* the spots, which means the photo hides
the thing it is drawn on. This style never draws a photo at all: it samples the
portrait at each cell's position and hands that colour to the cell's own dot. The
true spot geometry survives intact, and the face appears as a pointillist image
made of real cells.

Points that land outside the subject keep their region colour, which is why the
cut-outs from extract_faces.py matter — without them every point would be painted
with a piece of whatever was behind the subject.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from .silhouette import make_masks, mask_bbox, morph_into_mask


def _prepare(portrait: Image.Image) -> Image.Image:
    """Stretch the portrait's contrast without disturbing its cutout alpha."""
    if portrait.mode == "RGBA":
        stretched = ImageOps.autocontrast(portrait.convert("RGB"), cutoff=1)
        stretched.putalpha(portrait.getchannel("A"))
        return stretched
    return ImageOps.autocontrast(portrait, cutoff=1)


def _keep(alpha: np.ndarray, where: np.ndarray) -> np.ndarray:
    """Turn a soft alpha edge into a hard yes/no per point, without a hard edge.

    A dot is painted or it is not, so a half-opaque pixel cannot be half painted.
    Thresholding at the halfway mark throws that information away and leaves a
    cut line; comparing against a per-point value spends it instead, so half-
    opaque ground keeps half of its dots and the edge dissolves at dot scale.

    The comparison value is hashed from the point's own index rather than drawn
    at random, so the same point makes the same choice in the viewer and in the
    poster, on every run.
    """
    mixed = where.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)
    mixed ^= mixed >> np.uint64(29)
    mixed *= np.uint64(0xBF58476D1CE4E5B9)
    mixed ^= mixed >> np.uint64(32)
    scatter = (mixed % np.uint64(1 << 24)).astype(np.float32) / float(1 << 24)
    return alpha.astype(np.float32) / 255.0 > scatter


def paint_spots(
    x: np.ndarray,
    y: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
    portraits: list[Image.Image | None],
    grid_w: int = 1400,
    grid_h: int | None = None,
    morph: float = 0.45,
    limit: float = 1.9,
    feather: float = 0.0,
    masks: list[np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a colour for every cell from its cluster's morphed portrait.

    `x` and `y` are normalized to [0, 1] with y already pointing down. Returns an
    (n, 3) uint8 colour array and an (n,) bool saying whether the cell landed on
    the subject rather than on removed background.
    """
    grid_h = grid_h or grid_w
    if masks is None:
        masks = make_masks(x, y, labels, n_clusters, grid_w, grid_h)

    rgb = np.zeros((len(x), 3), dtype=np.uint8)
    on_subject = np.zeros(len(x), dtype=bool)
    px = np.clip((x * (grid_w - 1)).astype(int), 0, grid_w - 1)
    py = np.clip((y * (grid_h - 1)).astype(int), 0, grid_h - 1)

    for cluster, mask in enumerate(masks):
        portrait = portraits[cluster] if cluster < len(portraits) else None
        box = mask_bbox(mask)
        if portrait is None or box is None:
            continue
        x0, y0, x1, y1 = box
        plate = Image.fromarray((mask[y0:y1, x0:x1] * 255).astype(np.uint8))
        if feather > 0:
            plate = plate.filter(ImageFilter.GaussianBlur(max(1.0, feather * min(plate.size))))
        warped = np.asarray(morph_into_mask(_prepare(portrait), plate, morph=morph, limit=limit))

        hit = labels == cluster
        cx = np.clip(px[hit] - x0, 0, warped.shape[1] - 1)
        cy = np.clip(py[hit] - y0, 0, warped.shape[0] - 1)
        sample = warped[cy, cx]
        rgb[hit] = sample[:, :3]
        on_subject[hit] = _keep(sample[:, 3], np.flatnonzero(hit))
    return rgb, on_subject


def paint_patches(
    x: np.ndarray,
    y: np.ndarray,
    patches: list[dict],
    portraits: list[Image.Image | None],
    size: int = 320,
) -> tuple[np.ndarray, np.ndarray]:
    """Colour every cell that falls inside a square face patch.

    No morphing: the patch is square and the portrait is square, so the face goes
    in undistorted. Cells outside every patch, or on a patch's removed background,
    come back flagged so the caller can leave them their cluster colour.
    """
    rgb = np.zeros((len(x), 3), dtype=np.uint8)
    on_subject = np.zeros(len(x), dtype=bool)
    cache: dict[int, np.ndarray] = {}

    for patch in patches:
        region = patch["region"]
        portrait = portraits[region] if region < len(portraits) else None
        if portrait is None:
            continue
        if region not in cache:
            prepared = _prepare(portrait)
            cache[region] = np.asarray(prepared.convert("RGBA").resize((size, size), Image.Resampling.LANCZOS))
        art = cache[region]

        x0, y0, x1, y1 = patch["box"]
        inside = (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
        if not inside.any():
            continue
        cx = np.clip(((x[inside] - x0) / (x1 - x0) * (size - 1)).astype(int), 0, size - 1)
        cy = np.clip(((y[inside] - y0) / (y1 - y0) * (size - 1)).astype(int), 0, size - 1)
        sample = art[cy, cx]
        rgb[inside] = sample[:, :3]
        on_subject[inside] = _keep(sample[:, 3], np.flatnonzero(inside))
    return rgb, on_subject


def quantize_per_region(
    rgb: np.ndarray,
    on_subject: np.ndarray,
    labels: np.ndarray,
    n_regions: int,
    colors: int = 48,
) -> tuple[list[list[list[int]]], np.ndarray]:
    """Squeeze the painted colours into a small palette per cluster.

    The viewer redraws every frame, so it needs to batch points by colour rather
    than set a fill per point. Index 0 of each palette is reserved for cells that
    fell outside the subject; the caller fills it with the cluster's own colour.
    """
    colors = int(np.clip(colors, 2, 254))
    index = np.zeros(len(labels), dtype=np.uint8)
    palettes: list[list[list[int]]] = []

    for region in range(n_regions):
        hit = (labels == region) & on_subject
        if not hit.any():
            palettes.append([])
            continue
        strip = Image.fromarray(rgb[hit].reshape(1, -1, 3))
        reduced = strip.quantize(colors=colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
        codes = np.asarray(reduced).reshape(-1)
        flat = reduced.getpalette() or []
        used = int(codes.max()) + 1
        palettes.append([list(flat[i * 3:i * 3 + 3]) for i in range(used)])
        index[hit] = codes.astype(np.uint8) + 1
    return palettes, index


def ghost_color(
    region_rgb: np.ndarray, ghost: float, background: tuple[int, int, int] = (0, 0, 0)
) -> np.ndarray:
    """The faded region colour worn by cells that fall outside the subject.

    At full strength those cells shout over the faces, because a palette hue is far
    more saturated than skin. Fading them toward the *background* is what makes them
    recede — multiplying toward black would only make them louder on a white page.
    """
    base = np.asarray(background, dtype=np.float32)
    amount = float(np.clip(ghost, 0.0, 1.0))
    return np.clip(base + amount * (region_rgb.astype(np.float32) - base), 0, 255).astype(np.uint8)


def blend_with_region(
    rgb: np.ndarray,
    on_subject: np.ndarray,
    labels: np.ndarray,
    region_rgb: np.ndarray,
    mix: float,
    ghost: float = 0.45,
    background: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Per-cell colours eased between the plain region palette (0) and the photo (1)."""
    base = region_rgb[labels].astype(np.float32)
    faded = ghost_color(region_rgb, ghost, background)[labels]
    target = np.where(on_subject[:, None], rgb, faded).astype(np.float32)
    amount = float(np.clip(mix, 0.0, 1.0))
    return np.clip(base + amount * (target - base), 0, 255).astype(np.uint8)
