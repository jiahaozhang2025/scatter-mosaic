"""Take the background off a portrait, leaving the subject on transparency.

GrabCut needs to be told roughly where the subject is before it can model the
colours, and for a portrait that seed is knowable in advance: a head, a pair of
shoulders, and a wide band of definite background around them. That is all this
module is — the trimap, the segmentation, and the two touch-ups that keep the
result from betraying itself when it is resized.

It is a colour model, so it fails when the background shares the subject's
palette, and it fails badly on drawings: on an anime crop it takes the face and
throws the hair away. `cut_out` reports when it has fallen back to the template
shape rather than shipping a bad mask quietly.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import binary_closing, binary_fill_holes, label


def pad_to_frame(picture: Image.Image, margin: float = 0.16) -> Image.Image:
    """Give a tight crop some room, so the trimap's border band is not on the hair.

    An aligned face crop puts the top of the head against the frame, exactly where
    the trimap marks definite background, and the cut comes back with a haircut.
    The padding is filled with the crop's own border colour rather than a mirror
    of it: replicated pixels drag hair-coloured streaks up into the new space and
    the colour model happily keeps them.
    """
    art = np.asarray(picture.convert("RGB"))
    edge = np.concatenate([art[0], art[-1], art[:, 0], art[:, -1]])
    pad = int(round(art.shape[0] * margin))
    wide = cv2.copyMakeBorder(art, pad, pad, pad, pad, cv2.BORDER_CONSTANT,
                              value=[int(v) for v in np.median(edge, axis=0)])
    return Image.fromarray(wide)


def head_and_shoulders(side: int, grow: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """GrabCut seeds for a crop where square_crop has put the face at a known spot.

    Returns the trimap, the bare template silhouette (the fallback shape), and the
    envelope the result is clipped to so a far corner can never be claimed.
    """
    def ellipse(canvas, cx, cy, rx, ry):
        cv2.ellipse(canvas, (int(cx * side), int(cy * side)),
                    (int(rx * side), int(ry * side)), 0, 0, 360, 1, -1)

    body = np.zeros((side, side), np.uint8)
    ellipse(body, 0.50, 1.12, 0.62, 0.48)   # shoulders
    ellipse(body, 0.50, 0.44, 0.33, 0.41)   # head
    kernel = np.ones((3, 3), np.uint8)
    near = cv2.dilate(body, kernel, iterations=max(1, int(side * grow)))
    envelope = cv2.dilate(body, kernel, iterations=max(1, int(side * 0.14)))

    # Everything well outside the silhouette is definite background. That negative
    # evidence is what lets the colour model tell hair from whatever is behind it.
    trimap = np.where(near > 0, cv2.GC_PR_BGD, cv2.GC_BGD).astype(np.uint8)
    trimap[body > 0] = cv2.GC_PR_FGD
    core = np.zeros((side, side), np.uint8)
    ellipse(core, 0.50, 0.46, 0.17, 0.23)
    trimap[core > 0] = cv2.GC_FGD
    band = max(2, int(side * 0.02))
    trimap[:band, :] = trimap[-band:, :] = cv2.GC_BGD
    trimap[:, :band] = cv2.GC_BGD
    trimap[:, -band:] = cv2.GC_BGD
    return trimap, body > 0, envelope > 0


def cut_out(crop: Image.Image, grow: float, iterations: int) -> tuple[np.ndarray, bool]:
    """Boolean subject mask for a square crop, plus whether it fell back to the template."""
    side = crop.width
    bgr = cv2.cvtColor(np.asarray(crop.convert("RGB")), cv2.COLOR_RGB2BGR)
    trimap, body, envelope = head_and_shoulders(side, grow)
    cv2.grabCut(bgr, trimap, None, np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64),
                iterations, cv2.GC_INIT_WITH_MASK)

    subject = np.isin(trimap, (cv2.GC_FGD, cv2.GC_PR_FGD)) & envelope
    parts, count = label(binary_closing(subject, np.ones((5, 5)), iterations=2))
    if count:
        keep = parts[int(side * 0.46), int(side * 0.50)]
        if keep == 0:
            sizes = np.bincount(parts.ravel())
            sizes[0] = 0
            keep = int(sizes.argmax())
        subject = parts == keep
    subject = binary_fill_holes(subject)

    # Claiming nearly everything, or almost nothing, means the colour model never
    # separated subject from background. The template shape is the safer answer.
    failed = bool(subject[envelope].mean() > 0.95 or subject.mean() < 0.06)
    return (body if failed else subject), failed


def bleed_edges(rgb: np.ndarray, subject: np.ndarray, steps: int = 6) -> np.ndarray:
    """Grow subject colours a few pixels into the background.

    Without this the RGB hiding under transparent pixels bleeds back in whenever the
    cutout is resized or feathered, outlining every face in leftover background.
    """
    out = rgb.astype(np.float32)
    known = subject.copy()
    kernel = np.ones((3, 3), np.uint8)
    for _ in range(steps):
        grown = cv2.dilate(known.astype(np.uint8), kernel).astype(bool)
        fresh = grown & ~known
        if not fresh.any():
            break
        weight = cv2.blur(known.astype(np.float32), (3, 3))
        blurred = cv2.blur(np.where(known[..., None], out, 0.0), (3, 3))
        out[fresh] = (blurred / np.maximum(weight, 1e-6)[..., None])[fresh]
        known = grown
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_cutout(crop: Image.Image, subject: np.ndarray, feather: float) -> Image.Image:
    rgb = bleed_edges(np.asarray(crop.convert("RGB")), subject)
    alpha = Image.fromarray((subject * 255).astype(np.uint8))
    radius = max(0.6, feather * crop.width)
    alpha = alpha.filter(ImageFilter.GaussianBlur(radius))
    return Image.fromarray(np.dstack([rgb, np.asarray(alpha)]))


def checkerboard(size: int, cell: int = 14) -> Image.Image:
    """A grey check, so a review sheet shows where a cutout is transparent."""
    tile = (np.indices((size, size)).sum(axis=0) // cell) % 2
    return Image.fromarray(np.where(tile[..., None], 108, 84).repeat(3, 2).astype(np.uint8))
