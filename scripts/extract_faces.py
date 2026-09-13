"""Detect the face in each portrait, re-crop around it, and cut out the background.

The mosaic paints these onto cluster silhouettes, so whatever survives here is
what people see. Two steps:

1. Find the face and take a square, consistently framed crop around it.
2. Segment subject from background with GrabCut, seeded by a head-and-shoulders
   template placed where the crop puts the face. Output is RGBA, so downstream
   the background is genuinely absent rather than merely cropped.

Both use OpenCV's bundled machinery and need no downloads. Pass --model with a
YuNet ONNX file (opencv_zoo `face_detection_yunet_*.onnx`) for better detection
recall on tilted or partly turned faces.

GrabCut is a colour-model method: it fails when the background shares the
subject's palette (a brick wall behind a person in warm tones). Those cases are
reported with their coverage rather than silently shipped.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scatter_mosaic import images as img  # noqa: E402
from scatter_mosaic.cutout import apply_cutout, checkerboard, cut_out  # noqa: E402
from scatter_mosaic.theme import THEMES, use_theme  # noqa: E402


def arguments() -> argparse.Namespace:
    project = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description="Crop each portrait to its detected face.")
    p.add_argument("--input", type=Path, default=project / "images", help="Folder of source images.")
    p.add_argument("--output", type=Path, default=project / "images_cutout", help="Where the cut-outs go.")
    p.add_argument("--size", type=int, default=256, help="Edge length of each square output crop.")
    p.add_argument(
        "--pad",
        type=float,
        default=0.42,
        help="Margin around the detected face box, as a fraction of its longest edge.",
    )
    p.add_argument(
        "--focus-y",
        type=float,
        default=0.46,
        help="Where the face centre sits vertically in the crop. Below 0.5 leaves room for the chin.",
    )
    p.add_argument("--model", type=Path, default=None, help="Optional YuNet .onnx face detector.")
    p.add_argument("--min-confidence", type=float, default=0.6, help="YuNet score threshold.")
    p.add_argument(
        "--upscale",
        type=int,
        default=6,
        help="Factor the image is enlarged by before detection. Helps on small thumbnails.",
    )
    p.add_argument(
        "--min-quality",
        type=float,
        default=0.35,
        help="Reject detections below this size-and-centrality score and center-crop instead.",
    )
    p.add_argument(
        "--min-crop",
        type=float,
        default=0.50,
        help="Smallest crop as a fraction of the source, so a tiny face is not blown up to mush.",
    )
    p.add_argument(
        "--cutout",
        choices=("grabcut", "template", "none"),
        default="grabcut",
        help="grabcut segments the subject; template uses the plain head-and-shoulders shape; none keeps the background.",
    )
    p.add_argument(
        "--cutout-grow",
        type=float,
        default=0.07,
        help="How far past the template GrabCut may claim subject, as a fraction of the crop.",
    )
    p.add_argument("--cutout-iterations", type=int, default=8)
    p.add_argument(
        "--feather",
        type=float,
        default=0.008,
        help="Softness of the cutout edge, as a fraction of the crop.",
    )
    p.add_argument("--contact-sheet", type=Path, default=None, help="Where to write a review grid.")
    p.add_argument("--no-contact-sheet", action="store_true")
    p.add_argument("--theme", choices=tuple(THEMES), default="light", help="Colours for the review grid.")
    return p.parse_args()


# alt2 is the most reliable of the bundled cascades; default is looser and can
# latch onto clothing texture, and profile only earns a vote for turned heads.
CASCADES = (
    ("haarcascade_frontalface_alt2.xml", 1.0),
    ("haarcascade_frontalface_default.xml", 0.9),
    ("haarcascade_profileface.xml", 0.6),
)
ROUNDS = ((1.06, 5, 12), (1.10, 3, 18), (1.05, 2, 26))


def detect_haar(gray: np.ndarray) -> list[tuple[int, int, int, int, float]]:
    """Run every cascade at the strictest settings first, loosening only if nothing fires."""
    equalized = cv2.equalizeHist(gray)
    loaded = [
        (cv2.CascadeClassifier(cv2.data.haarcascades + name), score)
        for name, score in CASCADES
    ]
    for scale, neighbors, divisor in ROUNDS:
        found: list[tuple[int, int, int, int, float]] = []
        for cascade, score in loaded:
            if cascade.empty():
                continue
            boxes = cascade.detectMultiScale(
                equalized, scaleFactor=scale, minNeighbors=neighbors,
                minSize=(max(16, gray.shape[0] // divisor),) * 2,
            )
            found += [(int(x), int(y), int(w), int(h), score) for x, y, w, h in boxes]
        if found:
            return found
    return []


def detect_yunet(bgr: np.ndarray, model: Path, threshold: float) -> list[tuple[int, int, int, int, float]]:
    h, w = bgr.shape[:2]
    detector = cv2.FaceDetectorYN.create(str(model), "", (w, h), threshold, 0.3, 5000)
    _, faces = detector.detect(bgr)
    if faces is None:
        return []
    return [(int(f[0]), int(f[1]), int(f[2]), int(f[3]), float(f[-1])) for f in faces]


def box_quality(box: tuple[int, int, int, int, float], shape: tuple[int, int]) -> float:
    """How much a detection looks like the subject of a portrait rather than clutter.

    A portrait's face is large and near the middle of the frame. Cascades happily
    fire on foliage, brickwork and snow, but those hits are almost always both
    small and off to one side, so weighting on size and centrality separates them
    without needing a second model.
    """
    x, y, w, h, score = box
    height, width = shape
    relative = max(w, h) / max(min(width, height), 1)
    off = 2 * max(abs((x + w / 2) / width - 0.5), abs((y + h / 2) / height - 0.5))
    return score * min(relative / 0.35, 1.0) * (1 - 0.75 * off ** 2)


def best_box(
    boxes: list[tuple[int, int, int, int, float]], shape: tuple[int, int], minimum: float
) -> tuple[tuple[int, int, int, int] | None, float]:
    """The most plausible detection, or nothing when every candidate looks like clutter."""
    if not boxes:
        return None, 0.0
    best = max(boxes, key=lambda b: box_quality(b, shape))
    quality = box_quality(best, shape)
    if quality < minimum:
        return None, quality
    return best[:4], quality


def square_crop(
    image: Image.Image,
    box: tuple[int, int, int, int] | None,
    pad: float,
    focus_y: float,
    size: int,
    min_crop: float = 0.0,
) -> Image.Image:
    """Square crop centred on the face, reflecting the edges when it runs off the image."""
    if box is None:
        side = min(image.width, image.height)
        cx, cy = image.width / 2, image.height / 2
    else:
        x, y, w, h = box
        side = max(max(w, h) * (1 + 2 * pad), min_crop * min(image.width, image.height))
        cx = x + w / 2
        cy = y + h / 2 + side * (0.5 - focus_y)

    left = int(round(cx - side / 2))
    top = int(round(cy - side / 2))
    side = int(round(side))

    source = np.asarray(image)
    over = max(0, -left, -top, left + side - image.width, top + side - image.height)
    if over > 0:
        source = np.pad(source, ((over, over), (over, over), (0, 0)), mode="reflect")
        left += over
        top += over
    crop = Image.fromarray(source[top:top + side, left:left + side])
    return crop.resize((size, size), Image.Resampling.LANCZOS)


def contact_sheet(rows: list[dict], path: Path, theme: dict, thumb: int = 150) -> None:
    columns = min(7, max(1, len(rows)))
    lines = (len(rows) + columns - 1) // columns
    label_h = 34
    sheet = Image.new("RGB", (columns * thumb, lines * (thumb + label_h)), theme["panel"])
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 12)
    except OSError:
        font = ImageFont.load_default()
    tiles = checkerboard(thumb)
    for i, row in enumerate(rows):
        col, line = i % columns, i // columns
        x0, y0 = col * thumb, line * (thumb + label_h)
        art = row["crop"].resize((thumb, thumb), Image.Resampling.LANCZOS)
        if art.mode == "RGBA":
            art = Image.alpha_composite(tiles.convert("RGBA"), art).convert("RGB")
        sheet.paste(art, (x0, y0))
        clean = row["detected"] and not row["template"] and row["coverage"] <= 0.60
        good, bad = ("#1f9d4d", "#c72c1c") if theme["background"] == "#ffffff" else ("#7bed9f", "#ff6b6b")
        colour = good if clean else bad
        draw.rectangle((x0, y0, x0 + thumb - 1, y0 + thumb - 1), outline=colour, width=2)
        draw.text((x0 + 5, y0 + thumb + 4), row["name"][:24], font=font, fill=theme["ink"])
        note = "face" if row["detected"] else "center crop"
        if row["template"]:
            note += " · template"
        note += f" · {row['coverage']:.0%} kept"
        draw.text((x0 + 5, y0 + thumb + 18), note, font=font, fill=colour)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=92)


def main() -> None:
    args = arguments()
    theme = use_theme(args.theme)
    files = sorted(p for p in args.input.iterdir() if p.is_file() and p.suffix.lower() in img.SUFFIXES)
    if not files:
        raise SystemExit(f"No portraits found in {args.input}")

    model = args.model
    if model is None:
        found = sorted(Path(__file__).resolve().parents[1].glob("**/face_detection_yunet*.onnx"))
        model = found[0] if found else None
    if model is not None and not Path(model).is_file():
        raise SystemExit(f"Detector model not found: {model}")
    detector_name = f"YuNet ({Path(model).name})" if model else "OpenCV Haar cascade"

    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in files:
        image = img.load(path)
        big = image.resize(
            (image.width * args.upscale, image.height * args.upscale), Image.Resampling.LANCZOS
        )
        bgr = cv2.cvtColor(np.asarray(big), cv2.COLOR_RGB2BGR)
        if model:
            boxes = detect_yunet(bgr, Path(model), args.min_confidence)
        else:
            boxes = detect_haar(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
        box, quality = best_box(boxes, bgr.shape[:2], args.min_quality)
        crop = square_crop(big, box, args.pad, args.focus_y, args.size, args.min_crop)

        template = False
        if args.cutout == "none":
            cutout, coverage = crop.convert("RGB"), 1.0
        else:
            if args.cutout == "template":
                subject = head_and_shoulders(crop.width, args.cutout_grow)[1]
            else:
                subject, template = cut_out(crop, args.cutout_grow, args.cutout_iterations)
            coverage = float(subject.mean())
            cutout = apply_cutout(crop, subject, args.feather)

        suffix = ".jpg" if args.cutout == "none" else ".png"
        cutout.save(args.output / path.with_suffix(suffix).name)
        rows.append({
            "crop": cutout,
            "name": img.describe(path)["name"] or path.stem,
            "detected": box is not None,
            "quality": quality,
            "coverage": coverage,
            "template": template,
        })

    if not args.no_contact_sheet:
        sheet_path = args.contact_sheet or args.output.parent / "cutout_review.jpg"
        contact_sheet(rows, sheet_path, theme)
        print(f"Review grid: {sheet_path}")

    hits = sum(r["detected"] for r in rows)
    print(f"Detector: {detector_name}  ·  cutout: {args.cutout}")
    print(f"Wrote {len(rows)} crops to {args.output} ({hits} detected, {len(rows) - hits} center-cropped)")
    if hits < len(rows):
        missed = ", ".join(f"{r['name']} (best {r['quality']:.2f})" for r in rows if not r["detected"])
        print(f"Center-cropped, no plausible face: {missed}")
    # A head-and-shoulders crop should land near 30-50% coverage. Much more than that
    # means background survived the cut, which is worth a look rather than a silent pass.
    suspect = [r for r in rows if args.cutout == "grabcut" and (r["template"] or r["coverage"] > 0.55)]
    if suspect:
        detail = ", ".join(
            f"{r['name']} ({'template fallback' if r['template'] else f'{r['coverage']:.0%} kept'})"
            for r in suspect
        )
        print(f"Background may not have separated cleanly: {detail}")


if __name__ == "__main__":
    main()
