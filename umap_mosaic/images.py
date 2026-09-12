"""Finding, naming and loading the images that get placed on the map.

Nothing here knows what the images are of. A filename of the form
`NN_group_first-last.jpg` yields an order, a group label and a display name,
which is convenient but entirely optional: anything else falls back to the
filename as the label.
"""

from __future__ import annotations

import re
from pathlib import Path

from PIL import Image, ImageOps

SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

# Expanded only for readability in the legend; unknown values are title-cased.
GROUP_LABELS = {
    "faculty": "Faculty",
    "postdoc": "Postdoc",
    "grad": "Grad student",
    "research-associate": "Research associate",
    "staff": "Staff",
}


def _titlecase(token: str) -> str:
    return " ".join(part.capitalize() for part in token.replace("_", "-").split("-") if part)


def describe(path: Path) -> dict:
    """Turn `20_grad_jiahao-zhang.jpg` into an order, a group and a display name."""
    match = re.match(r"^(\d+)[_-]([a-zA-Z-]+)[_-](.+)$", path.stem)
    if match:
        order, group, name = int(match.group(1)), match.group(2).lower(), match.group(3)
        return {
            "filename": path.name,
            "order": order,
            "group": GROUP_LABELS.get(group, _titlecase(group)),
            "name": _titlecase(name),
        }
    return {"filename": path.name, "order": 10**6, "group": "", "name": _titlecase(path.stem)}


def listing(folder: Path) -> list[dict]:
    """Every usable image in `folder`, ordered by any numeric filename prefix.

    Names starting with an underscore are skipped, which keeps side-products like
    a contact sheet from being mistaken for content.
    """
    if not folder.is_dir():
        return []
    found = [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in SUFFIXES and not p.name.startswith("_")
    ]
    return sorted((describe(p) for p in found), key=lambda item: (item["order"], item["filename"]))


def resolve(folder: Path, filename: str) -> Path | None:
    """Find `filename` in `folder`, tolerating a different extension.

    Cut-outs are always written as PNG, so a mapping table generated from a folder
    of JPEGs still points at the right image after re-cropping.
    """
    if not filename:
        return None
    direct = folder / filename
    if direct.is_file():
        return direct
    stem = Path(filename).stem
    for suffix in sorted(SUFFIXES):
        candidate = folder / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def default_dir(project: Path, images: str = "images", cutouts: str = "images_cutout") -> Path:
    """Prefer the cut-out folder once extract_faces.py has produced one."""
    cropped = project / cutouts
    return cropped if listing(cropped) else project / images


def load(path: Path, keep_alpha: bool = False) -> Image.Image:
    """Load an image as RGB, or as RGBA when the file carries a cutout mask."""
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source)
        if keep_alpha and ("A" in image.getbands() or image.mode == "P"):
            return image.convert("RGBA")
        return image.convert("RGB")


def flatten(image: Image.Image, background: str) -> Image.Image:
    """Composite a cutout over a solid colour, for chips that cannot hold alpha."""
    if image.mode != "RGBA":
        return image.convert("RGB")
    plate = Image.new("RGBA", image.size, background)
    return Image.alpha_composite(plate, image).convert("RGB")
