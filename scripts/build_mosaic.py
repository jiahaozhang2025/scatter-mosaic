"""Build the standalone interactive mosaic from an embedding and a folder of images."""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from umap_mosaic import images as img  # noqa: E402
from umap_mosaic.dataset import dataset_name, load_embedding  # noqa: E402
from umap_mosaic.layout import compact_clusters, normalize_isotropic  # noqa: E402
from umap_mosaic.painting import paint_patches, paint_spots, quantize_per_region  # noqa: E402
from umap_mosaic.placement import allocate_faces, assign_regions  # noqa: E402
from umap_mosaic.silhouette import make_masks, mask_bbox, morph_into_mask  # noqa: E402
from umap_mosaic.theme import THEMES, color_for, use_theme  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Turn a 2-D embedding into a zoomable mosaic of images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    data = p.add_argument_group("data")
    data.add_argument("--embedding", type=Path, required=True, help="CSV with two coordinate columns.")
    data.add_argument("--images", type=Path, default=None, help="Folder of images. Defaults to images_cutout/ then images/.")
    data.add_argument("--x-column", default=None, help="Override the guessed x column.")
    data.add_argument("--y-column", default=None, help="Override the guessed y column.")
    data.add_argument("--id-column", default=None, help="Override the guessed identifier column.")
    data.add_argument("--keep-column", default=None, help="Boolean column saying which rows to use.")
    data.add_argument("--map", type=Path, default=None, help="Region-to-image table. Defaults to <out>/image_map.csv.")
    data.add_argument("--out", type=Path, default=ROOT / "results", help="Where outputs land.")
    data.add_argument("--remap", action="store_true", help="Rebuild the image table from the folder, discarding edits.")

    look = p.add_argument_group("look")
    look.add_argument("--title", default=None, help="Headline. Defaults to the dataset name taken from the path.")
    look.add_argument("--subtitle", default=None)
    look.add_argument("--theme", choices=tuple(THEMES), default="light", help="Theme the viewer opens on; both are switchable inside it.")
    look.add_argument("--style", choices=("paint", "veil"), default="paint")
    look.add_argument("--photo", type=float, default=None, help="How much image, 0-1. Defaults per theme.")
    look.add_argument("--ghost", type=float, default=None, help="How much colour points outside a face keep. Defaults per theme.")
    look.add_argument("--spot-scale", type=float, default=1.0, help="Starting multiplier on dot size; adjustable in the viewer.")
    look.add_argument("--paint-colors", type=int, default=48, help="Palette size per region for the painted dots.")

    place = p.add_argument_group("placement")
    place.add_argument("--placement", choices=("scatter", "cluster"), default="scatter")
    place.add_argument("--regions", type=int, default=0, help="Number of regions. 0 matches the image count.")
    place.add_argument("--face-size", type=float, default=0.055)
    place.add_argument("--face-fill", type=float, default=0.85)
    place.add_argument("--face-density", type=float, default=0.55, help="Density floor as a fraction of this cloud's typical covered density.")
    place.add_argument("--face-adjacency", type=float, default=0.6)
    place.add_argument("--face-group", type=int, default=4)
    place.add_argument("--face-min-size", type=float, default=0.016)
    place.add_argument("--group-column", default=None, help="Use an existing column as regions (cluster placement only).")
    place.add_argument("--compact", type=float, default=0.0, help="0-1. Packs regions inward; 0 keeps the plain embedding.")
    place.add_argument("--cluster-gap", type=float, default=0.010)
    place.add_argument("--morph", type=float, default=0.45, help="Cluster placement only.")
    place.add_argument("--morph-limit", type=float, default=1.9)
    place.add_argument("--seed", type=int, default=42)

    p.add_argument("--veil-grid", type=int, default=1400)
    p.add_argument("--veil-max-px", type=int, default=512)
    p.add_argument("--no-veil", action="store_true")
    p.add_argument("--no-paint", action="store_true")
    return p


def placeholder(index: int, color: str) -> str:
    svg = f"""<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 96 96'>
      <rect width='96' height='96' fill='{color}'/>
      <text x='48' y='60' text-anchor='middle' font-family='Arial' font-size='34'
            font-weight='700' fill='white'>{index + 1}</text>
    </svg>"""
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


def webp_uri(image: Image.Image, quality: int = 80) -> str:
    buffer = io.BytesIO()
    image.save(buffer, "WEBP", quality=quality, method=4)
    return "data:image/webp;base64," + base64.b64encode(buffer.getvalue()).decode()


def chip_uri(path: Path, size: int = 128) -> str:
    """A legend chip that keeps its alpha, so the region colour can sit behind it."""
    art = img.load(path, keep_alpha=True).convert("RGBA")
    side = min(art.size)
    left = (art.width - side) // 2
    top = int((art.height - side) * 0.46)
    art = art.crop((left, top, left + side, top + side)).resize((size, size), Image.Resampling.LANCZOS)
    return webp_uri(art)


def renumber_left_to_right(coords: np.ndarray, labels: np.ndarray, count: int) -> np.ndarray:
    centroids = np.vstack([coords[labels == c].mean(axis=0) for c in range(count)])
    order = np.lexsort((centroids[:, 1], centroids[:, 0]))
    remap = np.empty(count, dtype=np.int16)
    remap[order] = np.arange(count, dtype=np.int16)
    return remap[labels]


def ensure_map(path: Path, count: int, catalogue: list[dict], remap: bool) -> pd.DataFrame:
    """Region -> image table, auto-filled from the folder when empty."""
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = pd.read_csv(path, dtype=str).fillna("") if path.exists() else pd.DataFrame()
    usable = (
        not existing.empty
        and "image" in existing
        and existing["image"].str.strip().ne("").any()
        and len(existing) == count
    )
    if catalogue and (remap or not usable):
        rows = [
            {
                "region": i,
                "label": catalogue[i % len(catalogue)]["name"],
                "group": catalogue[i % len(catalogue)]["group"],
                "image": catalogue[i % len(catalogue)]["filename"],
            }
            for i in range(count)
        ]
    else:
        by_region = {}
        for row in existing.to_dict("records"):
            try:
                by_region[int(row["region"])] = row
            except (KeyError, TypeError, ValueError):
                continue
        rows = [
            {
                "region": i,
                "label": str(by_region.get(i, {}).get("label") or f"Image {i + 1:02d}"),
                "group": str(by_region.get(i, {}).get("group") or ""),
                "image": str(by_region.get(i, {}).get("image") or ""),
            }
            for i in range(count)
        ]
    table = pd.DataFrame(rows)
    table.to_csv(path, index=False)
    return table


def build_silhouette_overlays(x, y, labels, count, art, grid, max_px, morph, limit):
    """One morphed, soft-edged image per region, placed in normalized space."""
    masks = make_masks(x, y, labels, count, grid, grid)
    overlays, sprites = [], []
    for region, mask in enumerate(masks):
        source = art[region]
        box = mask_bbox(mask)
        if source is None or box is None:
            continue
        x0, y0, x1, y1 = box
        crop = Image.fromarray((mask[y0:y1, x0:x1] * 255).astype(np.uint8))
        longest = max(crop.size)
        if longest > max_px:
            crop = crop.resize(
                (max(8, round(crop.width * max_px / longest)), max(8, round(crop.height * max_px / longest))),
                Image.Resampling.LANCZOS,
            )
        crop = crop.filter(ImageFilter.GaussianBlur(radius=max(1.0, min(crop.size) / 90)))
        overlays.append({
            "region": region,
            "art": len(sprites),
            "box": [round(x0 / grid, 5), round(y0 / grid, 5), round(x1 / grid, 5), round(y1 / grid, 5)],
        })
        sprites.append(webp_uri(morph_into_mask(source, crop, morph=morph, limit=limit)))
    return overlays, sprites


def build_square_overlays(patches, art, max_px):
    """Square images for the scatter placement, stored once per region."""
    sprites, index, overlays = [], {}, []
    for patch in patches:
        region = patch["region"]
        source = art[region] if region < len(art) else None
        if source is None:
            continue
        if region not in index:
            index[region] = len(sprites)
            sprites.append(webp_uri(source.convert("RGBA").resize((max_px, max_px), Image.Resampling.LANCZOS)))
        overlays.append({"region": region, "art": index[region], "box": [round(v, 5) for v in patch["box"]]})
    return overlays, sprites


def main() -> None:
    args = parser().parse_args()
    theme = use_theme(args.theme)
    if args.photo is None:
        args.photo = theme["photo_alpha"]
    if args.ghost is None:
        args.ghost = theme["ghost"][args.placement]

    args.out.mkdir(parents=True, exist_ok=True)
    image_dir = args.images or img.default_dir(ROOT)
    catalogue = img.listing(image_dir)
    count = args.regions or len(catalogue) or 12
    if count < 2:
        raise SystemExit(f"Need at least 2 regions; found {len(catalogue)} images in {image_dir}")

    data = load_embedding(
        args.embedding, args.x_column, args.y_column, args.id_column, args.keep_column,
        require=(args.group_column,) if args.group_column else (),
    )
    coords = np.column_stack([data.x, data.y])

    if args.group_column:
        if args.group_column not in data.extras:
            raise SystemExit(f"--group-column {args.group_column!r} is not in {args.embedding.name}")
        grouping = data.extras[args.group_column].astype("category").cat.codes.to_numpy(np.int16)
    else:
        grouping = KMeans(n_clusters=count, n_init=20, random_state=args.seed).fit_predict(coords).astype(np.int16)
    layout_x, layout_y, packing = compact_clusters(
        coords[:, 0], coords[:, 1], grouping, int(grouping.max()) + 1,
        strength=args.compact, gap=args.cluster_gap,
    )
    x, y = normalize_isotropic(layout_x, layout_y)

    patches: list[dict] = []
    if args.placement == "scatter":
        patches, used = allocate_faces(
            x, y, count, size=args.face_size, min_fill=args.face_fill,
            min_density=args.face_density, adjacency=args.face_adjacency,
            group_cap=args.face_group, min_size=args.face_min_size, seed=args.seed,
        )
        if not patches:
            raise SystemExit("Could not place a single image. Lower --face-size or --face-density.")
        labels = assign_regions(x, y, patches)
        count = len(patches)
        source = f"{count} regions · {used:.3f} squares on the densest ground · points join their nearest image"
    else:
        labels = grouping
        count = int(labels.max()) + 1
        if not args.group_column:
            labels = renumber_left_to_right(np.column_stack([layout_x, layout_y]), labels, count)
            source = f"K-means on the embedding (k={count}, seed={args.seed})"
        else:
            source = f"existing column: {args.group_column}"
    labels = labels.astype(np.int16)
    if args.compact > 0:
        source += f" · compacted {args.compact:.2f}"

    table = ensure_map(args.map or args.out / "image_map.csv", count, catalogue, args.remap)
    entries, art = [], []
    for row in table.itertuples(index=False):
        region = int(row.region)
        path = img.resolve(image_dir, str(getattr(row, "image", "") or ""))
        art.append(img.load(path, keep_alpha=True) if path else None)
        entries.append({
            "region": region,
            "name": str(row.label),
            "group": str(getattr(row, "group", "") or ""),
            "src": chip_uri(path) if path else placeholder(region, color_for(region)),
            "source": path.name if path else "placeholder",
            "count": int((labels == region).sum()),
        })

    overlays, sprites = [], []
    if not args.no_veil:
        if args.placement == "scatter":
            overlays, sprites = build_square_overlays(patches, art, args.veil_max_px)
        else:
            overlays, sprites = build_silhouette_overlays(
                x, y, labels, count, art, args.veil_grid, args.veil_max_px, args.morph, args.morph_limit
            )

    paint_index, palettes = "", []
    if not args.no_paint:
        if args.placement == "scatter":
            painted, on_subject = paint_patches(x, y, patches, art)
        else:
            painted, on_subject = paint_spots(
                x, y, labels, count, art, grid_w=args.veil_grid,
                morph=args.morph, limit=args.morph_limit, feather=0.002,
            )
        palettes, codes = quantize_per_region(painted, on_subject, labels, count, args.paint_colors)
        # Index 0 stays free: the viewer paints it with the faded region colour of
        # whichever theme is showing, so the theme can change without a rebuild.
        for palette in palettes:
            palette.insert(0, None)
        paint_index = base64.b64encode(codes.tobytes()).decode()

    style = args.style
    if style == "veil" and not overlays:
        style = "paint"
    if style == "paint" and not palettes:
        style = "veil"

    payload = {
        "meta": {
            "title": args.title or f"{dataset_name(args.embedding)} — image mosaic",
            "subtitle": args.subtitle or f"{len(data):,} points · {count} regions",
            "source": f"{data.source} · {source}",
            "instructions": "Scroll to zoom, drag to pan, click a point or a legend image to inspect.",
            "groupLabel": "region" if args.placement == "scatter" else "cluster",
            "theme": args.theme,
            "style": style,
            "photo": round(float(np.clip(args.photo, 0, 1)), 3),
            "spotScale": round(float(np.clip(args.spot_scale, 0.2, 4.0)), 3),
        },
        "themes": {
            name: {
                "stage": spec["stage"],
                "ghost": spec["ghost"][args.placement],
                "palette": [color_for(i, name) for i in range(count)],
            }
            for name, spec in THEMES.items()
        },
        "x": np.round(x, 5).tolist(),
        "y": np.round(y, 5).tolist(),
        "region": labels.astype(int).tolist(),
        "id": data.ids.tolist(),
        "fields": data.tooltip_fields(),
        "entries": entries,
        "overlays": overlays,
        "sprites": sprites,
        "paint": {"index": paint_index, "palettes": palettes},
    }

    template = (ROOT / "umap_mosaic" / "template.html").read_text(encoding="utf-8")
    html = template.replace("__MOSAIC_DATA__", json.dumps(payload, separators=(",", ":"), ensure_ascii=False))
    if html == template:
        raise SystemExit("Template is missing the __MOSAIC_DATA__ placeholder")

    out_html = args.out / "mosaic.html"
    out_html.write_text(html, encoding="utf-8")
    pd.DataFrame({
        "id": data.ids, "region": labels,
        "layout_x": np.round(layout_x, 5), "layout_y": np.round(layout_y, 5),
    }).to_csv(args.out / "regions.csv", index=False)
    (args.out / "patches.json").write_text(
        json.dumps({"placement": args.placement, "patches": patches}, indent=1), encoding="utf-8"
    )

    real = sum(e["source"] != "placeholder" for e in entries)
    swatches = sum(max(0, len(p) - 1) for p in palettes)
    print(f"Built {out_html}  ({out_html.stat().st_size / 1e6:.1f} MB)")
    print(f"Embedded {len(data):,} points, {real}/{count} images, {len(overlays)} overlays, {swatches} paint swatches")
    if args.placement == "scatter":
        groups: dict[int, int] = {}
        for patch in patches:
            groups[patch["group"]] = groups.get(patch["group"], 0) + 1
        spread = np.bincount(labels, minlength=count)
        print(f"Placement: {len(patches)} squares at {patches[0]['size']:.4f}, packed {sorted(groups.values(), reverse=True)}, "
              f"regions {spread.min():,}-{spread.max():,} points")
    if args.compact > 0:
        print(f"Compaction: {packing['shrink']:.2f}x linear extent")
    print(f"Opens on theme={args.theme} style={style}; both switchable in the page")


if __name__ == "__main__":
    main()
