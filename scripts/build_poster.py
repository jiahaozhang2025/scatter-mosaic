from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scatter_mosaic import images as img  # noqa: E402
from scatter_mosaic.dataset import dataset_name, load_embedding  # noqa: E402
from scatter_mosaic.layout import normalize_isotropic  # noqa: E402
from scatter_mosaic.painting import blend_with_region, paint_patches, paint_spots  # noqa: E402
from scatter_mosaic.placement import allocate_faces  # noqa: E402
from scatter_mosaic.silhouette import make_masks, mask_bbox, morph_into_mask  # noqa: E402
from scatter_mosaic.theme import THEMES, color_for, theme_rgb, use_theme  # noqa: E402


def arguments() -> argparse.Namespace:
    project = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(
        description="Render a print-resolution poster from a built mosaic."
    )
    p.add_argument("--regions", type=Path, required=True, help="regions.csv written by build_mosaic.py.")
    p.add_argument("--map", type=Path, default=None, help="image_map.csv. Defaults to sit beside --regions.")
    p.add_argument("--patches", type=Path, default=None, help="patches.json. Defaults to sit beside --regions.")
    p.add_argument("--images", type=Path, default=None, help="Folder of images. Defaults to images_cutout/ then images/.")
    p.add_argument("--output", type=Path, default=None, help="Defaults to poster.png beside --regions.")
    p.add_argument("--width", type=int, default=4800)
    p.add_argument("--height", type=int, default=3600)
    p.add_argument("--title", default=None)
    p.add_argument("--subtitle", default=None)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--theme", choices=tuple(THEMES), default="light", help="Page colours and the palette built for them.")
    p.add_argument(
        "--placement",
        choices=("scatter", "cluster"),
        default="scatter",
        help="scatter drops square faces on dense patches; cluster fits one face to each cluster silhouette.",
    )
    p.add_argument("--face-size", type=float, default=0.055, help="Face square edge, used only when the builder left no face_patches.json.")
    p.add_argument("--face-fill", type=float, default=0.85)
    p.add_argument("--face-density", type=float, default=0.55)
    p.add_argument("--face-adjacency", type=float, default=0.6)
    p.add_argument("--face-group", type=int, default=4)
    p.add_argument("--seed", type=int, default=42, help="Seed for the fallback placement, matching the builder.")
    p.add_argument(
        "--style",
        choices=("paint", "veil"),
        default="paint",
        help="paint colours the spots from the portrait; veil lays a morphed photo over them.",
    )
    p.add_argument(
        "--photo-alpha",
        type=float,
        default=None,
        help="How much photo. In veil style the layer opacity; in paint style how far the spot colours move from their cluster colour.",
    )
    p.add_argument(
        "--morph",
        type=float,
        default=0.45,
        help="0 center-crops each portrait, 1 leans it as far onto the silhouette as the clamp allows.",
    )
    p.add_argument(
        "--morph-limit",
        type=float,
        default=1.9,
        help="Largest per-slice stretch or squeeze the morph may apply. Lower keeps faces truer.",
    )
    p.add_argument("--spot-radius", type=float, default=0.0, help="Spot radius in pixels. 0 picks one from the cell density.")
    p.add_argument("--spot-alpha", type=float, default=0.80, help="Opacity of the underlying spot field.")
    p.add_argument("--hide-spots", action="store_true", help="Drop the spot layer and show portraits on the plain background.")
    p.add_argument(
        "--ghost",
        type=float,
        default=None,
        help="Brightness of points outside the subject. Defaults to 0.30 for scatter, 0.45 for cluster.",
    )
    p.add_argument("--no-badges", action="store_true", help="Hide the numbered cluster badges.")
    p.add_argument("--outline", action="store_true", help="Draw cluster halos in paint style (always drawn in veil style).")
    return p.parse_args()


def font(size: int, bold: bool = False, serif: bool = False) -> ImageFont.FreeTypeFont:
    if serif:
        candidates = [Path(r"C:\Windows\Fonts\georgia.ttf"), Path(r"C:\Windows\Fonts\times.ttf")]
    elif bold:
        candidates = [Path(r"C:\Windows\Fonts\arialbd.ttf"), Path(r"C:\Windows\Fonts\segoeuib.ttf")]
    else:
        candidates = [Path(r"C:\Windows\Fonts\arial.ttf"), Path(r"C:\Windows\Fonts\segoeui.ttf")]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default(size=size)


def ellipsize(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, room: int) -> str:
    """Trim `text` with an ellipsis so a long name cannot run into the next legend cell."""
    if room <= 0 or draw.textlength(text, font=font) <= room:
        return text
    while text and draw.textlength(text + "…", font=font) > room:
        text = text[:-1]
    return text.rstrip() + "…"


def placeholder_portrait(cluster: int, color: str, size: int = 900) -> Image.Image:
    skin = ["#f3c6a2", "#dda77e", "#ad7552", "#74472f"][cluster % 4]
    hair = ["#211816", "#583b32", "#c39127", "#15273a"][cluster % 4]
    image = Image.new("RGB", (size, size), color)
    draw = ImageDraw.Draw(image)
    for radius, opacity in [(440, 24), (340, 34), (250, 45)]:
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        od.ellipse((size/2-radius, size/2-radius, size/2+radius, size/2+radius), fill=(255,255,255,opacity))
        image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.ellipse((size*.18, size*.14, size*.82, size*.86), fill=skin)
    draw.pieslice((size*.15, size*.06, size*.85, size*.68), 180, 360, fill=hair)
    draw.ellipse((size*.31, size*.43, size*.38, size*.51), fill="#17151a")
    draw.ellipse((size*.62, size*.43, size*.69, size*.51), fill="#17151a")
    draw.arc((size*.34, size*.50, size*.67, size*.74), 20, 160, fill="#8c3445", width=max(4, size//70))
    draw.text((size*.5, size*.92), f"{cluster + 1:02d}", anchor="mm", font=font(size//11, bold=True), fill="white")
    return image


def portrait_for(cluster: int, color: str, filename: str, faces_dir: Path) -> tuple[Image.Image, str]:
    path = img.resolve(faces_dir, filename)
    if path is not None:
        return img.load(path, keep_alpha=True), path.name
    return placeholder_portrait(cluster, color), "placeholder"


def draw_spot_field(
    size: tuple[int, int],
    px: np.ndarray,
    py: np.ndarray,
    clusters: np.ndarray,
    n_clusters: int,
    radius: float,
    colors: np.ndarray | None = None,
    background: str = "#000000",
) -> tuple[Image.Image, Image.Image]:
    """The real scatter: one soft dot per cell, coloured by its visual cluster.

    Returns the colour plate and a coverage mask, so the dots can be composited
    onto the background without dark colours turning translucent.
    """
    supersample = 2 if max(size) <= 6000 else 1
    big = (size[0] * supersample, size[1] * supersample)
    # Base the plate on the page colour: anti-aliasing blends toward it, and on a
    # white page a black base would ring every dot with a dark halo.
    canvas = Image.new("RGB", big, background)
    coverage = Image.new("L", big, 0)
    ink = ImageDraw.Draw(canvas)
    cover = ImageDraw.Draw(coverage)
    r = radius * supersample
    if colors is None:
        for cluster in range(n_clusters):
            hit = clusters == cluster
            colour = color_for(cluster)
            for x, y in zip(px[hit] * supersample, py[hit] * supersample):
                box = (x - r, y - r, x + r, y + r)
                ink.ellipse(box, fill=colour)
                cover.ellipse(box, fill=255)
    else:
        for x, y, rgb in zip(px * supersample, py * supersample, colors):
            box = (x - r, y - r, x + r, y + r)
            ink.ellipse(box, fill=(int(rgb[0]), int(rgb[1]), int(rgb[2])))
            cover.ellipse(box, fill=255)
    if supersample > 1:
        canvas = canvas.resize(size, Image.Resampling.LANCZOS)
        coverage = coverage.resize(size, Image.Resampling.LANCZOS)
    return canvas, coverage


def main() -> None:
    args = arguments()
    theme = use_theme(args.theme)
    # Scatter leaves most points outside a face, so their colour has to sit further back.
    if args.ghost is None:
        args.ghost = theme["ghost"][args.placement]
    if args.photo_alpha is None:
        args.photo_alpha = theme["photo_alpha"]
    # The poster consumes what build_mosaic.py already worked out: the layout, the
    # region each point belongs to, and exactly where the squares went. Recomputing
    # any of it would risk the two renders quietly disagreeing.
    home = args.regions.parent
    frame = pd.read_csv(args.regions)
    mapping = pd.read_csv(args.map or home / "image_map.csv", dtype=str).fillna("")
    args.output = args.output or home / "poster.png"
    faces_dir = args.images or img.default_dir(project)

    clusters = frame["region"].to_numpy(dtype=np.int16)
    n_clusters = int(clusters.max()) + 1
    raw_x = frame["layout_x"].to_numpy(dtype=np.float64)
    raw_y = frame["layout_y"].to_numpy(dtype=np.float64)
    title = args.title or f"{dataset_name(args.regions)} — image mosaic".upper()
    subtitle = args.subtitle or f"{len(frame):,} points · {n_clusters} regions"
    # One scale for both axes, matching the viewer, so a square face patch stays
    # square instead of being stretched to whatever aspect the page happens to be.
    x, y = normalize_isotropic(raw_x, raw_y)

    width, height = args.width, args.height
    margin_x = int(width * .045)
    header_h = int(height * .145)
    legend_h = int(height * .175)
    plot_box = (margin_x, header_h, width - margin_x, height - legend_h)
    plot_w, plot_h = plot_box[2] - plot_box[0], plot_box[3] - plot_box[1]

    # Scale the cloud's own bounding box into the plot box, not the whole unit
    # square: the layout rarely fills it, and fitting the square would throw away
    # most of the page.
    span_x, span_y = max(float(np.ptp(x)), 1e-9), max(float(np.ptp(y)), 1e-9)
    scale = min(plot_w / span_x, plot_h / span_y)
    art_w, art_h = int(round(span_x * scale)), int(round(span_y * scale))
    origin = (plot_box[0] + (plot_w - art_w) // 2, plot_box[1] + (plot_h - art_h) // 2)
    left, top = float(x.min()), float(y.min())

    def to_page(nx, ny):
        """Normalized layout coordinates to page pixels, equal scale on both axes."""
        return origin[0] + (np.asarray(nx) - left) * scale, origin[1] + (np.asarray(ny) - top) * scale

    grid = max(700, int(max(art_w, art_h) // 4))
    masks = make_masks(x, y, clusters, n_clusters, grid, grid)

    background = Image.new("RGB", (width, height), theme["background"])
    if args.theme == "dark":
        bg_draw = ImageDraw.Draw(background)
        for radius, tone in [(int(width*.46), "#171020"), (int(width*.34), "#12101b")]:
            bg_draw.ellipse((width*.5-radius, height*.48-radius, width*.5+radius, height*.48+radius), fill=tone)
        background = background.filter(ImageFilter.GaussianBlur(radius=int(width*.035)))

    map_rows = {int(row["region"]): row for row in mapping.to_dict("records") if str(row.get("region", "")).isdigit()}
    portraits: list[Image.Image] = []
    names: list[str] = []
    roles: list[str] = []
    sources: list[str] = []
    for cluster in range(n_clusters):
        row = map_rows.get(cluster, {})
        names.append(str(row.get("label") or f"Image {cluster + 1:02d}"))
        roles.append(str(row.get("group") or ""))
        portrait, source = portrait_for(cluster, color_for(cluster), str(row.get("image") or ""), faces_dir)
        portraits.append(portrait)
        sources.append(source)

    draw = ImageDraw.Draw(background)
    draw.text((margin_x, int(height*.035)), title, font=font(int(height*.041), serif=True), fill=theme["ink"])
    draw.text((margin_x, int(height*.097)), subtitle.upper(), font=font(int(height*.0105), bold=True), fill=theme["muted"], spacing=4)
    tag = (
        f"DOTS PAINTED FROM THE IMAGES · {round(args.photo_alpha * 100)}%" if args.style == "paint"
        else f"IMAGE OVERLAY · {round(args.photo_alpha * 100)}%"
    )
    tag_font = font(int(height*.009), bold=True)
    tag_box = draw.textbbox((0, 0), tag, font=tag_font)
    tag_w = tag_box[2] - tag_box[0]
    draw.rounded_rectangle((width-margin_x-tag_w-70, int(height*.045), width-margin_x, int(height*.075)), radius=20, fill=theme["accent"])
    draw.text((width-margin_x-35, int(height*.060)), tag, anchor="rm", font=tag_font, fill="#ffffff" if args.theme == "light" else "#17121d")

    photo_alpha = float(np.clip(args.photo_alpha, 0, 1))
    badges: list[tuple[int, int]] = []
    # The builder saves exactly where it put the faces, so the two renders agree
    # without both having to arrive at the same answer independently.
    patches: list[dict] = []
    if args.placement == "scatter":
        sidecar = args.patches or home / "patches.json"
        if sidecar.is_file():
            patches = json.loads(sidecar.read_text(encoding="utf-8")).get("patches", [])
        if patches:
            print(f"Images: {len(patches)} placed, from {sidecar.name}")
        else:
            patches, used = allocate_faces(
                x, y, n_clusters, size=args.face_size, min_fill=args.face_fill,
                min_density=args.face_density, adjacency=args.face_adjacency,
                group_cap=args.face_group, seed=args.seed,
            )
            print(f"Images: {len(patches)} recomputed at size {used:.4f} ({sidecar.name} not found)")

    # Layer 1: the spot field. In paint style this is the whole picture; in veil
    # style it is the ground the portraits are laid over.
    if not args.hide_spots:
        spot_r = args.spot_radius or max(1.6, math.sqrt(art_w * art_h / max(len(frame), 1)) * .30)
        painted = None
        if args.style == "paint":
            if args.placement == "scatter":
                sampled, on_subject = paint_patches(x, y, patches, portraits)
            else:
                sampled, on_subject = paint_spots(
                    x, y, clusters, n_clusters, portraits,
                    grid, grid, args.morph, args.morph_limit, feather=0.002, masks=masks,
                )
            cluster_rgb = np.array(
                [list(bytes.fromhex(color_for(c).lstrip("#"))) for c in range(n_clusters)], dtype=np.uint8
            )
            painted = blend_with_region(
                sampled, on_subject, clusters, cluster_rgb, photo_alpha, args.ghost, theme_rgb("background"),
            )
            spot_r *= 1.5   # the dots have to meet for the portrait to read
        page_x, page_y = to_page(x, y)
        spots, coverage = draw_spot_field(
            (art_w + 1, art_h + 1), page_x - origin[0], page_y - origin[1],
            clusters, n_clusters, spot_r, painted, theme["background"],
        )
        strength = np.asarray(coverage, dtype=np.float32) / 255.0
        opacity = 1.0 if args.style == "paint" else float(np.clip(args.spot_alpha, 0, 1))
        alpha = Image.fromarray(np.clip(strength * opacity * 255, 0, 255).astype(np.uint8))
        background.paste(spots, origin, alpha)
        draw = ImageDraw.Draw(background)
    for cluster, low_mask in enumerate(masks):
        box = mask_bbox(low_mask)
        if box is None:
            continue
        gx0, gy0, gx1, gy1 = box
        x0, y0 = to_page(gx0 / grid, gy0 / grid)
        x1, y1 = to_page(gx1 / grid, gy1 / grid)
        bbox = (max(plot_box[0], int(x0)), max(plot_box[1], int(y0)),
                min(plot_box[2], int(x1)), min(plot_box[3], int(y1)))
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue

        full = int(round(scale))
        mask_full = Image.fromarray((low_mask * 255).astype(np.uint8)).resize((full, full), Image.Resampling.LANCZOS)
        shift = (int(round(origin[0] - left * scale)), int(round(origin[1] - top * scale)))
        mask_crop = mask_full.crop((bbox[0]-shift[0], bbox[1]-shift[1], bbox[2]-shift[0], bbox[3]-shift[1]))
        # Two feathers: a wide one so the photo melts into the spot field, and a
        # tight one that only smooths the stair-steps of the low-resolution grid.
        feathered = mask_crop.filter(ImageFilter.GaussianBlur(radius=max(2.0, min(mask_crop.size) / 70)))
        traced = mask_crop.filter(ImageFilter.GaussianBlur(radius=max(1.5, min(mask_crop.size) / 190)))

        if args.style == "veil" and args.placement == "cluster":
            # Layer 2: the portrait warped onto the silhouette, over the spots at partial opacity.
            warped = morph_into_mask(portraits[cluster], feathered, morph=args.morph, limit=args.morph_limit)
            face_rgb = ImageOps.autocontrast(warped.convert("RGB"), cutoff=1)
            face_rgb = ImageEnhance.Brightness(face_rgb).enhance(1.0 + theme["photo_lift"] * (1 - photo_alpha))
            shape_alpha = np.asarray(warped.getchannel("A"), dtype=np.float32) / 255.0
            blend = Image.fromarray(np.clip(shape_alpha * photo_alpha * 255, 0, 255).astype(np.uint8))
            background.paste(face_rgb, (bbox[0], bbox[1]), blend)

        if (args.style == "veil" and args.placement == "cluster") or args.outline:
            # Colored halo keeps neighbouring silhouettes apart without hiding the photo.
            halo = traced.filter(ImageFilter.MaxFilter(13))
            edge = np.maximum(np.asarray(halo, dtype=np.int16) - np.asarray(traced, dtype=np.int16), 0)
            edge_mask = Image.fromarray(np.clip(edge * .95, 0, 255).astype(np.uint8))
            edge_layer = Image.new("RGB", mask_crop.size, color_for(cluster))
            background.paste(edge_layer, (bbox[0], bbox[1]), edge_mask)

        if not args.no_badges:
            ys, xs = np.where(low_mask)
            cx, cy = to_page(np.median(xs) / grid, np.median(ys) / grid)
            badge_r = max(22, int(height*.009))
            # A cluster's centroid is usually where its face patch landed too, so
            # slide the badge out from under any patch it would otherwise cover.
            for patch in patches:
                px0, py0 = to_page(patch["box"][0], patch["box"][1])
                px1, py1 = to_page(patch["box"][2], patch["box"][3])
                if px0 - badge_r < cx < px1 + badge_r and py0 - badge_r < cy < py1 + badge_r:
                    cy = py1 + badge_r * 1.35
            # Packed faces push their badges to the same strip of free space, so
            # step sideways until this one is clear of the badges already drawn.
            for _ in range(8):
                clash = next((b for b in badges if abs(b[0] - cx) < 2.1 * badge_r and abs(b[1] - cy) < 2.1 * badge_r), None)
                if clash is None:
                    break
                cx = clash[0] + 2.2 * badge_r
            cx = int(min(max(cx, plot_box[0] + badge_r), plot_box[2] - badge_r))
            cy = int(min(max(cy, plot_box[1] + badge_r), plot_box[3] - badge_r))
            badges.append((cx, cy))
            draw = ImageDraw.Draw(background)
            draw.ellipse((cx-badge_r, cy-badge_r, cx+badge_r, cy+badge_r), fill=theme["background"], outline=color_for(cluster), width=max(3, width//1200))
            draw.text((cx, cy), str(cluster + 1), anchor="mm", font=font(max(18, int(height*.009)), bold=True), fill=theme["ink"])

    # Layer 2 for the scatter placement: an undistorted square portrait per patch.
    if args.style == "veil" and args.placement == "scatter":
        for patch in patches:
            art = portraits[patch["region"]].convert("RGBA")
            bx0, by0, bx1, by1 = patch["box"]
            px0, py0 = (int(v) for v in to_page(bx0, by0))
            px1, py1 = (int(v) for v in to_page(bx1, by1))
            if px1 <= px0 or py1 <= py0:
                continue
            square = art.resize((px1 - px0, py1 - py0), Image.Resampling.LANCZOS)
            face_rgb = ImageOps.autocontrast(square.convert("RGB"), cutoff=1)
            face_rgb = ImageEnhance.Brightness(face_rgb).enhance(1.0 + theme["photo_lift"] * (1 - photo_alpha))
            shape_alpha = np.asarray(square.getchannel("A"), dtype=np.float32) / 255.0
            blend = Image.fromarray(np.clip(shape_alpha * photo_alpha * 255, 0, 255).astype(np.uint8))
            background.paste(face_rgb, (px0, py0), blend)

    # Legend: portrait, number, name, role and point count — until there are too
    # many regions for that to fit, at which point the names are dropped and it
    # becomes a numbered contact strip. Two fixed rows used to be assumed here,
    # and anything past the twenty-fourth region was drawn off the bottom edge.
    # Every measurement scales with the canvas so --width/--height stay usable.
    draw = ImageDraw.Draw(background)
    legend_top = height - legend_h + int(height*.025)
    band_w, band_h = width - 2*margin_x, legend_h - int(height*.045)
    named = n_clusters <= 24
    if named:
        rows = 2
        columns = max(1, math.ceil(n_clusters / rows))
    else:
        # Squarish grid, then relax until each row is tall enough to see.
        rows = max(1, round(math.sqrt(n_clusters * band_h / band_w)))
        while rows > 1 and band_h / rows < height * .028:
            rows -= 1
        columns = math.ceil(n_clusters / rows)
    cell_w, cell_h = band_w / columns, band_h / rows
    thumb_px = (max(24, int(height * .0228)) if named
                else max(16, int(min(cell_h * .82, cell_w * .82))))
    text_x = thumb_px + max(6, thumb_px // 7)
    name_font = font(max(11, int(height*.0068)), bold=True)
    detail_font = font(max(10, int(height*.0056)))
    text_room = int(cell_w - text_x - max(8, width * .004))
    for cluster in range(n_clusters):
        row, col = divmod(cluster, columns)
        x0 = int(margin_x + col*cell_w)
        y0 = int(legend_top + row*cell_h)
        chip = img.flatten(portraits[cluster], color_for(cluster))
        thumb = ImageOps.fit(chip, (thumb_px,)*2, method=Image.Resampling.LANCZOS, centering=(.5, .46))
        thumb_mask = Image.new("L", (thumb_px,)*2, 0)
        ImageDraw.Draw(thumb_mask).ellipse((2, 2, thumb_px-2, thumb_px-2), fill=255)
        background.paste(thumb, (x0, y0), thumb_mask)
        draw.ellipse((x0+1, y0+1, x0+thumb_px-1, y0+thumb_px-1), outline=color_for(cluster), width=max(2, thumb_px//16))
        if not named:
            continue
        count = int((clusters == cluster).sum())
        detail = f"{roles[cluster]} · {count:,} points" if roles[cluster] else f"{count:,} points"
        draw.text((x0+text_x, y0+int(thumb_px*.14)),
                  ellipsize(draw, f"{cluster+1:02d}  {names[cluster]}", name_font, text_room),
                  font=name_font, fill=theme["ink"])
        draw.text((x0+text_x, y0+int(thumb_px*.58)),
                  ellipsize(draw, detail, detail_font, text_room),
                  font=detail_font, fill=theme["muted"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    background.save(args.output, dpi=(args.dpi, args.dpi), optimize=True)
    preview = ImageOps.contain(background, (1600, 1200), Image.Resampling.LANCZOS)
    preview_path = args.output.with_name(args.output.stem + "_preview.jpg")
    preview.save(preview_path, quality=91, optimize=True)
    print(f"Saved poster: {args.output}")
    print(f"Saved preview: {preview_path}")
    print(f"Placement: {args.placement}" + (f" · {len(patches)} square patches" if args.placement == "scatter" else ""))
    print(f"Style: {args.style} · morph {args.morph:.2f} (limit {args.morph_limit:.2f}) · photo {photo_alpha:.2f} · spots {'off' if args.hide_spots else 'on'}")
    content = float(np.ptp(x) / max(np.ptp(y), 1e-9))
    print(f"Plot area {art_w}x{art_h} px of {plot_w}x{plot_h} available (layout aspect {content:.2f}:1)")
    print(f"Images: {sum(s != 'placeholder' for s in sources)} real, {sum(s == 'placeholder' for s in sources)} placeholder")


if __name__ == "__main__":
    main()
