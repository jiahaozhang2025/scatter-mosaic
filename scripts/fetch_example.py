"""Build a runnable example from a public dataset: an embedding plus its tile images.

Where the points come from (`--dataset`), and how they are laid out (`--layout`):

Nothing downstream of this script knows or cares how the coordinates were made. It
is a 2-D scatter painter, so `--layout` picks between UMAP, t-SNE and plain PCA, and
`cities` skips the question entirely by arriving with real coordinates already.

  mnist    70,000 handwritten digits, 28x28 — ~15 MB from OpenML. The default, and
           the best-looking map of the lot: long curved islands, wisps, and bridges
           between the digits that get confused for each other.
  fashion  70,000 garment photographs, 10 classes — ~30 MB from OpenML. Fewer,
           fatter masses than MNIST, which suits images that tile across a shape.
  kmnist   70,000 Kuzushiji characters, 10 classes — ~18 MB from OpenML. Cursive
           Japanese; the classes overlap more than digits do, so the map comes out
           stringier and more tangled.
  cartoon  100,000 avatars from Google's Cartoon Set — ~25 MB, embedded on the
           eighteen attributes that generated each face. Eight round islands: very
           clean, and duller for it.
  lfw      13,233 colour photographs of 5,749 people — ~200 MB, Labeled Faces in
           the Wild. The one that exercises the face-extraction step. Be warned
           that a UMAP of raw pixels barely separates people at all.
  digits   1,797 points, 10 digits, 8x8 — bundled with scikit-learn, no download.
  olivetti 400 points, 40 people, 64x64 — ~4.5 MB, the AT&T face database.
  cities   ~25,000 cities of over 15,000 people — ~3 MB from GeoNames. Not an
           embedding at all: the coordinates are longitude and latitude. Europe,
           India, China and the US eastern seaboard are the dense islands, and the
           oceans are the gaps between them.

Where the pictures come from (`--images-from`, optional):

  faces    AI-generated portraits from SFHQ. The one that needs cutting out, and
           the one that shows what `extract_faces.py` does: GrabCut against a
           head-and-shoulders trimap, so the points land on a person and not on
           the wall behind them. Synthetic, so nobody's likeness is in the repo.
  anime    anime portraits. No alpha to be had and GrabCut eats the hair, so the
           squares are feathered at the edge instead of cut out.
  emoji    one emoji from each Unicode group, straight from OpenMoji. Already
           transparent.
  cartoon  avatars from Google's Cartoon Set. Already transparent.

All four are picked so no two look alike from across the room. Leave
`--images-from` off and each dataset illustrates itself, one image per class. Set
it and the two come apart, which is the normal case: the points are your data and
the pictures are whatever you want the map to be made of.

Each writes `embedding.csv` and an `images/` folder into `examples/<name>/`, which
is all `build_mosaic.py` needs. Nothing here is committed to the repository; the
data is fetched on demand and cached.
"""

from __future__ import annotations

import argparse
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import binary_closing, binary_fill_holes, gaussian_filter
from sklearn.cluster import KMeans
from sklearn.datasets import (
    fetch_lfw_people,
    fetch_olivetti_faces,
    fetch_openml,
    load_digits,
)
from sklearn.decomposition import PCA
from umap import UMAP

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

sys.path.insert(0, str(ROOT))

from umap_mosaic.cutout import apply_cutout, cut_out, pad_to_frame  # noqa: E402
from umap_mosaic.remote import RemoteFile  # noqa: E402

DIGIT_NAMES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
GARMENT_NAMES = ["t-shirt", "trouser", "pullover", "dress", "coat",
                 "sandal", "shirt", "sneaker", "bag", "ankle-boot"]
# The ten Kuzushiji classes, by the hiragana each cursive form stands for.
KUZUSHIJI_NAMES = ["o", "ki", "su", "tsu", "na", "ha", "ma", "ya", "re", "wo"]
LFW_HOME = Path.home() / "scikit_learn_data" / "lfw_home" / "lfw_funneled"

GEONAMES_CITIES = "https://download.geonames.org/export/dump/cities15000.zip"
CARTOON_PARQUET = ("https://huggingface.co/api/datasets/cgarciae/cartoonset"
                   "/parquet/100k+features/train/{part}.parquet")
CARTOON_PARTS = 10
ANIME_PARQUET = ("https://huggingface.co/api/datasets/HK83/Anime_Faces"
                 "/parquet/default/train/0.parquet")
SFHQ_PARQUET = ("https://huggingface.co/api/datasets/canva999888/SFHQ-Tiny-512-Part1"
                "/parquet/default/validation/0.parquet")
OPENMOJI_INDEX = "https://raw.githubusercontent.com/hfg-gmuend/openmoji/master/data/openmoji.json"
OPENMOJI_PNG = "https://raw.githubusercontent.com/hfg-gmuend/openmoji/master/color/618x618/{code}.png"
# Round-robin over these so a set of ten is not ten yellow faces. `people-body` and
# the two `extras-` groups are skipped: mostly skin-tone variants of one gesture.
EMOJI_GROUPS = ("smileys-emotion", "animals-nature", "food-drink", "travel-places",
                "activities", "objects", "symbols", "flags")
# The attributes a viewer can actually see, promoted to the tooltip.
CARTOON_TOOLTIP = ["hair", "hair_color", "face_color", "glasses", "facial_hair", "eye_color"]

# What suits each map's shape. Cartoon Set makes eight round islands, so it wants
# one big square each — and a fill rule slack enough to let a square overhang a
# circle's edge, which the default 0.85 forbids.
TUNING = {
    "cartoon": {"tiles": 8, "build": ["--face-size", "0.16", "--face-fill", "0.62"]},
    "fashion": {"build": ["--face-size", "0.14"]},
    "mnist": {"build": ["--face-size", "0.14", "--face-fill", "0.72"]},
    "kmnist": {"build": ["--face-size", "0.14", "--face-fill", "0.72"]},
}


@dataclass
class Source:
    """A dataset reduced to what the mosaic needs: vectors, and a way to get tiles."""

    vectors: np.ndarray
    group: str
    mode: str
    labels: np.ndarray | None = None
    names: list[str] | None = None
    images: np.ndarray | None = None
    extras: pd.DataFrame | None = None
    # Given row indices, hand back the pictures. Used when there are no classes.
    pictures: Callable[[list[int]], dict[int, Image.Image]] | None = None
    notes: list[str] = field(default_factory=list)
    # Set when the data already has coordinates and no layout step is wanted.
    # Kept last: every other caller passes positionally.
    coords: np.ndarray | None = None


def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fetch a public dataset and lay it out for the mosaic tools.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset", choices=("mnist", "fashion", "kmnist", "cartoon", "lfw",
                                         "digits", "olivetti", "cities"),
                   default="mnist", help="Where the points come from.")
    p.add_argument("--layout", choices=("umap", "tsne", "pca"), default="umap",
                   help="How to get from features to two dimensions. Ignored by datasets "
                        "that arrive with coordinates of their own.")
    p.add_argument("--images-from", choices=("faces", "anime", "emoji", "cartoon"), default=None,
                   help="Take the tile pictures from here instead of from the dataset's "
                        "own classes. The points and the pictures are unrelated, which is "
                        "the normal case.")
    p.add_argument("--out", type=Path, default=None, help="Defaults to examples/<dataset>/")
    p.add_argument("--limit", type=int, default=40000, help="Cap on points, for speed. 0 keeps all.")
    p.add_argument("--neighbors", type=int, default=15, help="UMAP n_neighbors.")
    p.add_argument("--min-dist", type=float, default=0.1, help="UMAP min_dist.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tile-size", type=int, default=256, help="Edge of each written tile image.")
    p.add_argument("--tiles", type=int, default=None,
                   help="How many images to write. With classes, the best-represented ones; "
                        "without, one per region of the map. Defaults to what suits the dataset.")
    p.add_argument("--tile-pool", type=int, default=400,
                   help="Candidate pictures to choose the tiles from. Each hundred is one "
                        "more range request.")
    args = p.parse_args()
    if args.tiles is None:
        args.tiles = 10 if args.images_from else TUNING.get(args.dataset, {}).get("tiles", 24)
    return args


# --------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------
def cartoon(limit: int, pool: int) -> Source:
    """Google's Cartoon Set, read a slice at a time out of a 489 MB parquet.

    The embedding is built from the eighteen attributes that generated each
    avatar rather than from its pixels. That is both far cheaper — the attribute
    columns are a megabyte, the images are the other 488 — and far better: a
    UMAP of cartoon pixels comes out as thin wisps, while the attributes give
    solid islands the mosaic can actually sit images on.
    """
    import pyarrow.parquet as pq

    from umap_mosaic.remote import session

    connection = session()
    frames, first, columns = [], None, None
    print("Reading Cartoon Set attributes from HuggingFace...")
    for part in range(CARTOON_PARTS):
        remote = RemoteFile(CARTOON_PARQUET.format(part=part), connection)
        handle = pq.ParquetFile(remote, pre_buffer=True)
        if columns is None:
            columns = [n for n in handle.schema_arrow.names
                       if n != "img_bytes" and not n.endswith("_num_categories")]
            first = (handle, remote)
        frames.append(handle.read(columns=columns).to_pandas())
        have = sum(len(f) for f in frames)
        print(f"  part {part}: {have:,} avatars so far", flush=True)
        if limit and have >= limit:
            break
    frame = pd.concat(frames, ignore_index=True)
    if limit and limit < len(frame):
        frame = frame.iloc[:limit]
    handle, remote = first
    print(f"  {len(frame):,} avatars x {len(columns)} attributes")

    rows_per_group = handle.metadata.row_group(0).num_rows
    groups = max(1, min(handle.metadata.num_row_groups, -(-pool // rows_per_group)))

    def pictures(wanted: list[int]) -> dict[int, Image.Image]:
        """Decode the tile candidates. Only the leading row groups are fetched."""
        blobs: list[bytes] = []
        for group in range(groups):
            blobs += handle.read_row_group(group, columns=["img_bytes"])["img_bytes"].to_pylist()
        print(f"  tile candidates: {len(blobs)} avatars, {remote.megabytes:.0f} MB from the first part")
        return {i: Image.open(io.BytesIO(blobs[i])).convert("RGBA") for i in wanted if i < len(blobs)}

    return Source(
        vectors=frame[columns].to_numpy(dtype=np.float32),
        group="cartoon",
        mode="rgba",
        extras=frame[[c for c in CARTOON_TOOLTIP if c in frame]].reset_index(drop=True),
        pictures=pictures,
        notes=[f"tile candidates come from the first {groups * rows_per_group} rows"],
    )


def pick_by_colour(pool: list[Image.Image], count: int, seed: int) -> list[Image.Image]:
    """`count` pictures that do not look alike from across the room.

    Average colour is a crude descriptor and the right one here: at tile size and
    tile distance it is most of what distinguishes one picture from another. The
    most typical member of each colour group also keeps the odd crop out.
    """
    if len(pool) <= count:
        return pool
    swatch = np.stack([np.asarray(p.convert("RGB").resize((8, 8)), dtype=np.float32)
                       .reshape(-1, 3).mean(0) for p in pool])
    groups = KMeans(count, n_init=10, random_state=seed).fit(swatch)
    chosen = []
    for index, centre in enumerate(groups.cluster_centers_):
        members = np.flatnonzero(groups.labels_ == index)
        if members.size:
            chosen.append(int(members[np.argmin(((swatch[members] - centre) ** 2).sum(1))]))
    return [pool[i] for i in chosen]


def parquet_pictures(url: str, column: str, pool: int, what: str) -> list[Image.Image]:
    """Decode the leading `pool` pictures out of a remote parquet.

    Row groups are the unit: asking for an arbitrary row means fetching the whole
    group around it, so taking them from the front is what keeps this cheap.
    """
    import pyarrow.parquet as pq

    remote = RemoteFile(url)
    handle = pq.ParquetFile(remote, pre_buffer=True)
    per_group = handle.metadata.row_group(0).num_rows
    blobs: list[bytes] = []
    for group in range(max(1, min(handle.metadata.num_row_groups, -(-pool // per_group)))):
        rows = handle.read_row_group(group, columns=[column])[column].to_pylist()
        blobs += [r["bytes"] if isinstance(r, dict) else r for r in rows]
    print(f"  {len(blobs)} {what} in {remote.megabytes:.0f} MB")
    return [Image.open(io.BytesIO(b)).convert("RGBA") for b in blobs]


def anime_pictures(count: int, pool: int, seed: int) -> list[tuple[Image.Image, str]]:
    """Anime portraits: only ever pictures here, never points.

    A UMAP of these is one blob — raw pixels separate hair colour and background,
    not character — which is exactly the case `--images-from` is for.
    """
    print("Reading anime portraits from HuggingFace (range requests)...")
    faces = parquet_pictures(ANIME_PARQUET, "image", pool, "anime portraits")
    return [(f, "") for f in pick_by_colour(faces, count, seed)]


def cartoon_pictures(count: int, pool: int, seed: int) -> list[tuple[Image.Image, str]]:
    """Cartoon Set avatars, which ship their own alpha channel."""
    print("Reading Cartoon Set avatars from HuggingFace (range requests)...")
    faces = parquet_pictures(CARTOON_PARQUET.format(part=0), "img_bytes", pool, "avatars")
    return [(f, "") for f in pick_by_colour(faces, count, seed)]


def synthetic_pictures(count: int, pool: int, seed: int) -> list[tuple[Image.Image, str]]:
    """AI-generated portraits from SFHQ — the ones that get cut out.

    Synthetic on purpose. Publishing a mosaic built from photographs of real
    people is a consent question; a generated face is nobody, so the example can
    ship the whole recipe without one.
    """
    print("Reading synthetic portraits (SFHQ) from HuggingFace (range requests)...")
    faces = parquet_pictures(SFHQ_PARQUET, ".jpg", pool, "synthetic portraits")
    return [(f, "") for f in pick_by_colour(faces, count, seed)]


def emoji_pictures(count: int, pool: int, seed: int) -> list[tuple[Image.Image, str]]:
    """One emoji from the front of each Unicode group, taken in turn.

    Unicode order is roughly canonical order, so the first entry of a subgroup is
    the plain one — the grinning face rather than its fourteen cousins. Skin-tone
    variants and multi-codepoint sequences are dropped for the same reason.
    """
    from umap_mosaic.remote import session

    print("Reading the OpenMoji index...")
    connection = session()
    index = connection.get(OPENMOJI_INDEX, timeout=120).json()

    ranked: dict[str, list[dict]] = {}
    seen: set[tuple[str, str]] = set()
    for entry in sorted(index, key=lambda e: e.get("order") if isinstance(e.get("order"), int) else 10 ** 9):
        if entry.get("skintone") or "-" in entry["hexcode"]:
            continue
        key = (entry["group"], entry["subgroups"])
        if key in seen:
            continue
        seen.add(key)
        ranked.setdefault(entry["group"], []).append(entry)

    picked: list[dict] = []
    for depth in range(max((len(v) for v in ranked.values()), default=0)):
        for group in EMOJI_GROUPS:
            members = ranked.get(group, [])
            if depth < len(members) and len(picked) < count:
                picked.append(members[depth])
        if len(picked) >= count:
            break

    pictures = []
    for entry in picked:
        answer = connection.get(OPENMOJI_PNG.format(code=entry["hexcode"]), timeout=60)
        answer.raise_for_status()
        pictures.append((Image.open(io.BytesIO(answer.content)).convert("RGBA"),
                         entry["annotation"].replace(" ", "-").lower()))
    print(f"  {len(pictures)} emoji from {len({e['group'] for e in picked})} groups")
    return pictures


PICTURES = {
    "faces": {"fetch": synthetic_pictures, "finish": "cutout", "group": "person", "stem": "face"},
    "anime": {"fetch": anime_pictures, "finish": "feather", "group": "anime", "stem": "face"},
    "emoji": {"fetch": emoji_pictures, "finish": "alpha", "group": "emoji", "stem": "emoji"},
    "cartoon": {"fetch": cartoon_pictures, "finish": "alpha", "group": "cartoon", "stem": "avatar"},
}


def cities(limit: int) -> Source:
    """Cities of over 15,000 people, at their real longitude and latitude.

    The point of this one is that no algorithm made the coordinates. It is the same
    painter working on a scatter that happens to be a world map, and it exercises
    exactly the same placement rules: Europe and India are dense enough to hold an
    image, the Pacific is not.
    """
    import io as _io
    import zipfile

    from umap_mosaic.remote import session

    print("Fetching GeoNames cities15000 (~3 MB)...")
    answer = session().get(GEONAMES_CITIES, timeout=180)
    answer.raise_for_status()
    with zipfile.ZipFile(_io.BytesIO(answer.content)) as bundle:
        with bundle.open("cities15000.txt") as handle:
            frame = pd.read_csv(handle, sep="\t", header=None, dtype=str,
                                usecols=[1, 4, 5, 8, 14],
                                names=["name", "lat", "lon", "country", "population"])

    frame = frame.dropna(subset=["lat", "lon"])
    lat = frame["lat"].astype(float).to_numpy()
    lon = frame["lon"].astype(float).to_numpy()
    if limit and limit < len(frame):
        frame, lat, lon = frame.iloc[:limit], lat[:limit], lon[:limit]
    print(f"  {len(frame):,} cities")

    # Latitude grows north and the layout's y grows downward, so negate it or the
    # world arrives upside down.
    coords = np.column_stack([lon, -lat])
    extras = pd.DataFrame({
        "city": frame["name"].to_numpy(),
        "country": frame["country"].to_numpy(),
        "population": pd.to_numeric(frame["population"], errors="coerce").fillna(0).astype(int),
    })
    return Source(vectors=coords, group="place", mode="photo", coords=coords, extras=extras)


def gather(name: str, limit: int, seed: int, pool: int) -> Source:
    if name == "cities":
        return cities(limit)

    if name == "cartoon":
        return cartoon(limit, pool)

    if name == "digits":
        bunch = load_digits()
        return Source(bunch.data, "digit", "ink", bunch.target, DIGIT_NAMES, bunch.images / 16.0)

    if name in ("mnist", "fashion", "kmnist"):
        which, names, group, mode = {
            "mnist": ("mnist_784", DIGIT_NAMES, "digit", "ink"),
            "kmnist": ("Kuzushiji-MNIST", KUZUSHIJI_NAMES, "character", "ink"),
            "fashion": ("Fashion-MNIST", GARMENT_NAMES, "garment", "silhouette"),
        }[name]
        size = {"mnist": "~15 MB", "kmnist": "~18 MB"}.get(name, "~30 MB")
        print(f"Fetching {which} ({size}) from OpenML; scikit-learn caches it under ~/scikit_learn_data.")
        bunch = fetch_openml(which, version=1, as_frame=False, parser="auto")
        data, target = bunch.data.astype(np.float32), bunch.target.astype(int)
        if limit and limit < len(data):
            pick = np.random.default_rng(seed).choice(len(data), limit, replace=False)
            data, target = data[pick], target[pick]
        return Source(data, group, mode, target, names, data.reshape(-1, 28, 28) / 255.0)

    if name == "lfw":
        print("Fetching Labeled Faces in the Wild (~200 MB on first run); scikit-learn caches it.")
        bunch = fetch_lfw_people(min_faces_per_person=0, color=True, resize=0.5)
        data, target = bunch.data.astype(np.float32), bunch.target.astype(int)
        names = [n.replace(" ", "-").lower() for n in bunch.target_names]
        images = bunch.images
        if limit and limit < len(data):
            pick = np.random.default_rng(seed).choice(len(data), limit, replace=False)
            data, target, images = data[pick], target[pick], images[pick]
        return Source(data, "person", "photo", target, names, images)

    print("Fetching the Olivetti face database (~4.5 MB); scikit-learn caches it.")
    bunch = fetch_olivetti_faces(shuffle=False, random_state=seed)
    names = [f"subject-{i + 1:02d}" for i in range(40)]
    return Source(bunch.data, "subject", "photo", bunch.target, names, bunch.images)


# --------------------------------------------------------------------------
# Tiles
# --------------------------------------------------------------------------
def lfw_portrait(display_name: str) -> Image.Image | None:
    """The most typical original JPEG for one person, straight off disk.

    scikit-learn hands back small resized arrays; the cached originals are 250x250,
    which is what a tile actually wants. Picking the medoid avoids the odd pose.
    """
    folder = LFW_HOME / display_name.replace(" ", "_")
    shots = sorted(folder.glob("*.jpg"))
    if not shots:
        return None
    if len(shots) == 1:
        return Image.open(shots[0]).convert("RGB")
    thumbs = np.stack([
        np.asarray(Image.open(s).convert("L").resize((48, 48)), dtype=np.float32) for s in shots
    ])
    centre = thumbs.mean(axis=0)
    return Image.open(shots[int(np.argmin(((thumbs - centre) ** 2).sum(axis=(1, 2))))]).convert("RGB")


def medoid(images: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """The most typical member of a class, which beats the blurry class mean."""
    members = images[mask]
    centre = members.mean(axis=0)
    return members[np.argmin(((members - centre) ** 2).sum(axis=(1, 2)))]


def spread_over_map(coords: np.ndarray, count: int, pool: int, seed: int) -> list[int]:
    """One representative row per region of the map, when there are no classes.

    k-means on the layout carves the cloud into as many regions as there are
    tiles; each tile is then the candidate nearest its region's centre. Candidates
    are limited to the first `pool` rows because fetching an arbitrary row from a
    remote columnar file means fetching the whole row group around it.
    """
    cells = KMeans(count, n_init=10, random_state=seed).fit(coords)
    reachable = np.arange(min(pool, len(coords)))
    chosen: list[int] = []
    for centre in cells.cluster_centers_:
        gap = ((coords[reachable] - centre) ** 2).sum(axis=1)
        for candidate in np.argsort(gap):
            row = int(reachable[candidate])
            if row not in chosen:
                chosen.append(row)
                break
    return chosen


def resize_rgba(image: Image.Image, size: int) -> Image.Image:
    """Downscale without dragging the transparent background into the edges.

    Resizing RGBA directly averages colour across the alpha boundary, so a face
    on transparent black picks up a dark rim. Weighting by alpha first — a
    premultiplied resize — keeps the edge the colour of the subject.
    """
    source = np.asarray(image.convert("RGBA"), dtype=np.float32)
    alpha = source[..., 3:4] / 255.0
    premultiplied = np.dstack([source[..., :3] * alpha, source[..., 3]])
    small = np.asarray(
        Image.fromarray(premultiplied.astype(np.uint8), "RGBA")
        .resize((size, size), Image.Resampling.LANCZOS), dtype=np.float32)
    weight = np.maximum(small[..., 3:4] / 255.0, 1e-4)
    straight = np.clip(small[..., :3] / weight, 0, 255)
    return Image.fromarray(np.dstack([straight, small[..., 3]]).astype(np.uint8), "RGBA")


def feather(image: Image.Image, size: int, edge: float = 0.22) -> Image.Image:
    """Give a picture with no alpha channel a soft edge instead of a hard square.

    Cutting these out properly is not on: GrabCut is a colour model built for
    photographs of people, and on an anime crop it takes the face and throws the
    hair away. A square is honest, but a square *edge* is not — it reads as a
    sticker on the map. Fading the outer fifth to nothing lets the picture end in
    its region's colour, which is what the rest of the mosaic does too.
    """
    art = np.asarray(image.convert("RGB").resize((size,) * 2, Image.Resampling.LANCZOS))
    axis = np.linspace(0, 1, size, dtype=np.float32)
    ramp = np.clip(np.minimum(axis, 1 - axis) / max(edge, 1e-6), 0, 1)
    ramp = ramp * ramp * (3 - 2 * ramp)  # smoothstep, so the fade has no seam
    alpha = (np.outer(ramp, ramp) * 255).astype(np.uint8)
    return Image.fromarray(np.dstack([art, alpha]), "RGBA")


def fit_alpha(picture: Image.Image, size: int, margin: float = 0.05) -> Image.Image:
    """Crop away the transparent margin before resizing, keeping the square square.

    A tile's square is the whole budget: every row of empty margin the source
    shipped with is a row of dots that paint nothing. Emoji arrive with a wide one
    and a cut-out portrait is mostly air, so both come out half the size they
    could be. The crop is squared up around its own centre rather than stretched,
    so nobody ends up oval.
    """
    rgba = picture.convert("RGBA")
    box = rgba.getchannel("A").point(lambda v: 255 if v > 8 else 0).getbbox()
    if box is None:
        return resize_rgba(rgba, size)
    left, top, right, bottom = box
    side = max(right - left, bottom - top)
    side = int(side * (1 + 2 * margin))
    cx, cy = (left + right) // 2, (top + bottom) // 2
    square = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    square.paste(rgba, (side // 2 - cx, side // 2 - cy))
    return resize_rgba(square, size)


def cut_face(picture: Image.Image, size: int, grow: float = 0.06, rounds: int = 6) -> Image.Image:
    """The same cut-out `extract_faces.py` runs, on a picture already square."""
    square = pad_to_frame(picture).resize((size, size), Image.Resampling.LANCZOS)
    subject, fell_back = cut_out(square, grow=grow, iterations=rounds)
    if fell_back:
        print("  note: a portrait fell back to the template shape")
    return apply_cutout(square, subject, feather=0.006)


def finish_picture(picture: Image.Image, how: str, size: int) -> Image.Image:
    """Turn a fetched picture into a tile: cut out, feathered, or left alone."""
    if how == "feather":
        # The alpha here *is* the fade, so there is no margin to crop.
        return feather(picture, size)
    if how == "cutout":
        return fit_alpha(cut_face(picture, size * 2), size)
    return fit_alpha(picture, size)


def write_tile(image: np.ndarray, path: Path, size: int, mode: str) -> None:
    """Save a tile as RGBA, cut out according to what the subject actually is.

    `ink`        thin bright strokes on black: the intensity *is* the subject, so it
                 becomes the alpha over a flat dark colour.
    `silhouette` a solid object on black: the alpha has to be the object's *outline*,
                 not its brightness — keying on brightness would punch holes through
                 every dark part of a boot and leave the shape unreadable.
    `photo`      already a picture; keep it opaque.
    """
    art = np.clip(image, 0, 1)
    if mode == "ink":
        alpha = (art ** 0.8 * 255).astype(np.uint8)
        rgba = np.dstack([np.full(art.shape + (3,), 38, np.uint8), alpha])
    else:
        grey = (art * 255).astype(np.uint8)
        if mode == "silhouette":
            solid = binary_fill_holes(binary_closing(art > 0.06, np.ones((2, 2))))
            alpha = (gaussian_filter(solid.astype(np.float32), 0.6) * 255).astype(np.uint8)
        else:
            alpha = np.full(art.shape, 255, np.uint8)
        rgba = np.dstack([grey, grey, grey, alpha])
    Image.fromarray(rgba).resize((size, size), Image.Resampling.LANCZOS).save(path)


def write_tiles(source: Source, coords: np.ndarray, images_dir: Path, args: argparse.Namespace) -> int:
    """Pick the images that become tiles and write them, classes or no classes."""
    if args.images_from:
        spec = PICTURES[args.images_from]
        found = spec["fetch"](args.tiles, args.tile_pool, args.seed)
        for slot, (picture, label) in enumerate(found, start=1):
            stem = label or f"{spec['stem']}-{slot:02d}"
            finish_picture(picture, spec["finish"], args.tile_size).save(
                images_dir / f"{slot:02d}_{spec['group']}_{stem}.png")
        return len(list(images_dir.glob("*.png")))

    if source.labels is None:
        rows = spread_over_map(coords, args.tiles, args.tile_pool, args.seed)
        pictures = source.pictures(rows)
        for slot, row in enumerate(sorted(rows), start=1):
            picture = pictures.get(row)
            if picture is not None:
                fit_alpha(picture, args.tile_size).save(
                    images_dir / f"{slot:02d}_{source.group}_avatar-{slot:02d}.png")
        return len(list(images_dir.glob("*.png")))

    # A dataset with thousands of classes cannot give every one a tile, so take the
    # best-represented ones. With ten classes this keeps all ten.
    names = source.names or []
    counts = np.bincount(source.labels, minlength=len(names))
    present = [i for i in range(len(names)) if counts[i]]
    chosen = sorted(present, key=lambda i: -counts[i])[:args.tiles] if len(present) > args.tiles else present
    chosen.sort(key=lambda i: names[i])

    for slot, index in enumerate(chosen, start=1):
        stem = images_dir / f"{slot:02d}_{source.group}_{names[index]}.png"
        if source.mode == "photo" and args.dataset == "lfw":
            portrait = lfw_portrait(names[index].replace("-", " ").title())
            if portrait is not None:
                portrait.resize((args.tile_size,) * 2, Image.Resampling.LANCZOS).save(stem)
        else:
            write_tile(medoid(source.images, source.labels == index), stem, args.tile_size, source.mode)
    return len(list(images_dir.glob("*.png")))


def lay_out(vectors: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    """Features to two dimensions, by whichever method was asked for."""
    if args.layout == "pca":
        print("Taking the first two principal components...")
        return PCA(n_components=2, random_state=args.seed).fit_transform(vectors)

    if args.layout == "tsne":
        from sklearn.manifold import TSNE

        print(f"Running t-SNE (perplexity={args.neighbors * 2})... this is the slow one.")
        return TSNE(n_components=2, perplexity=args.neighbors * 2, init="pca",
                    random_state=args.seed, verbose=0).fit_transform(vectors)

    print(f"Running UMAP (n_neighbors={args.neighbors}, min_dist={args.min_dist})...")
    return UMAP(n_components=2, n_neighbors=args.neighbors, min_dist=args.min_dist,
                random_state=args.seed, verbose=False).fit_transform(vectors)


def main() -> None:
    args = arguments()
    name = f"{args.dataset}-{args.images_from}" if args.images_from else args.dataset
    if args.layout != "umap" and args.dataset != "cities":
        name = f"{args.layout}-{name}"
    out = args.out or ROOT / "examples" / name
    images_dir = out / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    source = gather(args.dataset, args.limit, args.seed, args.tile_pool)
    vectors = source.vectors
    classes = "unlabelled" if source.labels is None else f"{len(set(source.labels))} classes"
    print(f"{len(vectors):,} points, {classes}, {vectors.shape[1]} features")

    if source.coords is not None:
        coords = source.coords
        print(f"Coordinates come with the data; no {args.layout} step.")
    else:
        # PCA first on wide inputs: standard practice, and it turns a multi-minute
        # layout on raw pixels into well under one.
        if vectors.shape[1] > 60:
            components = min(50, vectors.shape[1], len(vectors) - 1)
            print(f"PCA {vectors.shape[1]} -> {components} dimensions...")
            vectors = PCA(n_components=components, random_state=args.seed).fit_transform(vectors)
        else:
            # Attribute tables come in arbitrary units; else the widest column wins.
            vectors = (vectors - vectors.mean(0)) / (vectors.std(0) + 1e-9)
        coords = lay_out(vectors, args)

    frame = pd.DataFrame({
        "id": [f"{args.dataset}-{i}" for i in range(len(coords))],
        "x": np.round(coords[:, 0], 5),
        "y": np.round(coords[:, 1], 5),
    })
    if source.labels is not None and source.names is not None:
        frame["label"] = [source.names[int(v)] for v in source.labels]
    if source.extras is not None:
        frame = pd.concat([frame, source.extras.iloc[:len(frame)]], axis=1)
    frame.to_csv(out / "embedding.csv", index=False)

    written = write_tiles(source, coords, images_dir, args)
    for note in source.notes:
        print(f"  note: {note}")
    print(f"Wrote {out / 'embedding.csv'} and {written} tiles to {images_dir}")

    # Fewer images can afford bigger squares; where the map's shape has been
    # measured the tuned flags are better than that guess.
    tuned = TUNING.get(args.dataset, {}).get("build")
    flags = " ".join(tuned) if tuned else (
        f"--face-size {max(0.05, min(0.18, 0.45 / max(written, 1) ** 0.5)):.2f}")
    print("\nNext:")
    print(f"  python scripts/build_mosaic.py --embedding {out / 'embedding.csv'} "
          f"--images {images_dir} --out {out / 'mosaic'} {flags}")


if __name__ == "__main__":
    main()
