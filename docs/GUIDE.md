# Guide

Reference for Scatter Mosaic. The [README](../README.md) is the short version.

- [Three independent choices](#three-independent-choices)
- [Where each image goes](#where-each-image-goes)
- [Controls in the page](#controls-in-the-page)
- [Colours](#colours)
- [Cutting images out](#cutting-images-out)
- [Your own data](#your-own-data)
- [The examples](#the-examples)
- [Project layout](#project-layout)
- [Optional: compacting the layout](#optional-compacting-the-layout)
- [Making thin data paintable](#making-thin-data-paintable)
- [Known limits](#known-limits)
- [Credits](#credits)

## Three independent choices

|  |  |
|---|---|
| **Layout** | `--layout umap` (default), `tsne` or `pca` turns features into two dimensions. `cities` skips it: those coordinates are longitude and latitude. Nothing downstream can tell the difference — the painter only ever sees two columns of numbers, which is why a PCA cloud with no clusters at all works as well as a UMAP with ten. |
| **Placement** | `scatter` (default) packs equal squares onto the parts of the cloud that can support them, then gives each point to its nearest image. No clustering involved, and every image gets a home. `cluster` fits one image to each group's silhouette instead — good when your points already carry a label. |
| **Style** | `paint` (default) colours every point's own dot from the image, so the picture is made of data and nothing is hidden. `overlay` lays the image over the dots. |

Placement is the one build-time decision. Both styles, both themes and every slider
live inside the page, so exploring costs nothing.

A dot is painted or it is not — it cannot be half painted — so a soft edge in the
source image is spent as *probability*: half-opaque ground keeps half of its dots,
and the edge dissolves at dot scale instead of ending on a cut line. The choice is
hashed from each point's own index, so the viewer and the poster agree, every run.

## Where each image goes

`scatter_mosaic/placement.py` packs one equal square per image:

1. Score every square position. `--face-fill` (0.85) is the share of the square the
   cloud covers, which rejects holes; `--face-density` (0.55) keeps an image off
   ground too thin for it to read — measured as a fraction of *this* cloud's own
   typical density, so 1,800 points and 70,000 need the same number.
2. Take them best-supported first. A square that can sit **flush** against one
   already placed gets a `--face-adjacency` (0.6) bonus, which is what makes images
   tile into the roomy parts of the map rather than each sitting alone in the middle
   of its own territory.
3. A group stops accepting neighbours at `--face-group` (4), so one dense corner
   cannot swallow the whole set while a large region still holds three or four.
4. If `--face-size` (0.055 of the layout) will not seat everyone, shrink and retry —
   strictest ground rules first, so a smaller image on solid ground always beats a
   larger one on haze.

Then every point joins its nearest image. The regions exist because the images do,
which is the point: tie images to clusters instead and anything whose cluster is too
small vanishes from the artwork.

**Shape matters more than any of those numbers.** A square cannot fill a circle, so
a map of round islands needs `--face-fill` relaxed — around 0.72 for MNIST's curved
masses, 0.62 for Cartoon Set's discs — before a square is allowed to overhang the
curve. Leave it at 0.85 and every image shrinks to the inscribed square, and the map
reads as coloured blobs with stamps on them. Relax it too far and a face spills past
its island, which breaks the one promise the picture makes.

Placement is deterministic given the layout and `--seed`. The builder writes it to
`patches.json` and the poster reads that file, so the two renders always agree.

## Controls in the page

| Control | Effect |
|---|---|
| **Painted dots / Image overlay** | The two styles. |
| **Light / Dark** | Swaps background *and* palette. Not a filter — each theme has its own colours. |
| **Image %** | How much image. At 0 you get the bare embedding in region colours. |
| **Dots ×** | Dot size. Smaller resolves finer structure; larger reads from further away. |
| **Reset view, All regions, Export PNG** | Click a legend image to isolate one region; click a point for its values. |

## Colours

Each theme carries its own palette and defaults, because the two backgrounds do not
behave the same way:

- **Palette.** The light palette is twelve hues at two lightness levels, every colour
  held between 3.5:1 and 6.4:1 contrast against white and no two closer than ΔE 14,
  so a one-pixel dot still reads. The dark palette's neon hues would glare on paper.
- **Fading.** Points outside an image fade toward the *page*, not toward black.
  Fading toward white bleaches a hue much faster, so light keeps more colour
  (`--ghost` 0.70 vs 0.30) and leans harder on the image (`--photo` 0.90 vs 0.70).
- **Dots.** The poster's dot plate is drawn on the page colour so anti-aliasing
  blends toward it; a black plate would ring every dot on a white page.

## Cutting images out

Only meaningful for photographs of people. `--images-from faces` does it for you;
this is the same code (`scatter_mosaic/cutout.py`) pointed at a folder of your own:

```bash
python scripts/extract_faces.py --input images --output images_cutout
```

It finds the face (OpenCV's bundled Haar cascades, no download), re-crops around it,
and removes the background with GrabCut seeded by a head-and-shoulders template.
Output is RGBA, which is what lets the painted style colour only the points that land
on a subject rather than on somebody's office wall. A review grid shows each cut-out
over a checkerboard with the share of the frame it kept; anything suspicious is
reported rather than shipped quietly.

Two details decide whether a cut-out is any good:

- **Pad a tight crop first.** An aligned portrait puts the hair against the frame,
  exactly where the trimap marks definite background, and the subject comes back with
  a haircut. Pad with the crop's own border colour, not a replicated edge —
  replication drags hair-coloured streaks into the new space and the colour model
  keeps them.
- **Crop back to the alpha afterwards.** A tile's square is the whole budget, and a
  cut-out head is mostly air; without this the subject lands at half the size the
  square could hold.

GrabCut is a colour model, so it fails when the background shares the subject's
palette — and it fails badly on drawings, where it will take an anime face and throw
the hair away. Those cases are flagged. Pictures that cannot be cut out are better
feathered than squared off, which is what `--images-from anime` does. For better
detection recall on turned heads, drop a YuNet ONNX file
(`face_detection_yunet_*.onnx` from opencv_zoo) in the repo.

## Your own data

Two coordinate columns is the only requirement:

```csv
id,x,y,label,score
cell-0,3.71,-8.20,neuron,0.94
```

- Coordinates are guessed from the usual names (`x`/`y`, `UMAP1`/`UMAP2`,
  `tsne_1`/`tsne_2`, `PC1`/`PC2`, …). Override with `--x-column` / `--y-column`.
- An id column is guessed too, or generated. `--keep-column` names a boolean column
  saying which rows to use.
- Any other columns are carried through and shown when you click a point.

Images come from a folder. A filename like `07_postdoc_won-jung.jpg` gives an order,
a group label and a display name; anything else falls back to the filename. The
region-to-image table is written to `image_map.csv` next to the output — edit it to
change which picture goes where, and rebuild.

## The examples

Points (`--dataset`):

| | | |
|---|---|---|
| `mnist` | 70,000 handwritten digits | ~15 MB. The default, and the best-looking map: long curved islands with wisps and bridges between them. |
| `fashion` | 70,000 garments, 10 classes | ~30 MB. Fewer, fatter masses, which suits images that tile across a shape. Wants `--face-size 0.14`. |
| `kmnist` | 70,000 Kuzushiji characters | ~18 MB. Cursive Japanese; the classes overlap more than digits do, so the map comes out stringier and more tangled. |
| `cartoon` | 100,000 avatars from Google's Cartoon Set | ~25 MB. Embedded on the eighteen attributes that generated each face. Eight round islands — very clean, and duller for it. Wants `--face-size 0.16 --face-fill 0.62`. |
| `lfw` | 13,233 photographs of 5,749 people | ~200 MB. Fair warning: a UMAP of raw pixels separates pose and lighting, not identity, so the map is close to one blob. |
| `digits` | 1,797 digits, 8×8 | Bundled with scikit-learn, no download. |
| `olivetti` | 400 faces, 40 people | ~4.5 MB. |
| `cities` | ~70,000 cities over 5,000 people | ~5 MB from GeoNames. Longitude and latitude, `--layout` ignored. Coastlines are thin and ragged, so this one wants *many small* pictures on *many* points: `--densify 5 --tiles 60 --face-size 0.030 --face-fill 0.35`. Fewer, larger pictures have nowhere on a coastline to sit. |
| `stars` | ~100,000 stars as an HR diagram | ~32 MB from the HYG catalogue. Colour index against absolute magnitude, both axes standardized because they are in unrelated units. One broad diagonal band — the main sequence — with the giant branch as a spur off the top. Wants `--limit 40000 --densify 2 --face-size 0.20 --face-fill 0.72`. |
| `quakes` | ~25,000 earthquakes of M2.5+ over a year | ~4 MB from USGS, paged because one query is capped at 20,000. A beautiful map and a poor mosaic — see below. |

Layouts (`--layout`), for the datasets that need one:

| | |
|---|---|
| `umap` | The default. Tight islands with wisps and bridges between them. |
| `tsne` | Rounder, more evenly sized blobs, packed closer together. Much slower — budget minutes, and consider `--limit 25000`. |
| `pca` | One continuous cloud, no clusters whatsoever. Worth trying precisely because it has no structure to lean on: the images tile edge to edge across the densest part and it still reads. |

Pictures (`--images-from`) are listed in the [README](../README.md). All four choose
their tiles by average colour, so no two look alike from across the room.

The cartoon example is the one worth reading the code for. Its embedding is of the
*attributes*, not the pixels — a UMAP of cartoon pixels comes out as thin wisps,
while the attributes give solid islands, and the attribute columns are one megabyte
against the images' 488. And it never downloads that file: `scatter_mosaic/remote.py`
turns a URL into something `pyarrow` can seek inside, so a 489 MB parquet is read one
column and one row group at a time.

## Project layout

```
scatter_mosaic/        the library
  dataset.py        read any embedding CSV
  placement.py      pack equal squares, hand out regions
  painting.py       colour the dots from an image
  silhouette.py     cluster shapes and the warp onto them
  cutout.py         take the background off a portrait
  layout.py         isotropic normalization, optional compaction
  remote.py         range-read a remote columnar file instead of downloading it
  theme.py          palettes and chrome
  template.html     the viewer
scripts/            command-line entry points
notebooks/          walkthrough.ipynb
```

Run any script with `--help` for its full flag list.

**Notebook kernel.** `walkthrough.ipynb` needs an interpreter with these
requirements. In VS Code or Cursor just select it. For JupyterLab, register a kernel
bound to it by absolute path first — the stock `python3` kernelspec has a bare
`python` in its `argv`, which jupyter_client rewrites to whichever interpreter runs
the Jupyter *server*:

```bash
/path/to/python -m ipykernel install --user --name umap-mosaic --display-name "UMAP mosaic"
```

## Optional: compacting the layout

UMAP spends a lot of canvas on the space *between* clusters. `--compact 1.0` slides
each group inward until it nearly touches its neighbours, closing that space to about
0.77× the original linear extent. Groups move rigidly, so internal structure is
untouched, but it does change where they sit relative to one another — a choice for
the artwork, not a claim about the data. Off by default.

## Making thin data paintable

`--densify 3` simulates two extra points near every real one until the scatter can
carry a picture. It is kernel density estimation run backwards — sampling from the
KDE rather than evaluating it — with one refinement that matters: the bandwidth is
per point, set to the distance to that point's own sixth neighbour. A fixed
bandwidth blurs tight structure while barely filling sparse ground; a local one
moves a dot in a dense clump only a little and a dot on a fringe more, so the
outline stays where it was.

The world-cities example is the clearest case: at 34,000 points the avatars are
faint smudges, and at 102,000 they are legible.

**These points are not data.** They are flagged in a `simulated` column, every other
field is left blank for them, and nothing should be measured from them. The count in
the page header includes them, so say so if you publish the number.

## Known limits

**Above 24 regions the poster stops naming them** and the legend becomes a strip of
numbered thumbnails, because two rows of labelled text is all that fits.

**Point count sets the page size**, at roughly 65 bytes a point in the HTML: 40,000
points make a 1.8 MB page and 350,000 make a 20 MB one. The poster does not care.
Densify with that in mind.

**Filaments cannot hold a picture, densified or not.** `--dataset quakes` is the
honest counter-example: a year of earthquakes draws the plate boundaries beautifully,
but the arcs are one or two points wide, and at 4× density the faces still come out
as smudges strung along a line with several regions holding nothing legible at all.
Densifying a filament thickens it into a blur rather than making room. The tool needs
ground with area, which is why `cities` works and `quakes` does not.

Small source images stay soft when blown up to a large square; the painted style is
less affected than the overlay, since it only samples colours.

"Solid ground" is measured on a 256-pixel raster, so a square can still overlap a
thin bridge between two parts of the cloud. Raise `--face-density` if that shows —
though past about 1.0 (the cloud's own median) the eligible ground fragments, the
squares shrink and the images stop tiling.

## Credits

Everything is fetched at run time and none of it is redistributed here.

**Points.** MNIST, Fashion-MNIST and Kuzushiji-MNIST via OpenML; scikit-learn's
digits and the Olivetti/AT&T face database; Labeled Faces in the Wild; Google's
Cartoon Set (CC BY 4.0), read through the `cgarciae/cartoonset` mirror; the
`cities15000` gazetteer from GeoNames (CC BY 4.0); the USGS earthquake catalogue
(public domain); and the HYG star database (CC BY-SA 2.5).

**Pictures.** OpenMoji (CC BY-SA 4.0). SFHQ, which is generated rather than
photographed. The anime portraits come from the `HK83/Anime_Faces` mirror of the
widely-circulated Getchu crops; the uploader states AFL-3.0, which is a claim about
the mirror rather than clean provenance for the underlying art, so treat that one as
a demo and not as a licence to redistribute.
