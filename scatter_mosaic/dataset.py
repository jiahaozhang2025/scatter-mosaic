"""Read any 2-D embedding into the one shape the renderers expect.

The only hard requirement is a CSV with two coordinate columns. Everything else
is optional: an identifier column if you have one, a boolean column saying which
rows to keep, and any number of extra columns, which are carried through and
shown when a point is clicked.

Column names are guessed from the usual conventions (`x`/`y`, `UMAP1`/`UMAP2`,
`tsne_1`/`tsne_2`, …) and can always be given explicitly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

X_PATTERNS = (r"^x$", r"^(umap|tsne|t-sne|pca|pc|dim|comp\w*)[\s_-]?0?1$", r"^\w+_x$", r"^x_\w+$")
Y_PATTERNS = (r"^y$", r"^(umap|tsne|t-sne|pca|pc|dim|comp\w*)[\s_-]?0?2$", r"^\w+_y$", r"^y_\w+$")
ID_PATTERNS = (r"^id$", r"^\w*_?id$", r"^index$", r"^name$", r"^label$", r"^barcode$")
KEEP_PATTERNS = (r"^included\w*$", r"^keep$", r"^use$", r"^valid$", r"^passed?\w*$")


def _match(columns: list[str], patterns: tuple[str, ...]) -> str | None:
    for pattern in patterns:
        for column in columns:
            if re.match(pattern, column.strip().lower()):
                return column
    return None


@dataclass
class Embedding:
    """A tidy 2-D embedding: identifiers, coordinates, and whatever else came along."""

    ids: pd.Series
    x: np.ndarray
    y: np.ndarray
    extras: pd.DataFrame = field(default_factory=pd.DataFrame)
    source: str = ""

    def __len__(self) -> int:
        return len(self.x)

    def tooltip_fields(self, limit: int = 6) -> list[dict]:
        """Extra columns rendered as `label: value` when a point is clicked."""
        fields = []
        for column in list(self.extras.columns)[:limit]:
            values = self.extras[column]
            if pd.api.types.is_numeric_dtype(values):
                decimals = 0 if float(np.nanmax(np.abs(values.to_numpy()))) >= 100 else 3
                shown = values.round(decimals).tolist()
            else:
                shown = values.astype(str).tolist()
            fields.append({"label": column.replace("_", " "), "values": shown})
        return fields


def load_embedding(
    path: Path | str,
    x_column: str | None = None,
    y_column: str | None = None,
    id_column: str | None = None,
    keep_column: str | None = None,
    require: tuple[str, ...] = (),
    max_extras: int = 6,
) -> Embedding:
    """Read `path` and normalize it, guessing column names where not given."""
    path = Path(path)
    frame = pd.read_csv(path)
    columns = list(frame.columns)

    x_column = x_column or _match(columns, X_PATTERNS)
    y_column = y_column or _match(columns, Y_PATTERNS)
    if x_column is None or y_column is None:
        raise SystemExit(
            f"Could not find coordinate columns in {path.name} (saw {columns}). "
            "Pass --x-column and --y-column."
        )

    id_column = id_column or _match([c for c in columns if c not in (x_column, y_column)], ID_PATTERNS)
    keep_column = keep_column or _match(columns, KEEP_PATTERNS)

    rows = frame[x_column].notna() & frame[y_column].notna()
    if keep_column is not None and keep_column in frame:
        rows &= frame[keep_column].astype(bool)
    frame = frame.loc[rows].copy()
    if frame.empty:
        raise SystemExit(f"No usable rows in {path.name} after filtering.")

    ids = (
        frame[id_column].astype(str)
        if id_column and id_column in frame
        else pd.Series([str(i) for i in range(len(frame))], index=frame.index)
    )
    spent = {x_column, y_column, id_column, keep_column} - {None}
    rest = frame.drop(columns=[c for c in spent if c in frame])
    # Columns the caller depends on are kept whatever the extras budget says.
    wanted = [c for c in require if c in rest]
    extras = pd.concat([rest[wanted], rest.drop(columns=wanted).iloc[:, :max_extras]], axis=1)

    return Embedding(
        ids=ids.reset_index(drop=True),
        x=frame[x_column].to_numpy(dtype=np.float64),
        y=frame[y_column].to_numpy(dtype=np.float64),
        extras=extras.reset_index(drop=True),
        source=f"{path.name} · {x_column}/{y_column}"
        + (f" · filtered on {keep_column}" if keep_column else ""),
    )


GENERIC_NAMES = {
    "embedding", "embeddings", "umap", "tsne", "pca", "coords", "coordinates",
    "data", "output", "out", "results", "result", "mosaic", "poster", "examples",
    "regions", "region", "patches", "image map", "images",
}


def dataset_name(path: Path | str, fallback: str = "embedding") -> str:
    """A human name for a dataset, from the first non-generic part of its path.

    `examples/fashion/embedding.csv` and `examples/fashion/mosaic/regions.csv`
    both answer "fashion", so the page and the poster carry the same headline.
    """
    here = Path(path).resolve()
    for part in (here.stem, *(p.name for p in here.parents)):
        cleaned = re.sub(r"[_-]+", " ", part).strip()
        if cleaned and cleaned.lower() not in GENERIC_NAMES:
            return cleaned
    return fallback
