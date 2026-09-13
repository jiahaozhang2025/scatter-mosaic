"""Colour schemes, and the palettes built for each background.

Swapping a background is not enough on its own. A palette tuned for a dark page
glares on a white one, the pale entries vanish, and "fading a colour back" means
the opposite direction in each case. So each theme carries its own palette, its
own chrome, and its own defaults for how far colours fade.

The light palette is twelve hues at two lightness levels, every colour held
between 3.5:1 and 6.4:1 contrast against white and no two closer than dE 14, so
a one-pixel dot still reads.
"""

from __future__ import annotations

THEMES: dict[str, dict] = {
    "light": {
        "background": "#ffffff",
        "ink": "#15141b",
        "muted": "#6a6572",
        "panel": "#f5f5f8",
        "line": "#e2e1e8",
        "accent": "#b8860b",
        "on_accent": "#ffffff",
        "stage": ["#ffffff", "#fbfbfd", "#f4f4f8"],
        # Fading toward white bleaches a hue much faster than fading toward black,
        # so the light theme keeps more colour and leans harder on the photo.
        "photo_lift": -0.10,
        "photo_alpha": 0.90,
        "ghost": {"scatter": 0.70, "cluster": 0.62},
        "palette": [
            "#2a8bec", "#fa4632", "#239e37", "#d16f19", "#b663ed", "#b27d62",
            "#de55ac", "#1996a9", "#8b8e2b", "#2e997f", "#d651d6", "#8282c8",
            "#0e5fb0", "#b82312", "#0d6e1d", "#964a08", "#853ab6", "#83533b",
            "#a52e7a", "#066978", "#606314", "#156b56", "#9d2c9d", "#57579f",
        ],
    },
    "dark": {
        "background": "#09080e",
        "ink": "#f6f0e8",
        "muted": "#aaa6b2",
        "panel": "#16121e",
        "line": "#2a2635",
        "accent": "#ffcf5c",
        "on_accent": "#17131f",
        "stage": ["#20182e", "#100d18", "#08070b"],
        "photo_lift": 0.35,
        "photo_alpha": 0.70,
        "ghost": {"scatter": 0.30, "cluster": 0.45},
        "palette": [
            "#ff6b6b", "#4ecdc4", "#ffe66d", "#5b8ff9", "#a66cff", "#ff9f43",
            "#2ed573", "#ff6fb5", "#48dbfb", "#f368e0", "#10ac84", "#ee5253",
            "#54a0ff", "#c8d6e5", "#feca57", "#1dd1a1", "#ff9ff3", "#00d2d3",
            "#ff7f50", "#7bed9f", "#70a1ff", "#eccc68", "#ff4757", "#5352ed",
        ],
    },
}

ACTIVE = THEMES["light"]
PALETTE = ACTIVE["palette"]


def use_theme(name: str) -> dict:
    """Switch the palette and chrome colours every renderer reads from."""
    global ACTIVE, PALETTE
    if name not in THEMES:
        raise ValueError(f"Unknown theme {name!r}; pick one of {sorted(THEMES)}")
    ACTIVE = THEMES[name]
    PALETTE = ACTIVE["palette"]
    return ACTIVE


def color_for(index: int, theme: str | None = None) -> str:
    """The palette colour for a region, cycling if there are more regions than hues."""
    palette = THEMES[theme]["palette"] if theme else PALETTE
    return palette[index % len(palette)]


def rgb(value: str) -> tuple[int, int, int]:
    return tuple(bytes.fromhex(value.lstrip("#")))


def theme_rgb(key: str) -> tuple[int, int, int]:
    return rgb(ACTIVE[key])
