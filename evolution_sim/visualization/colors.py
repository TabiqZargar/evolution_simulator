"""Colour helpers and palettes (pure functions, no pygame state).

Colors are plain tuples because the rest of the visualization layer blits
them through pygame. Everything here is deterministic given an input, which
keeps rendering cheap and testable.
"""

from __future__ import annotations

from evolution_sim.simulation.environment import Biome

RGB = tuple[int, int, int]

# Terrain / biome palette -------------------------------------------------
BIOME_COLORS: dict[Biome, RGB] = {
    Biome.WATER: (31, 64, 110),
    Biome.BARREN: (158, 132, 92),
    Biome.GRASS: (96, 150, 74),
    Biome.FOREST: (44, 96, 52),
}

WATER_COLOR = (31, 64, 110)
GRASS_COLOR = (96, 150, 74)
FOOD_COLOR = (235, 226, 138)
FOOD_DEPLETED_COLOR = (70, 84, 50)

# Trait gradient anchors (viridis-ish). ------------------------------------
VALUE_ANCHORS: list[tuple[float, RGB]] = [
    (0.0, (68, 1, 84)),
    (0.25, (59, 82, 139)),
    (0.5, (33, 145, 140)),
    (0.75, (94, 201, 98)),
    (1.0, (253, 231, 37)),
]


def trait_color(value: float) -> RGB:
    """Map a trait value on [0, 1] onto the viridis gradient."""
    value = min(max(value, 0.0), 1.0)
    for i in range(len(VALUE_ANCHORS) - 1):
        lo_t, lo_c = VALUE_ANCHORS[i]
        hi_t, hi_c = VALUE_ANCHORS[i + 1]
        if lo_t <= value <= hi_t:
            span = max(hi_t - lo_t, 1e-9)
            t = (value - lo_t) / span
            r = int(lo_c[0] + (hi_c[0] - lo_c[0]) * t)
            g = int(lo_c[1] + (hi_c[1] - lo_c[1]) * t)
            b = int(lo_c[2] + (hi_c[2] - lo_c[2]) * t)
            return (r, g, b)
    return VALUE_ANCHORS[-1][1]


def species_color(species_id: int) -> RGB:
    """Deterministic hue for a species id (golden-angle spread)."""
    if species_id <= 0:
        return (180, 180, 180)
    hue = (species_id * 0.618033988749895) % 1.0
    return _hsb_to_rgb(hue, 0.65, 0.9)


def _hsb_to_rgb(hue: float, sat: float, val: float) -> RGB:
    i = int(hue * 6)
    f = hue * 6 - i
    p = val * (1 - sat)
    q = val * (1 - f * sat)
    t = val * (1 - (1 - f) * sat)
    i %= 6
    r, g, b = (
        (val, t, p),
        (q, val, p),
        (p, val, t),
        (p, q, val),
        (t, p, val),
        (val, p, q),
    )[i]
    return (int(r * 255), int(g * 255), int(b * 255))


def fade(color: RGB, alpha: float) -> tuple[int, int, int, int]:
    return (color[0], color[1], color[2], int(min(max(alpha, 0.0), 1.0) * 255))


# Neutral UI palette -------------------------------------------------------
BACKGROUND = (13, 17, 23)
PANEL_BG = (24, 30, 40)
PANEL_BORDER = (58, 70, 90)
TEXT = (215, 222, 232)
TEXT_DIM = (140, 152, 168)
ACCENT = (110, 190, 255)
WARN = (240, 130, 90)
GOOD = (120, 210, 130)
