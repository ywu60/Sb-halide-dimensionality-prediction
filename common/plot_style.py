"""Shared palette and axis styling for every figure produced in this folder.

Typography and the DIM_COLOR set follow Nature-family conventions: Arial/Helvetica sans-serif,
and the Okabe-Ito colorblind-safe palette (Wong, B. "Points of view: Color blindness." Nat Methods
8, 441 (2011) -- https://www.nature.com/articles/nmeth.1618), the de facto standard categorical
palette in Nature Portfolio journals. The remaining colors are the validated default
categorical/sequential/diverging palette from the dataviz skill (references/palette.md), checked
with scripts/validate_palette.js -- all adjacent-pair checks pass in light mode. One shared module
keeps every PNG (EDA, ML, LLM, comparison) reading as one system.
"""
import matplotlib
# All project figures are written to files.  A non-interactive backend avoids macOS
# GUI/font-cache failures when the plotting scripts run headlessly.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# DejaVu Sans ships with Matplotlib.  Put it first so figures also render reliably on
# machines without the macOS Helvetica/Arial font-cache directories.
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]
matplotlib.rcParams["axes.linewidth"] = 0.8

BG, INK, INK_SOFT, GRID, AXIS = "#ffffff", "#0b0b0b", "#52514e", "#e3e2dc", "#c7c6bf"

# Validated categorical order (blue, orange, aqua, yellow, violet, red -- adjacent CVD-safe).
BLUE, ORANGE, AQUA, YELLOW, VIOLET, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7", "#e34948"

# Okabe-Ito colorblind-safe set (orange / blue / bluish-green / vermillion), skipping the
# palette's yellow -- too low-contrast on a white ground for a thin stacked segment (this is
# exactly the 3D sliver that needed a visual-thickness floor in the earlier iteration).
DIM_COLOR = {"0D": "#E69F00", "1D": "#0072B2", "2D": "#009E73", "3D": "#D55E00"}
# target_non0D: 0 = 0D, 1 = non-0D (analysis-3 convention; flipped from analysis 2's 1 = 0D).
BINARY_COLOR = {"0D": ORANGE, "non-0D": BLUE}
# Ordinal ramp for prompting conditions: light -> dark = fewer -> more in-context examples.
PROMPTING_RAMP = {"zero": "#86b6ef", "few_shot": BLUE, "all_shot": "#104281"}
POSITIVE, NEGATIVE = BLUE, RED  # diverging pair (signed gain charts)
ML_COLOR, LLM_COLOR = BLUE, ORANGE  # method-family color for the final combined comparison
INORGANIC_COLOR, CATION_COLOR = BLUE, ORANGE  # feature-block color for importance charts


def style_axes(ax, grid_axis="y"):
    ax.set_facecolor(BG)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=INK_SOFT, labelcolor=INK)


def new_fig(figsize, **kwargs):
    fig, ax = plt.subplots(figsize=figsize, **kwargs)
    fig.patch.set_facecolor(BG)
    if hasattr(ax, "flat"):
        for a in ax.flat: a.set_facecolor(BG)
    else:
        ax.set_facecolor(BG)
    return fig, ax


def save(fig, path, dpi=200):
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, facecolor=BG)
    print(f"Saved {path}")
