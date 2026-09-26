"""
Publication-quality figure generator for the Sb-halide LLM extraction pipeline.

Produces true vector output (SVG + PDF) plus a high-DPI PNG preview, built
entirely from matplotlib primitives (no raster icon assets) so every element
stays editable text/paths in Illustrator, Inkscape, or Affinity Designer.

Run:
    python make_pipeline_figure.py
"""

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"   # keep text as real <text>, not paths
matplotlib.rcParams["pdf.fonttype"] = 42        # embed as TrueType, not Type3 bitmaps
matplotlib.rcParams["font.family"] = "Helvetica"
matplotlib.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, PathPatch
from matplotlib.path import Path
from matplotlib.lines import Line2D
import numpy as np

# ----------------------------------------------------------------------------
# Palette (semantic roles fixed by the pipeline spec, section 11.2)
# ----------------------------------------------------------------------------
BLUE    = "#2E5C8A"   # document preparation & retrieval infrastructure
PURPLE  = "#6A4C93"   # LLM semantic processing (RAG, structuring, reasoning)
ORANGE  = "#C97A1A"   # evidence texts / retained outputs
RED     = "#B23A2E"   # verification failure / targeted re-retrieval
GREEN   = "#2E7D5B"   # human validation & final dataset
INK     = "#232327"   # primary text
SUBINK  = "#5B5B62"   # secondary / caption text
LINE    = "#3A3A3F"   # neutral connector lines
FAINT   = "#9A9AA2"

FONT = "Helvetica"

# ----------------------------------------------------------------------------
# Canvas: 190 mm x 108 mm, drawn in mm units for precise, print-scale control
# ----------------------------------------------------------------------------
W, H = 190.0, 113.0
fig = plt.figure(figsize=(W / 25.4, H / 25.4))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.set_aspect("equal")
ax.axis("off")


# ----------------------------------------------------------------------------
# Generic drawing helpers
# ----------------------------------------------------------------------------
def to_rgb(hexcolor):
    hexcolor = hexcolor.lstrip("#")
    return tuple(int(hexcolor[i:i + 2], 16) / 255 for i in (0, 2, 4))


def box(x, y, w, h, color, fill_alpha=0.09, lw=1.05, ls="-", rounding=1.6, z=2):
    # Leave patch-level alpha at None so the independent RGBA alpha channels
    # on face/edge are respected (a shared patch.alpha would override both).
    face = (*to_rgb(color), fill_alpha)
    edge = (*to_rgb(color), 1.0)
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={rounding}",
        linewidth=lw, edgecolor=edge, facecolor=face,
        zorder=z, linestyle=ls,
    )
    ax.add_patch(p)
    return p


def label(x, y, text, size=7.0, color=INK, weight="bold", ha="center",
          va="center", style="normal"):
    ax.text(x, y, text, fontsize=size, color=color, fontweight=weight,
             ha=ha, va=va, family=FONT, style=style, zorder=6)


def block(x, y, w, h, title, subtitle=None, color=PURPLE, fill_alpha=0.09,
          icon=None, icon_kw=None, title_size=7.0, sub_size=5.6, ls="-", icon_size=3.7,
          sub_offset=2.3):
    """A titled box: icon top-center, title below it (anchored top-down so
    multi-line titles never collide with the icon), subtitle pinned near the
    bottom edge (anchored bottom-up so it never collides with the title)."""
    box(x, y, w, h, color, fill_alpha=fill_alpha, ls=ls)
    top = y + h
    if icon is not None:
        icon_cy = top - 4.1
        icon(ax, x + w / 2, icon_cy, icon_size, color, **(icon_kw or {}))
        title_top = icon_cy - 3.1
    else:
        title_top = top - 2.6 if subtitle else None

    if title_top is not None:
        ax.text(x + w / 2, title_top, title, fontsize=title_size, color=INK,
                 fontweight="bold", ha="center", va="top", family=FONT, zorder=6,
                 linespacing=1.28)
    else:
        label(x + w / 2, y + h / 2, title, size=title_size, color=INK, ha="center")

    if subtitle:
        ax.text(x + w / 2, y + sub_offset, subtitle, fontsize=sub_size, color=SUBINK,
                 fontweight="normal", ha="center", va="bottom", family=FONT,
                 zorder=6, linespacing=1.2)


# Single consistent stroke weight and arrowhead size for every connector in
# the diagram — no branch should read as more or less important than another.
FLOW_LW = 1.05
FLOW_MUTATION = 7.5


def arrow(x1, y1, x2, y2, color=LINE, lw=FLOW_LW, ls="-", connectionstyle="arc3,rad=0",
          mutation=FLOW_MUTATION, z=3):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=mutation,
                         linewidth=lw, edgecolor=color, facecolor=color,
                         connectionstyle=connectionstyle, zorder=z, linestyle=ls,
                         shrinkA=0, shrinkB=0)
    ax.add_patch(a)


def elbow(points, color=LINE, lw=FLOW_LW, ls="-", z=3, arrow_last=True, mutation=FLOW_MUTATION):
    """Right-angle / polyline connector through a list of (x,y) points."""
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        if arrow_last and i == len(points) - 2:
            arrow(x1, y1, x2, y2, color=color, lw=lw, ls=ls, mutation=mutation, z=z)
        else:
            ax.add_line(Line2D([x1, x2], [y1, y2], color=color, linewidth=lw,
                                linestyle=ls, zorder=z, solid_capstyle="round"))


def zone_header(x, w, y, text, color):
    label(x + w / 2, y + 3.3, text, size=8.4, color=INK, weight="bold")
    ax.add_line(Line2D([x, x + w], [y, y], color=color, linewidth=1.8,
                        zorder=5, solid_capstyle="round"))


def tag(x, y, text, color, size=5.7, ha="center", va="center"):
    ax.text(x, y, text, fontsize=size, color=color, fontweight="bold",
             ha=ha, va=va, family=FONT, zorder=6)


# ----------------------------------------------------------------------------
# Icon library — minimal monoline glyphs, uniform stroke, no fills except dots
# ----------------------------------------------------------------------------
def _stroke(pts, color, lw=0.9, closed=False, z=7):
    codes = [Path.MOVETO] + [Path.LINETO] * (len(pts) - 1)
    if closed:
        pts = pts + [pts[0]]
        codes = codes + [Path.CLOSEPOLY]
    p = PathPatch(Path(pts, codes), edgecolor=color, facecolor="none",
                   linewidth=lw, zorder=z, capstyle="round", joinstyle="round")
    ax.add_patch(p)


def icon_documents(ax, cx, cy, s, color, **kw):
    # two stacked pages with fold + text lines
    for dx, dy, a in [(0.6, -0.5, 0.55), (0, 0, 1.0)]:
        x0, y0 = cx - s * 0.55 + dx, cy - s * 0.6 + dy
        w, h = s * 1.1, s * 1.2
        rect = Rectangle((x0, y0), w, h, fill=False, edgecolor=color,
                          linewidth=0.85, alpha=a, zorder=7)
        ax.add_patch(rect)
    x0, y0 = cx - s * 0.55, cy - s * 0.6
    w, h = s * 1.1, s * 1.2
    for i, frac in enumerate([0.72, 0.52, 0.32]):
        yy = y0 + h * frac
        ax.add_line(Line2D([x0 + w * 0.18, x0 + w * 0.82], [yy, yy],
                            color=color, linewidth=0.55, zorder=7))


def icon_checklist(ax, cx, cy, s, color, **kw):
    x0, y0 = cx - s * 0.55, cy - s * 0.6
    w, h = s * 1.1, s * 1.2
    ax.add_patch(Rectangle((x0, y0), w, h, fill=False, edgecolor=color,
                            linewidth=0.85, zorder=7))
    for frac in [0.72, 0.46, 0.20]:
        yy = y0 + h * frac
        _stroke([(x0 + w * 0.14, yy), (x0 + w * 0.24, yy - h * 0.07),
                 (x0 + w * 0.40, yy + h * 0.10)], color, lw=0.7)
        ax.add_line(Line2D([x0 + w * 0.48, x0 + w * 0.86], [yy + h * 0.02, yy + h * 0.02],
                            color=color, linewidth=0.55, zorder=7))


def icon_layout(ax, cx, cy, s, color, **kw):
    x0, y0 = cx - s * 0.55, cy - s * 0.6
    w, h = s * 1.1, s * 1.2
    ax.add_patch(Rectangle((x0, y0), w, h, fill=False, edgecolor=color,
                            linewidth=0.85, zorder=7))
    ax.add_line(Line2D([x0, x0 + w], [y0 + h * 0.62, y0 + h * 0.62],
                        color=color, linewidth=0.6, zorder=7))
    ax.add_line(Line2D([x0 + w * 0.5, x0 + w * 0.5], [y0, y0 + h * 0.62],
                        color=color, linewidth=0.6, zorder=7))
    for yy in [y0 + h * 0.82]:
        ax.add_line(Line2D([x0 + w * 0.12, x0 + w * 0.88], [yy, yy],
                            color=color, linewidth=0.45, zorder=7))
    for yy in [y0 + h * 0.18, y0 + h * 0.36]:
        ax.add_line(Line2D([x0 + w * 0.10, x0 + w * 0.42], [yy, yy],
                            color=color, linewidth=0.45, zorder=7))
        ax.add_line(Line2D([x0 + w * 0.58, x0 + w * 0.90], [yy, yy],
                            color=color, linewidth=0.45, zorder=7))


def icon_registry(ax, cx, cy, s, color, **kw):
    rx, ry = s * 0.55, s * 0.18
    x0 = cx - rx
    y_top = cy + s * 0.55
    y_bot = cy - s * 0.55
    for yy in [y_top, y_bot]:
        el = matplotlib.patches.Ellipse((cx, yy), rx * 2, ry * 2, fill=False,
                                         edgecolor=color, linewidth=0.8, zorder=7)
        ax.add_patch(el)
    ax.add_line(Line2D([x0, x0], [y_bot, y_top], color=color, linewidth=0.8, zorder=7))
    ax.add_line(Line2D([x0 + rx * 2, x0 + rx * 2], [y_bot, y_top], color=color,
                        linewidth=0.8, zorder=7))
    el2 = matplotlib.patches.Ellipse((cx, y_top), rx * 2, ry * 2, fill=False,
                                      edgecolor=color, linewidth=0.8, zorder=7)
    ax.add_patch(el2)


def icon_rag(ax, cx, cy, s, color, **kw):
    # magnifying glass over three small linked nodes
    r = s * 0.36
    gx, gy = cx - s * 0.12, cy + s * 0.14
    circ = Circle((gx, gy), r, fill=False, edgecolor=color, linewidth=0.9, zorder=7)
    ax.add_patch(circ)
    ang = np.deg2rad(45)
    hx1, hy1 = gx + r * np.cos(ang), gy - r * np.sin(ang)
    hx2, hy2 = hx1 + s * 0.34 * np.cos(ang), hy1 - s * 0.34 * np.sin(ang)
    ax.add_line(Line2D([hx1, hx2], [hy1, hy2], color=color, linewidth=1.15,
                        zorder=7, solid_capstyle="round"))
    for dx, dy in [(-0.4, -0.35), (0.05, -0.5), (0.42, -0.3)]:
        ax.add_patch(Circle((gx + dx * s * 0.5, gy + dy * s * 0.5), s * 0.05,
                             facecolor=color, edgecolor="none", zorder=7))


def icon_molecule(ax, cx, cy, s, color, **kw):
    pts = [(cx - s * 0.32, cy - s * 0.15), (cx + s * 0.05, cy + s * 0.35),
           (cx + s * 0.38, cy - s * 0.05), (cx - s * 0.05, cy - s * 0.45)]
    ax.add_line(Line2D([pts[0][0], pts[1][0]], [pts[0][1], pts[1][1]], color=color, linewidth=0.7, zorder=7))
    ax.add_line(Line2D([pts[1][0], pts[2][0]], [pts[1][1], pts[2][1]], color=color, linewidth=0.7, zorder=7))
    ax.add_line(Line2D([pts[0][0], pts[3][0]], [pts[0][1], pts[3][1]], color=color, linewidth=0.7, zorder=7))
    for (px, py) in pts:
        ax.add_patch(Circle((px, py), s * 0.09, facecolor="white", edgecolor=color,
                             linewidth=0.7, zorder=8))


def icon_formula(ax, cx, cy, s, color, **kw):
    # a small chemical-formula pictogram: "AB2" with a true subscript
    ax.text(cx - s * 0.05, cy, "AB", fontsize=s * 2.1, color=color, fontweight="bold",
            ha="right", va="center", family=FONT, zorder=7)
    ax.text(cx + s * 0.02, cy - s * 0.22, "2", fontsize=s * 1.4, color=color, fontweight="bold",
            ha="left", va="center", family=FONT, zorder=7)


def icon_network(ax, cx, cy, s, color, **kw):
    nodes = [(cx - s * 0.4, cy - s * 0.35), (cx - s * 0.05, cy + s * 0.4),
              (cx + s * 0.4, cy - s * 0.1), (cx + s * 0.02, cy - s * 0.45)]
    edges = [(0, 1), (1, 2), (0, 3), (3, 2), (0, 2)]
    for i, j in edges:
        ax.add_line(Line2D([nodes[i][0], nodes[j][0]], [nodes[i][1], nodes[j][1]],
                            color=color, linewidth=0.65, zorder=7))
    for (px, py) in nodes:
        ax.add_patch(Circle((px, py), s * 0.085, facecolor=color, edgecolor="none", zorder=8))


def icon_flask(ax, cx, cy, s, color, **kw):
    x0 = cx - s * 0.12
    neck_top = cy + s * 0.55
    neck_bot = cy + s * 0.12
    body_r = s * 0.42
    ax.add_line(Line2D([x0 - s * 0.14, x0 + s * 0.14], [neck_top, neck_top], color=color, linewidth=0.8, zorder=7))
    ax.add_line(Line2D([x0 - s * 0.1, x0 - s * 0.1], [neck_top, neck_bot], color=color, linewidth=0.8, zorder=7))
    ax.add_line(Line2D([x0 + s * 0.1, x0 + s * 0.1], [neck_top, neck_bot], color=color, linewidth=0.8, zorder=7))
    arc = matplotlib.patches.Arc((x0, cy - s * 0.05), body_r * 2, body_r * 2,
                                  angle=0, theta1=200, theta2=340, edgecolor=color,
                                  linewidth=0.8, zorder=7)
    ax.add_patch(arc)
    _stroke([(x0 - s * 0.1, neck_bot), (x0 - body_r * 0.92, cy - s * 0.28)], color, lw=0.8)
    _stroke([(x0 + s * 0.1, neck_bot), (x0 + body_r * 0.92, cy - s * 0.28)], color, lw=0.8)
    ax.add_patch(Circle((x0 - s * 0.05, cy - s * 0.2), s * 0.045, facecolor=color, edgecolor="none", zorder=8))
    ax.add_patch(Circle((x0 + s * 0.12, cy - s * 0.3), s * 0.045, facecolor=color, edgecolor="none", zorder=8))


def icon_table(ax, cx, cy, s, color, **kw):
    x0, y0 = cx - s * 0.55, cy - s * 0.5
    w, h = s * 1.1, s * 1.0
    ax.add_patch(Rectangle((x0, y0), w, h, fill=False, edgecolor=color, linewidth=0.85, zorder=7))
    ax.add_line(Line2D([x0, x0 + w], [y0 + h * 0.65, y0 + h * 0.65], color=color, linewidth=0.6, zorder=7))
    for fx in [1 / 3, 2 / 3]:
        ax.add_line(Line2D([x0 + w * fx, x0 + w * fx], [y0, y0 + h], color=color, linewidth=0.5, zorder=7))


def icon_reason(ax, cx, cy, s, color, **kw):
    # branching directional network denoting dimensionality reasoning
    origin = (cx - s * 0.3, cy - s * 0.4)
    ends = [(cx - s * 0.3, cy + s * 0.45), (cx + s * 0.15, cy + s * 0.15), (cx + s * 0.42, cy - s * 0.35)]
    for e in ends:
        arrow(origin[0], origin[1], e[0], e[1], color=color, lw=0.75, mutation=4.2, z=7)
    ax.add_patch(Circle(origin, s * 0.09, facecolor=color, edgecolor="none", zorder=8))


def icon_scroll(ax, cx, cy, s, color, **kw):
    x0, y0 = cx - s * 0.45, cy - s * 0.5
    w, h = s * 0.9, s * 1.0
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=0.5",
                                 fill=False, edgecolor=color, linewidth=0.8, zorder=7))
    for frac in [0.72, 0.5, 0.28]:
        ax.add_line(Line2D([x0 + w * 0.16, x0 + w * 0.84], [y0 + h * frac, y0 + h * frac],
                            color=color, linewidth=0.5, zorder=7))


def icon_record(ax, cx, cy, s, color, **kw):
    x0, y0 = cx - s * 0.55, cy - s * 0.5
    w, h = s * 1.1, s * 1.0
    ax.add_patch(Rectangle((x0, y0), w, h, fill=False, edgecolor=color, linewidth=0.85, zorder=7))
    for i, frac in enumerate([0.78, 0.5, 0.22]):
        yy = y0 + h * frac
        ax.add_line(Line2D([x0, x0 + w], [yy + h * 0.11, yy + h * 0.11], color=color, linewidth=0.45, zorder=7))
        dotc = [GREEN, PURPLE, ORANGE][i]
        ax.add_patch(Circle((x0 + w * 0.16, yy), s * 0.05, facecolor=dotc, edgecolor="none", zorder=8))


def icon_shield(ax, cx, cy, s, color, **kw):
    x0, y0 = cx, cy + s * 0.55
    pts = [(x0 - s * 0.4, y0), (x0 + s * 0.4, y0), (x0 + s * 0.4, y0 - s * 0.65),
           (x0, y0 - s * 1.15), (x0 - s * 0.4, y0 - s * 0.65)]
    _stroke(pts, color, lw=0.85, closed=True)
    _stroke([(x0 - s * 0.18, y0 - s * 0.55), (x0 - s * 0.03, y0 - s * 0.72),
             (x0 + s * 0.24, y0 - s * 0.32)], color, lw=0.85)


def icon_person(ax, cx, cy, s, color, **kw):
    ax.add_patch(Circle((cx, cy + s * 0.32), s * 0.22, fill=False, edgecolor=color, linewidth=0.85, zorder=7))
    arc = matplotlib.patches.Arc((cx, cy - s * 0.32), s * 0.9, s * 0.8, angle=0,
                                  theta1=20, theta2=160, edgecolor=color, linewidth=0.85, zorder=7)
    ax.add_patch(arc)
    _stroke([(cx + s * 0.18, cy - s * 0.02), (cx + s * 0.3, cy - s * 0.16),
             (cx + s * 0.52, cy + s * 0.16)], color, lw=0.85)


def icon_dataset(ax, cx, cy, s, color, **kw):
    icon_registry(ax, cx - s * 0.28, cy, s * 0.62, color)
    x0 = cx + s * 0.05
    bw = s * 0.11
    for i, hh in enumerate([0.35, 0.6, 0.45]):
        xx = x0 + i * (bw + s * 0.04)
        ax.add_patch(Rectangle((xx, cy - s * 0.45), bw, s * hh, facecolor=color,
                                edgecolor="none", alpha=0.9, zorder=7))


# ----------------------------------------------------------------------------
# Layout constants
# ----------------------------------------------------------------------------
HEAD_Y = 105.5
# CATION row's top edge is pinned to match the "Articles" input box's top
# edge (both sit 4mm below their zone header line); row gaps are then
# stretched evenly so the SYNTHESIS row's bottom still lands at its
# original position — top-aligned without leaving dead space at the bottom.
ROW_CATION_Y0, ROW_CATION_H = 81.0, 15.5
ROW_CONN_Y0, ROW_CONN_H     = 52.5, 15.5
ROW_SYN_Y0,  ROW_SYN_H      = 24.0, 15.5

# ============================== ZONE A =====================================
zoneA_x, zoneA_w = 3.5, 46.0
zone_header(zoneA_x, zoneA_w, HEAD_Y, "Document Preparation", BLUE)

# Input
inx, iny, inw, inh = 8, 84, 34, 12.5
block(inx, iny, inw, inh, "Articles", color=BLUE, icon=icon_documents, title_size=6.9)

# Preparation container (subtle group background)
grp_x, grp_y, grp_w, grp_h = 5.5, 20.5, 38, 56.5
arrow(inx + inw / 2, iny, inx + inw / 2, grp_y + grp_h, color=LINE)
box(grp_x, grp_y, grp_w, grp_h, BLUE, fill_alpha=0.035, lw=0.6, rounding=2.4, z=1)

prep_x, prep_w = 9, 32
b1y, b1h = 62.5, 13
b2y, b2h = 43.5, 13
b3y, b3h = 24.5, 13

block(prep_x, b1y, prep_w, b1h, "Layout-aware parsing", None,
      color=BLUE, icon=icon_layout, title_size=6.5, sub_size=4.6, sub_offset=1.2)
arrow(prep_x + prep_w / 2, b1y, prep_x + prep_w / 2, b2y + b2h, color=LINE)

block(prep_x, b2y, prep_w, b2h, "Eligibility screening", None,
      color=BLUE, icon=icon_checklist, title_size=6.5, sub_size=5.0, sub_offset=1.2)
arrow(prep_x + prep_w / 2, b2y, prep_x + prep_w / 2, b3y + b3h, color=LINE)

block(prep_x, b3y, prep_w, b3h, "Sb-compound registry", None,
      color=BLUE, icon=icon_registry, title_size=6.5, sub_size=5.0, sub_offset=1.2)

# ============================== ZONE B =====================================
zoneB_x, zoneB_w = 52.0, 99.0
zone_header(zoneB_x, zoneB_w, HEAD_Y, "Compound-Conditioned RAG LLM Extraction", PURPLE)

# connector from registry to RAG — target y is the RAG box's true vertical
# center (ROW_CONN_Y0 + ROW_CONN_H/2), not a stale hardcoded value, so the
# arrowhead always lands in the middle of the box regardless of layout tweaks.
_rag_entry_y = ROW_CONN_Y0 + ROW_CONN_H / 2
elbow([(prep_x + prep_w, b3y + b3h / 2), (48.5, b3y + b3h / 2), (48.5, _rag_entry_y), (53.5, _rag_entry_y)],
      color=LINE)

ragx, ragy, ragw, ragh = 53.5, 47.75, 25.0, 25
block(ragx, ragy, ragw, ragh, "Compound-\nconditioned\nretrieval-augmented\ngeneration (RAG)", None,
      color=PURPLE, fill_alpha=0.11, icon=icon_rag, title_size=5.6, sub_size=4.9)

# Row geometry
evx, evw = 86.0, 24
prx, prw = 113.0, 24.5

cation_cy = ROW_CATION_Y0 + ROW_CATION_H / 2
conn_cy = ROW_CONN_Y0 + ROW_CONN_H / 2
syn_cy = ROW_SYN_Y0 + ROW_SYN_H / 2

# One connector that exits the RAG box and forks smoothly into the three
# branches, instead of three independent arrows (two of which previously
# started in mid-air, since the RAG box only spans the connectivity row).
# The stub-out point is pinned to conn_cy (not the RAG box's own geometric
# center) so the middle branch is a single straight line with no kink at
# the fork — the two centers are close but not pixel-identical otherwise.
rag_cy = conn_cy
fork_x = ragx + ragw + 2.5
ax.add_line(Line2D([ragx + ragw, fork_x], [rag_cy, rag_cy], color=LINE, linewidth=FLOW_LW,
                    zorder=3, solid_capstyle="round"))
ax.add_line(Line2D([fork_x, fork_x], [syn_cy, cation_cy], color=LINE, linewidth=FLOW_LW,
                    zorder=3, solid_capstyle="round"))
for _cy in (cation_cy, conn_cy, syn_cy):
    arrow(fork_x, _cy, evx, _cy, color=LINE)


def row_branch(y0, h, cy, ev_title, ev_icon, tagtxt, proc_title, proc_sub, proc_icon,
               proc_color=PURPLE, proc_alpha=0.10, merge=True, proc_w=prw, proc_ls="-",
               proc_title_size=6.3):
    tag(evx + evw / 2, y0 + h + 2.6, tagtxt, ORANGE, size=5.8)
    block(evx, y0, evw, h, ev_title, color=ORANGE, fill_alpha=0.11, icon=ev_icon, title_size=6.1)
    if merge:
        arrow(evx + evw, cy, prx, cy, color=LINE)
        block(prx, y0, proc_w, h, proc_title, proc_sub, color=proc_color,
              fill_alpha=proc_alpha, icon=proc_icon, title_size=proc_title_size, sub_size=4.9, ls=proc_ls,
              sub_offset=1.2)


row_branch(ROW_CATION_Y0, ROW_CATION_H, cation_cy, "Cation evidence\ntext", icon_molecule,
           "CATION", "Structured\nextraction", None,
           icon_table, proc_color=PURPLE)

row_branch(ROW_CONN_Y0, ROW_CONN_H, conn_cy, "Sb–halide connectivity\nevidence text", icon_network,
           "CONNECTIVITY", "Dimensionality\nreasoning", None,
           icon_reason, proc_color=PURPLE)

# Synthesis: single unified box (no separate LLM structuring step) — deliberately
# distinct from the two rows above to signal that synthesis text is retained, not reasoned.
tag(evx + evw / 2, ROW_SYN_Y0 + ROW_SYN_H + 2.6, "SYNTHESIS", ORANGE, size=5.8)
syn_w = evw + 5 + prw
block(evx, ROW_SYN_Y0, syn_w, ROW_SYN_H, "Synthesis evidence text",
      None,
      color=ORANGE, fill_alpha=0.11, icon=icon_flask, title_size=6.1, sub_size=4.9)

prx_right = prx + prw

# ============================== ZONE C =====================================
zoneC_x, zoneC_w = 153.5, 33.0
zone_header(zoneC_x, zoneC_w, HEAD_Y, "Verification & Curation", GREEN)

cx0, cw = 157, 26
rec_y, rec_h   = 82.5, 14.0
ver_y, ver_h   = 57.2, 15.0
man_y, man_h   = 32.8, 14.0
fin_y, fin_h   = 9.0, 13.5

# converge cation / connectivity / synthesis into record assembly
merge_x = 150.6
elbow([(prx_right, cation_cy), (merge_x, cation_cy), (merge_x, rec_y + rec_h * 0.72), (cx0, rec_y + rec_h * 0.72)],
      color=LINE)
elbow([(prx_right, conn_cy), (merge_x + 1.1, conn_cy), (merge_x + 1.1, rec_y + rec_h * 0.5), (cx0, rec_y + rec_h * 0.5)],
      color=LINE)
elbow([(evx + syn_w, syn_cy), (merge_x + 2.2, syn_cy), (merge_x + 2.2, rec_y + rec_h * 0.28), (cx0, rec_y + rec_h * 0.28)],
      color=LINE)

block(cx0, rec_y, cw, rec_h, "Compound-level\nrecord assembly", None, color=INK,
      fill_alpha=0.05, icon=icon_record, title_size=6.5)

arrow(cx0 + cw / 2, rec_y, cx0 + cw / 2, ver_y + ver_h, color=LINE)
block(cx0, ver_y, cw, ver_h, "Source-grounded\nverification", None,
      color=GREEN, fill_alpha=0.10, icon=icon_shield, title_size=6.4)

# supported -> human validation
arrow(cx0 + cw / 2, ver_y, cx0 + cw / 2, man_y + man_h, color=LINE)

block(cx0, man_y, cw, man_h, "Human\nvalidation", None, color=GREEN,
      fill_alpha=0.10, icon=icon_person, title_size=6.4)

arrow(cx0 + cw / 2, man_y, cx0 + cw / 2, fin_y + fin_h, color=LINE)
block(cx0, fin_y, cw, fin_h, "Final curated dataset", None,
      color=GREEN, fill_alpha=0.13, icon=icon_dataset, title_size=6.4, sub_size=4.9)

# ---- feedback loop: incomplete / conflicting -> targeted re-retrieval ------
fb_y = 11.5
fb_x = 147.5
elbow([(cx0, ver_y + ver_h * 0.28), (fb_x, ver_y + ver_h * 0.28), (fb_x, fb_y),
       (ragx + ragw / 2, fb_y), (ragx + ragw / 2, ragy)],
      color=RED, lw=1.0, ls=(0, (4, 2.2)), arrow_last=True, mutation=5.5)
label(102, fb_y + 2.3, "targeted re-retrieval — failed field only", size=5.5, color=RED,
      weight="bold", style="italic")

# ----------------------------------------------------------------------------
plt.tight_layout(pad=0)

out_dir = "/Users/yuanhong/Research/2026 intern NSF/analysis 3/sb_literature_pipeline/figures"
for ext, dpi in [("svg", None), ("pdf", None), ("png", 600)]:
    kwargs = {"transparent": False, "facecolor": "white"}
    if dpi:
        kwargs["dpi"] = dpi
    fig.savefig(f"{out_dir}/extraction_workflow.{ext}", **kwargs)

print("done")
