"""Numerical figure renderer from the archived paper build."""

from __future__ import annotations

import colorsys
import hashlib
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.scale import SymmetricalLogTransform  # noqa: E402
from matplotlib.ticker import NullLocator  # noqa: E402

HERE = Path(__file__).resolve().parent
REDESIGN = HERE
REPO = HERE
sys.path.insert(0, str(REDESIGN / "research2"))
sys.path.insert(0, str(REDESIGN / "qa"))

import craft  # noqa: E402

# ---------------------------------------------------------------------------
# 0.  Frozen inputs — SHA-256 asserted before anything is read
# ---------------------------------------------------------------------------

FROZEN = REPO / "inputs/abc/unified_windows_v1.json"
ORIGINAL_ARCHIVE_SHA256 = "75431e13cb3cf0ebba9f8a86df443fcdda5c7fd10b6cd2fb28fd390909d241ac"  # Historical pre-release identity.
FROZEN_SHA = "a500fe10e875eacf602c1d3ed02e796b2de1b06d5051064d250692eb44f6203b"  # Public input bytes.
CONTRACT = HERE / "inputs/figure2_values.json"

_RAW = FROZEN.read_bytes()
_SHA = hashlib.sha256(_RAW).hexdigest()
assert _SHA == FROZEN_SHA, f"frozen input changed: {FROZEN} sha256={_SHA}"
EST = json.loads(_RAW.decode())["estimates"]

CONTRACT_JSON = json.loads(CONTRACT.read_text())
assert CONTRACT_JSON["source_sha256"] == FROZEN_SHA
assert CONTRACT_JSON["sha256_verified"] is True

MODELS = [
    "Qwen3.6",
    "Qwen3",
    "GLM-4.7-Flash",
    "Lightning",
    "Nano",
    "Gemma4",
    "OLMoE",
    "GPT-OSS-20B",
    "GPT-OSS-120B",
]
ARMS_TOP_TO_BOTTOM = ["router_logits", "all_router", "post_moe_residual"]


def row(model, **criteria):
    rows = [r for r in EST if r["model"] == model and all(r[k] == v for k, v in criteria.items())]
    assert len(rows) == 1, (model, criteria, len(rows))
    r = rows[0]
    assert r["ci_low"] <= r["mean"] <= r["ci_high"]
    assert r["draws"] == 50000 and r["seed"] == 20260908
    return r


def fmt(x):
    return f"{x:.3f}".replace("-", "−")


FIGURE_SPACE = " "


def sign_pad(s: str) -> str:
    """Reserve a sign column so the nine closing brackets align (v3 fix a2)."""
    return s if s.startswith("−") else FIGURE_SPACE + s


# ---------------------------------------------------------------------------
# 1.  Palette — VISUAL_LANGUAGE §2, unchanged
# ---------------------------------------------------------------------------

INK = "#11181F"
MUTED = "#38424C"
RULE = "#6B7580"
HAIR = "#98A2AB"
GRID = "#D3D9DE"
BAND = "#EDEFF2"
WHITE = "#FFFFFF"

ROUTER, ROUTER_DARK = "#83929E", "#424C55"
ALLROUTER, ALLROUTER_DARK = "#087F8C", "#05535C"
RESIDUAL, RESIDUAL_DARK = "#893957", "#7A334E"

#: VL §2.1 "ROUTER fill L* 90" — the one pale neutral the window blocks are printed in.
#: It names no interface: both windows carry it, because the two windows ARE the same
#: object at two positions, which is the schematic's whole point.
WINDOW_FILL = "#DFE3E6"

PALETTE = {INK, MUTED, RULE, HAIR, GRID, BAND, WHITE, WINDOW_FILL,
           ROUTER, ROUTER_DARK, ALLROUTER, ALLROUTER_DARK, RESIDUAL, RESIDUAL_DARK}

#: fill = support (v3 ITEM 11, paper-wide): a single-layer transfer is an open mark,
#: an all-router transfer is filled.
#:
#: ROUND-4 FIX (marker diameter, 3.6 -> 3.2 pt).  On plan A the all-router square and
#: the single-layer residual triangle are near-coincident in x in three rows
#: (Lightning dx = 0.33 pt, Nano 2.15 pt, GPT-OSS-120B 6.34 pt), so their separation
#: is carried almost entirely by the in-row vertical offset.  Shrinking every mark by
#: 0.4 pt buys 0.4 pt of white between the two marks in exactly those rows and costs
#: nothing elsewhere: the shortest rendered interval in the figure is 4.63 pt, so the
#: occlusion margin *improves* from 1.03 pt to 1.43 pt.  All four series move together
#: — the figure keeps one marker size.
MS = 3.2

STYLE = {
    "router_logits": dict(color=ROUTER, marker="o", ms=MS, mfc=WHITE, mew=1.0,
                          open=True, label="Single-layer router", dark=ROUTER_DARK),
    "all_router": dict(color=ALLROUTER, marker="s", ms=MS, mfc=ALLROUTER, mew=0.60,
                       open=False, label="All-router", dark=ALLROUTER_DARK),
    "post_moe_residual": dict(color=RESIDUAL, marker="^", ms=MS, mfc=WHITE, mew=1.0,
                              open=True, label="Single-layer residual",
                              dark=RESIDUAL_DARK),
}
#: ROUND 10 (PALETTE_SPEC §6.1).  Panel D is ONE series and it is derived — the
#: divergence-8 minus suffix-8 shift of the residual-minus-all-router gap.  There is no
#: categorical distinction here for a hue to carry, and reusing the residual arm's
#: crimson for a difference that is not the residual arm gave crimson a second meaning
#: inside this figure.  Achromatic for a derived quantity, as Figure 3 already does for
#: its receiver average.
D_STYLE = dict(color=INK, marker="o", ms=MS, mfc=INK, mew=0.60, open=False)


def mec_of(st):
    """Edge colour.  An open mark is outlined in the DARK step of its own hue, not
    the mid step: at 3.2 pt the ring is the mark, so it carries the higher-contrast
    end of the three-step lightness ladder.  A filled mark keeps its white halo,
    which is what stops it merging with a neighbouring mark in a crowded row."""
    return st["dark"] if st["open"] else WHITE


LW_GRID, LW_HAIRLINE, LW_ZERO, LW_FRAME, LW_CARD, LW_DATA = 0.50, 0.60, 0.64, 0.80, 0.90, 1.00
R_CHIP = 2.5
CAPSIZE, CAPTHICK, ELW = 1.5, 1.0, 1.00
DASH_ZERO = (0, (2.55, 1.91))
DASH_GRID = (0, (2.4, 2.0))

SZ_LETTER, SZ_BODY = 9.0, 8.0
BOLD_SIZES = {SZ_BODY, SZ_LETTER}
CAP = craft.CAP_HEIGHT_EM           # 0.662 em
BBOX_TOP, BBOX_BOT = 5.64, -1.66    # measured Times bbox at 8.0 pt, rel. baseline

# ---------------------------------------------------------------------------
# 2.  Canvas and layout — points from the lower-left corner
# ---------------------------------------------------------------------------

W_PT = 396.0

NAME_RIGHT = 60.0
B_AX = (68.0, 226.0)                # the shared symlog T axes
C_MEAN_DEC = 242.5                  # decimal x of the exact mean column
C_CI_X = 260.5                      # left edge of the CI string
C_LEFT = 234.0                      # measured left ink edge of the mean column
D_AX = (323.0, 390.0)               # the shared symlog gap-shift axes

#: The whole vertical stack is derived from ONE free parameter, the row pitch, so the
#: canvas height can be trimmed to hold the float on its page without re-deciding any
#: gap.  Every gap below was measured against the 8.0 pt Times bbox (top +5.64,
#: bottom −1.66 from the baseline) and clears 3 pt.
BLOCK_BOT = 32.0
#: The two estimate panels are drawn 2 pt taller than the nine-row block at each end,
#: so the first and last rows' marks clear their own frame by 4 pt instead of 2.  Row
#: positions in POINTS are unchanged; only the spines move.
AX_PAD = 2.0
TICK_BASE = 20.0                    # tick-label baseline for both estimate panels
NOTE_BASE = 9.0                     # axis-note baseline
STRIP = (68.0, 390.0)               # the judged step, drawn across the full field
N_CELLS = 24
DIV_CELL = 8                        # the first causal divergence token
DIV_WIN = (5, 13)                   # divergence-8  = cells 5..12
SUF_WIN = (16, 24)                  # suffix-8      = cells 16..23

PITCH = 18.0
#: ROUND-4 FIX (in-row arm offset, 4.5/4.2 -> 5.6/5.2 pt).  The three interfaces were
#: already offset, but by less than 1.2 marker diameters, which is not enough when two
#: of them also coincide in x.  Widening the offset is the first-choice fix because it
#: separates the marks without touching a single value, an endpoint or the axis.  The
#: ceiling is row grouping: the arms span 2*ARM_DY and the gutter rule sits at
#: PITCH/2, so at pitch 17 the bottom arm of a row still clears its own separator by
#: 3.3 pt and the nearest arm of the next row by 6.6 pt — wider than the 5.2 pt
#: within-row step, so rows still read as rows.
ARM_DY = 5.6                        # row arm offsets, top-to-bottom +5.6 / 0 / -5.6
BLOCK_TOP = 194.0
LEGEND_CY = 207.0
LETTER_BASE = 218.0                 # B / C / D letter row
WIN_LABEL_BASE = 231.0
STRIP_Y0, STRIP_Y1 = 240.0, 254.0
DIV_LABEL_BASE = 259.0
SCHEMA_HEAD_BASE = 272.0
H_PT = 282.0


def configure(pitch: float = 18.0):
    """Re-derive the vertical stack from the row pitch."""
    global PITCH, ARM_DY, BLOCK_TOP, LEGEND_CY, LETTER_BASE, WIN_LABEL_BASE
    global STRIP_Y0, STRIP_Y1, DIV_LABEL_BASE, SCHEMA_HEAD_BASE, H_PT, ROW_Y
    PITCH = float(pitch)
    ARM_DY = 5.6 if PITCH >= 18.0 else 5.2
    BLOCK_TOP = BLOCK_BOT + 9 * PITCH
    LEGEND_CY = BLOCK_TOP + 13.0
    LETTER_BASE = BLOCK_TOP + 24.0
    WIN_LABEL_BASE = LETTER_BASE + 13.0
    STRIP_Y0 = WIN_LABEL_BASE + 9.0
    STRIP_Y1 = STRIP_Y0 + 14.0
    DIV_LABEL_BASE = STRIP_Y1 + 5.0
    SCHEMA_HEAD_BASE = DIV_LABEL_BASE + 13.0
    H_PT = SCHEMA_HEAD_BASE + 10.0
    ROW_Y = [BLOCK_TOP - PITCH * (i + 0.5) for i in range(9)]

# the two symmetric-log scales.  ONE pair of parameters for the whole figure.
LINTHRESH, LINSCALE = 0.03, 0.4425
B_XLIM = (-0.018, 7.0)      # both ends chosen so no interval cap lands on a spine
B_TICKS = [0.0, 0.03, 0.1, 0.3, 1.0, 3.0]
B_TICKLABELS = ["0", "0.03", "0.1", "0.3", "1", "3"]
D_XLIM = (-1.7, 0.02)
D_TICKS = [-1.0, -0.1, 0.0]
D_TICKLABELS = ["−1", "−0.1", "0"]

ROW_Y = [BLOCK_TOP - PITCH * (i + 0.5) for i in range(9)]

INVENTORY: list[dict] = []
INTERVALS: list[dict] = []          # for the occlusion report
CI_TEXTS: list = []
ASSET = "abc_window_support_20260908"
VARIANT = "fig2"


# ---------------------------------------------------------------------------
# 3.  Frozen-structure assertions
# ---------------------------------------------------------------------------

def frozen_structure():
    """Model order, arm order, every estimate's sign partition, and the fact that
    every value the figure plots lies inside the display window it is drawn in."""
    assert MODELS == CONTRACT_JSON["model_order"], "model order moved"
    assert ARMS_TOP_TO_BOTTOM == CONTRACT_JSON["arm_order_top_to_bottom"]
    rows = {r["model"]: r for r in CONTRACT_JSON["rows"]}
    assert list(rows) == MODELS

    partition = {"positive": [], "crosses": [], "negative": []}
    for m in MODELS:
        c = rows[m]
        for a in ARMS_TOP_TO_BOTTOM:
            r = row(m, estimand="T", window="full_step", control="paired", arm=a)
            k = c["A_estimates"][a]
            assert (r["mean"], r["ci_low"], r["ci_high"]) == (k["mean"], k["ci_low"], k["ci_high"])
            assert B_XLIM[0] <= r["ci_low"] and r["ci_high"] <= B_XLIM[1], (m, a)

        gap = row(m, estimand="residual_minus_arm", window="full_step",
                  arm="all_router", control="paired")
        assert fmt(gap["mean"]) == c["B_display"]["mean"]
        assert f"[{fmt(gap['ci_low'])}, {fmt(gap['ci_high'])}]" == c["B_display"]["ci"]
        if gap["ci_low"] > 0:
            partition["positive"].append(m)
        elif gap["ci_high"] < 0:
            partition["negative"].append(m)
        else:
            partition["crosses"].append(m)

        eq = row(m, estimand="gap_window_difference", arm="all_router", control="paired",
                 left="divergence_k", right="suffix_k")
        assert eq["ci_high"] < 0, (m, eq["ci_high"])       # all nine below zero
        assert D_XLIM[0] <= eq["ci_low"] and eq["ci_high"] <= D_XLIM[1], m

    assert len(partition["positive"]) == 6, partition
    assert partition["crosses"] == ["Lightning", "Nano"], partition
    assert partition["negative"] == ["GPT-OSS-120B"], partition
    return partition


def palette_ladder():
    def lightness(hexc):
        r, g, b = matplotlib.colors.to_rgb(hexc)
        return colorsys.rgb_to_hls(r, g, b)[1]
    for ground, mark, dark in ((BAND, ROUTER, ROUTER_DARK),
                               (BAND, ALLROUTER, ALLROUTER_DARK),
                               (BAND, RESIDUAL, RESIDUAL_DARK)):
        assert lightness(ground) > lightness(mark) > lightness(dark) > lightness(INK)
    order = [INK, MUTED, RULE, HAIR, GRID, WINDOW_FILL, BAND, WHITE]
    ls = [lightness(c) for c in order]
    assert ls == sorted(ls), ls
    assert craft.contrast_ratio(INK, WINDOW_FILL) >= 10.0
    assert craft.contrast_ratio(MUTED, WINDOW_FILL) >= 7.0


# ---------------------------------------------------------------------------
# 4.  Point-space helpers
# ---------------------------------------------------------------------------

def hline(fig, x0, x1, y, color, lw, z=2, dashes=None):
    ln = Line2D([x0, x1], [y, y], color=color, lw=lw, solid_capstyle="butt", zorder=z)
    ln.set_transform(craft.pt_transform(fig))
    if dashes is not None:
        ln.set_linestyle(dashes)
    fig.add_artist(ln)
    return ln


def vline(fig, x, y0, y1, color, lw, z=2, dashes=None):
    ln = Line2D([x, x], [y0, y1], color=color, lw=lw, solid_capstyle="butt", zorder=z)
    ln.set_transform(craft.pt_transform(fig))
    if dashes is not None:
        ln.set_linestyle(dashes)
    fig.add_artist(ln)
    return ln


def num_cell(fig, x_dec_pt, y_pt, s, size=SZ_BODY, color=INK, weight="normal", z=5):
    """`craft.num_cell`, but an integer stays an integer."""
    tr = craft.pt_transform(fig)
    head, dot, tail = str(s).partition(".")
    a = fig.text(x_dec_pt, y_pt, head, fontsize=size, color=color, fontweight=weight,
                 ha="right", va="baseline", zorder=z)
    a.set_transform(tr)
    if not dot:
        return (a,)
    b = fig.text(x_dec_pt, y_pt, dot + tail, fontsize=size, color=color,
                 fontweight=weight, ha="left", va="baseline", zorder=z)
    b.set_transform(tr)
    return (a, b)


def text_width(fig, s, size, weight="normal"):
    t = fig.text(0, 0, s, fontsize=size, fontweight=weight)
    w = t.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi * 72.0
    t.remove()
    return w


def keep(panel, model, series, r):
    INVENTORY.append(dict(asset=ASSET, variant=VARIANT, panel=panel, model=model,
                          series=series, mean=r["mean"], ci_low=r["ci_low"],
                          ci_high=r["ci_high"], N=r["N"], K=r["K"],
                          draws=r["draws"], seed=r["seed"]))


def symlog_pos(v, lo, hi, x0, x1):
    """Rendered x in points of value `v` on a symlog axis spanning [lo, hi]."""
    tr = SymmetricalLogTransform(10, LINTHRESH, LINSCALE)
    f = lambda z: float(tr.transform_non_affine(np.array([z], dtype=float))[0])  # noqa: E731
    return x0 + (f(v) - f(lo)) / (f(hi) - f(lo)) * (x1 - x0)


def note_interval(panel, model, series, r, lo, hi, x0, x1, ms):
    left = symlog_pos(r["ci_low"], lo, hi, x0, x1)
    right = symlog_pos(r["ci_high"], lo, hi, x0, x1)
    INTERVALS.append(dict(panel=panel, model=model, series=series,
                          length_pt=round(right - left, 3), marker_pt=ms,
                          occluded=bool(right - left < ms)))


# ---------------------------------------------------------------------------
# 5.  Panel A — the judged step and its two eight-token windows
# ---------------------------------------------------------------------------

def draw_schematic(fig):
    cell_w = (STRIP[1] - STRIP[0]) / N_CELLS
    h = STRIP_Y1 - STRIP_Y0

    def cx(i):
        return STRIP[0] + i * cell_w

    # the two windows, drawn first so the cell rules print over them
    for lo, hi in (DIV_WIN, SUF_WIN):
        craft.rounded_box(fig, cx(lo), STRIP_Y0, cx(hi) - cx(lo), h, r_pt=R_CHIP, z=1.6,
                          facecolor=WINDOW_FILL, edgecolor=MUTED, linewidth=LW_CARD)
    # the first causal divergence token
    r = Rectangle((cx(DIV_CELL), STRIP_Y0 + 0.45), cell_w, h - 0.9, facecolor=MUTED,
                  edgecolor="none", zorder=1.8)
    r.set_transform(craft.pt_transform(fig))
    fig.add_artist(r)
    # the step itself
    craft.rounded_box(fig, STRIP[0], STRIP_Y0, STRIP[1] - STRIP[0], h, r_pt=R_CHIP,
                      z=1.2, facecolor=WHITE, edgecolor=RULE, linewidth=LW_FRAME)
    for i in range(1, N_CELLS):
        if i in (DIV_CELL, DIV_CELL + 1):
            continue                       # the dark cell draws its own edges
        vline(fig, cx(i), STRIP_Y0, STRIP_Y1, HAIR, LW_GRID, z=1.9)

    # the row label, in the model-name column: the strip IS the full step
    craft.put_pt(fig, NAME_RIGHT, (STRIP_Y0 + STRIP_Y1) / 2.0 - CAP * SZ_BODY / 2.0,
                 "full step", size=SZ_BODY, color=INK, weight="bold", ha="right", z=5)

    # the divergence callout
    div_c = cx(DIV_CELL) + cell_w / 2.0
    vline(fig, div_c, STRIP_Y1, DIV_LABEL_BASE - 2.0, MUTED, LW_HAIRLINE, z=1.5)
    craft.put_pt(fig, div_c, DIV_LABEL_BASE, "first causal divergence", size=SZ_BODY,
                 color=MUTED, ha="center", z=5)

    # the two window names
    for (lo, hi), name in ((DIV_WIN, "divergence-8"), (SUF_WIN, "suffix-8")):
        craft.put_pt(fig, (cx(lo) + cx(hi)) / 2.0, WIN_LABEL_BASE, name, size=SZ_BODY,
                     color=INK, weight="bold", ha="center", z=5)
    return dict(cell_w_pt=round(cell_w, 4),
                div_window_pt=[round(cx(DIV_WIN[0]), 2), round(cx(DIV_WIN[1]), 2)],
                suf_window_pt=[round(cx(SUF_WIN[0]), 2), round(cx(SUF_WIN[1]), 2)],
                window_len_pt=round(cx(DIV_WIN[1]) - cx(DIV_WIN[0]), 3))


# ---------------------------------------------------------------------------
# 6.  The builder
# ---------------------------------------------------------------------------

def symlog_axes(fig, x0, x1, xlim, ticks, labels):
    ylo, yhi = -0.5 - AX_PAD / PITCH, 8.5 + AX_PAD / PITCH
    ax = fig.add_axes([x0 / W_PT, (BLOCK_BOT - AX_PAD) / H_PT, (x1 - x0) / W_PT,
                       (BLOCK_TOP - BLOCK_BOT + 2 * AX_PAD) / H_PT])
    ax.set_zorder(3)
    ax.patch.set_alpha(0.0)
    ax.set_xscale("symlog", linthresh=LINTHRESH, linscale=LINSCALE, base=10)
    ax.set_xlim(*xlim)
    ax.set_ylim(ylo, yhi)
    ax.invert_yaxis()
    ax.set_axisbelow(True)
    ax.set_xticks(ticks, labels)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color(RULE)
        s.set_linewidth(LW_FRAME)
    pad = (BLOCK_BOT - AX_PAD) - (TICK_BASE + BBOX_TOP)
    ax.tick_params(axis="x", direction="in", length=2.0, width=LW_HAIRLINE,
                   color=RULE, labelsize=SZ_BODY, labelcolor=INK, pad=pad)
    ax.tick_params(axis="y", length=0)
    # gridlines: dashed GRID at every tick except zero, which is the reference rule
    for t in ticks:
        if t == 0:
            continue
        ln = Line2D([t, t], [ylo, yhi], color=GRID, lw=LW_GRID,
                    solid_capstyle="butt", zorder=0.6)
        ln.set_linestyle(DASH_GRID)
        ax.add_line(ln)
    return ax, pad


def build_figure():
    partition = frozen_structure()
    palette_ladder()

    craft.apply_paper_rc(extra={"axes.unicode_minus": True, "savefig.pad_inches": 0.0,
                                "xtick.labelsize": SZ_BODY, "ytick.labelsize": SZ_BODY})
    fig = plt.figure(figsize=(W_PT / 72.0, H_PT / 72.0))
    fig.patch.set_facecolor(WHITE)
    fig.canvas.draw()

    schema = draw_schematic(fig)

    axb, pad_b = symlog_axes(fig, B_AX[0], B_AX[1], B_XLIM, B_TICKS, B_TICKLABELS)
    axd, pad_d = symlog_axes(fig, D_AX[0], D_AX[1], D_XLIM, D_TICKS, D_TICKLABELS)

    # zero reference: dashed where zero is just a value (B), solid where it is the
    # boundary of the claim (D).  Drawn under the marks; markers occlude it.
    ylo, yhi = -0.5 - AX_PAD / PITCH, 8.5 + AX_PAD / PITCH
    zb = Line2D([0, 0], [ylo, yhi], color=HAIR, lw=LW_ZERO, zorder=1.5)
    zb.set_linestyle(DASH_ZERO)
    axb.add_line(zb)
    axd.add_line(Line2D([0, 0], [ylo, yhi], color=HAIR, lw=LW_ZERO, zorder=1.5))

    # row separators: one rule per gutter, carried across every column
    for i in range(1, 9):
        y = BLOCK_TOP - PITCH * i
        hline(fig, 6.0, NAME_RIGHT + 4.0, y, GRID, LW_GRID, z=1.0)
        hline(fig, B_AX[1], D_AX[0], y, GRID, LW_GRID, z=1.0)
        for ax in (axb, axd):
            ln = Line2D([-1e9, 1e9], [i - 0.5, i - 0.5], color=GRID, lw=LW_GRID,
                        solid_capstyle="butt", zorder=0.6)
            ax.add_line(ln)

    max_ci_right = -1e9
    for i, m in enumerate(MODELS):
        yc = ROW_Y[i]
        craft.put_pt(fig, NAME_RIGHT, yc - CAP * SZ_BODY / 2.0, m, size=SZ_BODY,
                     color=INK, weight="bold", ha="right", z=5)

        # ---- panel B: full-step T, one shared symmetric-log scale --------------
        for j, arm in enumerate(ARMS_TOP_TO_BOTTOM):
            r = row(m, estimand="T", window="full_step", control="paired", arm=arm)
            st = STYLE[arm]
            dy = (ARM_DY - j * ARM_DY) / PITCH
            eb = axb.errorbar(r["mean"], i - dy,
                              xerr=[[r["mean"] - r["ci_low"]], [r["ci_high"] - r["mean"]]],
                              fmt=st["marker"], color=st["color"], ms=st["ms"],
                              mfc=st["mfc"], mec=mec_of(st), mew=st["mew"],
                              elinewidth=ELW, capsize=CAPSIZE, capthick=CAPTHICK,
                              zorder=3)
            eb.lines[0].set_zorder(3.5)
            keep("B full-step T, shared symmetric-log scale", m, arm, r)
            note_interval("B", m, arm, r, *B_XLIM, *B_AX, st["ms"])

        # ---- panel C: the exact numeric column (v3's column B, verbatim) -------
        gap = row(m, estimand="residual_minus_arm", window="full_step",
                  arm="all_router", control="paired")
        # ROUND 10 (PALETTE_SPEC §6.1): the sign class is carried by the neutral
        # ramp, never by a hue.  Excludes zero -> INK, spans zero -> RULE; which side
        # of zero is already printed, as a minus.
        colour = INK if (gap["ci_low"] > 0 or gap["ci_high"] < 0) else RULE
        base = yc - CAP * SZ_BODY / 2.0
        num_cell(fig, C_MEAN_DEC, base, fmt(gap["mean"]), size=SZ_BODY,
                 color=colour, weight="bold", z=5)
        ci = craft.put_pt(
            fig, C_CI_X, base,
            f"[{sign_pad(fmt(gap['ci_low']))}, {sign_pad(fmt(gap['ci_high']))}]",
            size=SZ_BODY, color=RULE, ha="left", z=5)
        CI_TEXTS.append(ci)
        keep("C residual minus all-router, full step", m, "residual_minus_arm", gap)

        # ---- panel D: the gap shift between the two equal-length windows ------
        eq = row(m, estimand="gap_window_difference", arm="all_router", control="paired",
                 left="divergence_k", right="suffix_k")
        ebd = axd.errorbar(eq["mean"], i,
                           xerr=[[eq["mean"] - eq["ci_low"]], [eq["ci_high"] - eq["mean"]]],
                           fmt=D_STYLE["marker"], color=D_STYLE["color"], ms=D_STYLE["ms"],
                           mfc=D_STYLE["mfc"], mec=mec_of(D_STYLE), mew=D_STYLE["mew"],
                           elinewidth=ELW, capsize=CAPSIZE, capthick=CAPTHICK, zorder=3)
        ebd.lines[0].set_zorder(3.5)
        keep("D gap shift, divergence-8 minus suffix-8", m, "gap_window_difference", eq)
        note_interval("D", m, "gap_window_difference", eq, *D_XLIM, *D_AX, D_STYLE["ms"])

    rend = fig.canvas.get_renderer()
    inv = craft.pt_transform(fig).inverted()
    max_ci_right = max(t.get_window_extent(rend).transformed(inv).x1 for t in CI_TEXTS)

    # ---- the legend --------------------------------------------------------
    half_len, cap_pt, pad_text, gap_pt = 6.5, 3.0, 2.8, 10.0
    widths = [2 * half_len + pad_text + text_width(fig, STYLE[a]["label"], SZ_BODY)
              for a in ARMS_TOP_TO_BOTTOM]
    x = B_AX[0]
    for a, w in zip(ARMS_TOP_TO_BOTTOM, widths):
        st = STYLE[a]
        cx = x + half_len
        hline(fig, x, x + 2 * half_len, LEGEND_CY, st["color"], ELW, z=4.5)
        for s in (x, x + 2 * half_len):
            vline(fig, s, LEGEND_CY - cap_pt / 2.0, LEGEND_CY + cap_pt / 2.0,
                  st["color"], ELW, z=4.5)
        dot = Line2D([cx], [LEGEND_CY], marker=st["marker"], ms=st["ms"], mfc=st["mfc"],
                     mec=mec_of(st), mew=st["mew"], ls="none", zorder=4.7)
        dot.set_transform(craft.pt_transform(fig))
        fig.add_artist(dot)
        # ROUND 10 (PALETTE_SPEC §3.4): the label is INK.  The three dark steps span
        # 0.16 L* and are ΔE00 2.06 apart for a protanope, so a coloured label is not
        # a key; the marker miniature to its left is.
        craft.put_pt(fig, x + 2 * half_len + pad_text, LEGEND_CY - CAP * SZ_BODY / 2.0,
                     st["label"], size=SZ_BODY, color=INK, z=4.6)
        x += w + gap_pt
    legend_right = x - gap_pt

    # ---- letter rows -------------------------------------------------------
    for lx, ly, letter, label in (
            (STRIP[0], SCHEMA_HEAD_BASE, "A", "One judged step, two windows"),
            (B_AX[0], LETTER_BASE, "B", "Full-step $T$"),
            (C_LEFT, LETTER_BASE, "C", "Residual − all-router"),
            (D_AX[0], LETTER_BASE, "D", "Gap shift")):
        craft.put_pt(fig, lx, ly, letter, size=SZ_LETTER, color=INK, weight="bold", z=5)
        craft.put_pt(fig, lx + text_width(fig, letter, SZ_LETTER, "bold") + 4.0, ly,
                     label, size=SZ_BODY, color=INK, weight="bold", z=5)

    # ---- axis notes: the scale, said in the figure --------------------------
    craft.put_pt(fig, (B_AX[0] + B_AX[1]) / 2.0, NOTE_BASE,
                 "symmetric log, linear within ±0.03", size=SZ_BODY, color=RULE,
                 ha="center", z=5)
    craft.put_pt(fig, (D_AX[0] + D_AX[1]) / 2.0, NOTE_BASE, "divergence-8 − suffix-8",
                 size=SZ_BODY,
                 color=RULE, ha="center", z=5)

    assert fig.get_size_inches()[0] == 5.5
    fig._rv = dict(partition=partition, schema=schema, tick_pad=[pad_b, pad_d],
                   legend_right=legend_right, ci_right=max_ci_right)
    return fig


# ---------------------------------------------------------------------------
# 7.  Reports
# ---------------------------------------------------------------------------

#: The same measurement taken on the shipped v3 figure, recomputed from the frozen
#: JSON and v3's own geometry (column A width 80 pt over nine per-row `A_xlim`;
#: column C width 108 pt over xlim (-1.22, 0.08)).  Reproduces the round-4 brief's
#: nine router lengths exactly.
BEFORE_V3 = dict(
    n_intervals=36, marker_pt=3.6, n_shorter_than_marker=9,
    single_layer_router_shorter="8 of 9",
    router_lengths_pt=[0.69, 1.02, 1.08, 1.09, 1.79, 2.18, 2.58, 2.95, 4.74],
    also_short=[dict(panel="C", model="Nano", length_pt=2.97)],
    min_length_pt=0.69,
)


def occlusion_report(fig=None):
    """Rendered interval length vs marker diameter, for every plotted interval."""
    short = [d for d in INTERVALS if d["occluded"]]
    return dict(before_v3=BEFORE_V3,
                n_intervals=len(INTERVALS),
                n_shorter_than_marker=len(short),
                shorter_than_marker=short,
                min_length_pt=round(min(d["length_pt"] for d in INTERVALS), 3),
                min_owner=min(INTERVALS, key=lambda d: d["length_pt"]),
                max_length_pt=round(max(d["length_pt"] for d in INTERVALS), 3))


#: The measurement the round-4 fix exists to move: plan A as built, at the shipped
#: pitch of 17 (ARM_DY 4.2, marker 3.6 pt).
BEFORE_SEPARATION = {
    "Lightning":    dict(dx_pt=-0.33, dy_pt=-4.2, centre_pt=4.21, edge_gap_pt=0.61),
    "Nano":         dict(dx_pt=-2.15, dy_pt=-4.2, centre_pt=4.72, edge_gap_pt=1.12),
    "GPT-OSS-120B": dict(dx_pt=-6.34, dy_pt=-4.2, centre_pt=7.61, edge_gap_pt=4.01),
    "_params": dict(arm_dy_pt=4.2, marker_pt=3.6),
}


def separation_report():
    """Centre-to-centre distance, in points on the printed page, between the
    all-router square and the single-layer residual triangle in every row.

    `edge_gap_pt` is that distance minus one marker diameter: the width of the white
    channel between the two drawn marks, measured along the line joining them.  It is
    the number that decides whether a reader sees two marks or one clump; the centre
    distance alone can look healthy while the marks still touch.
    """
    j = {a: k for k, a in enumerate(ARMS_TOP_TO_BOTTOM)}
    rows = {}
    for m in MODELS:
        x = {}
        for arm in ("all_router", "post_moe_residual"):
            r = row(m, estimand="T", window="full_step", control="paired", arm=arm)
            x[arm] = symlog_pos(r["mean"], *B_XLIM, *B_AX)
        dx = x["post_moe_residual"] - x["all_router"]
        dy = (ARM_DY - j["post_moe_residual"] * ARM_DY) - (ARM_DY - j["all_router"] * ARM_DY)
        centre = math.hypot(dx, dy)
        rows[m] = dict(dx_pt=round(dx, 3), dy_pt=round(dy, 3),
                       centre_pt=round(centre, 3),
                       edge_gap_pt=round(centre - MS, 3))
    return dict(params=dict(arm_dy_pt=ARM_DY, marker_pt=MS, row_pitch_pt=PITCH),
                before=BEFORE_SEPARATION,
                after=rows,
                min_centre_pt=round(min(v["centre_pt"] for v in rows.values()), 3),
                min_edge_gap_pt=round(min(v["edge_gap_pt"] for v in rows.values()), 3),
                min_owner=min(rows, key=lambda k: rows[k]["centre_pt"]),
                # row grouping must survive the wider offset: the nearest mark in the
                # next row has to stay further away than the next arm in this row.
                within_row_step_pt=round(ARM_DY, 3),
                between_row_step_pt=round(PITCH - 2 * ARM_DY, 3),
                gutter_clearance_pt=round(PITCH / 2.0 - ARM_DY - MS / 2.0, 3))


def geometry_report(fig):
    texts = [t for t in fig.findobj(matplotlib.text.Text)
             if t.get_visible() and t.get_text().strip()]
    sizes = sorted({round(t.get_fontsize(), 2) for t in texts})
    bold_sizes = sorted({round(t.get_fontsize(), 2) for t in texts
                         if str(t.get_fontweight()) in ("bold", "700")})
    lws = sorted({round(float(a.get_linewidth()), 2)
                  for a in fig.findobj(matplotlib.lines.Line2D)
                  if a.get_visible() and a.get_linewidth()})
    for ax in fig.axes:
        for s in ax.spines.values():
            if s.get_visible():
                lws.append(round(float(s.get_linewidth()), 2))
    lws = sorted(set(lws))
    radii = sorted({round(float(p.get_boxstyle().rounding_size), 2)
                    for p in fig.findobj(matplotlib.patches.FancyBboxPatch)
                    if hasattr(p.get_boxstyle(), "rounding_size")
                    and (p.get_alpha() is None or p.get_alpha() >= 1.0)})
    text_colors = sorted({matplotlib.colors.to_hex(t.get_color()).upper() for t in texts})
    rv = fig._rv
    return dict(
        variant=VARIANT, plan="A (round-4 merged: in-row separation fix)",
        size_in=list(fig.get_size_inches()), size_pt=[W_PT, H_PT],
        n_text=len(texts), distinct_font_sizes=sizes, bold_sizes=bold_sizes,
        min_font_pt=min(sizes), max_font_pt=max(sizes), text_colors=text_colors,
        radii_pt=radii, stroke_widths_pt=lws,
        scale=dict(kind="symlog", base=10, linthresh=LINTHRESH, linscale=LINSCALE,
                   B_xlim=list(B_XLIM), B_ticks=B_TICKS,
                   D_xlim=list(D_XLIM), D_ticks=D_TICKS),
        row_pitch_pt=PITCH, block_pt=[BLOCK_BOT, BLOCK_TOP],
        panel_B_pt=list(B_AX), panel_D_pt=list(D_AX),
        mean_decimal_x_pt=C_MEAN_DEC, ci_left_x_pt=C_CI_X,
        ci_right_ink_pt=round(rv["ci_right"], 3),
        legend_right_pt=round(rv["legend_right"], 3),
        schematic=rv["schema"], tick_pad_pt=[round(p, 3) for p in rv["tick_pad"]],
        n_estimates=len(INVENTORY), sign_partition=rv["partition"],
        occlusion=occlusion_report(fig),
        separation=separation_report(),
    )




