"""craft.py -- verified Matplotlib craft helpers for 5.5 in paper figures.

Every helper in this module was rendered to a vector PDF with matplotlib 3.10.8 on
macOS 15 (Darwin 25.6.0), rasterised with `pdftoppm -r 400..1200` and inspected before
being written down.  The companion write-up is `matplotlib_cookbook.md`; the visual
proof sheet is `cookbook_samples.pdf`.

Design contract
---------------
* Importing this module has NO side effects: it does not touch `matplotlib.rcParams`,
  does not create figures, and does not read or write files.  Call `apply_paper_rc()`
  explicitly.
* Every geometric argument named `*_pt` is in **typographic points** (1 pt = 1/72 in)
  measured from the figure's lower-left corner, so the drawing is identical whatever
  `figure.dpi` or `savefig.dpi` happen to be.
* Nothing here rasterises: with these helpers a saved PDF contains zero image
  XObjects.  The two exceptions are called out in their docstrings
  (`gradient_box_raster`, and `halo()` applied to `Text`, which converts glyphs to
  outlines but stays vector).

Quick start
-----------
    import matplotlib.pyplot as plt
    import craft

    craft.apply_paper_rc()
    fig = plt.figure(figsize=(5.5, 2.9))
    craft.card(fig, 12, 12, 120, 40, r_pt=3.0, fc="#EDF2FB", ec="#2E5AA8")
    fig.savefig("out.pdf")          # never bbox_inches="tight"
"""

from __future__ import annotations

import colorsys

import numpy as np
import matplotlib.patheffects as _pe
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_hex, to_rgb
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, FancyArrowPatch, FancyBboxPatch
from matplotlib.transforms import Affine2D

__all__ = [
    "PAPER_RC", "TIMES_MATH_RC", "apply_paper_rc",
    "pt_transform", "axes_rect_pt", "pt_to_fig",
    "rounded_box", "card", "pill", "soft_shadow",
    "gradient_cmap", "gradient_box_vector", "gradient_box_raster",
    "panel_ground", "row_bands",
    "style_legend", "HandlerMarkerInterval",
    "arrow", "dashed_arrow", "connect_axes", "ARROW_HEAD_FRACTION",
    "halo", "text_plate", "small_caps", "num_cell", "put_pt",
    "CAP_HEIGHT_EM", "row_axes",
    "SEMANTIC", "tint", "shade", "ramp", "relative_luminance", "contrast_ratio",
    "luma", "grayscale", "set_luma", "spread_luma", "gray_gap", "palette_report",
]

# --------------------------------------------------------------------------
# 0.  rcParams
# --------------------------------------------------------------------------

#: Baseline rcParams for an ICLR-style Times figure embedded at 5.5 in, scale 1.
#: `pdf.fonttype: 42` gives subsetted CID TrueType (Type 42) faces, which is what
#: `pdffonts` reports as "CID TrueType ... emb yes sub yes".
PAPER_RC = {
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIX Two Text"],
    "mathtext.fontset": "stix",
    "font.size": 8.0,
    "axes.labelsize": 8.0,
    "axes.titlesize": 8.0,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "xtick.major.size": 2.4,
    "ytick.major.size": 2.4,
    "xtick.minor.size": 1.2,
    "ytick.minor.size": 1.2,
    "lines.linewidth": 1.0,
    "lines.markersize": 3.0,
    "grid.linewidth": 0.4,
    "legend.frameon": True,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "path.simplify": False,
}

#: Optional overlay: render mathtext in Times New Roman itself, so the PDF embeds
#: only Times faces (no STIXGeneral, no DejaVu).  Greek and the common relations all
#: resolve from Times New Roman + Times New Roman Italic.  Caveats: big operators
#: (\sum, \int) still pull in Cmex10, and superscripts sit slightly high -- if a
#: figure needs those, keep `mathtext.fontset: "stix"` instead.
TIMES_MATH_RC = {
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
    "mathtext.cal": "Times New Roman:italic",
    "mathtext.sf": "Times New Roman",
}


def apply_paper_rc(times_math: bool = False, extra: dict | None = None) -> dict:
    """Push `PAPER_RC` (and optionally `TIMES_MATH_RC`) into `plt.rcParams`.

    Returns the dict that was applied, so a caller can log it.  This is the only
    function in the module that mutates global state, and it is never called on
    import.
    """
    rc = dict(PAPER_RC)
    if times_math:
        rc.update(TIMES_MATH_RC)
    if extra:
        rc.update(extra)
    plt.rcParams.update(rc)
    return rc


# --------------------------------------------------------------------------
# 1.  Point-space geometry
# --------------------------------------------------------------------------

def pt_transform(fig):
    """A transform whose data units are POINTS from the figure's lower-left corner.

    This is the single most useful object in the module.  `fig.dpi_scale_trans` maps
    inches to display units at draw time, so scaling by 1/72 in front of it gives a
    point-space that is exact and isotropic in both axes at any dpi.  Use it for every
    container, connector and hand-placed label:

        tr = craft.pt_transform(fig)
        fig.add_artist(Rectangle((12, 12), 120, 40, transform=tr))
    """
    return Affine2D().scale(1.0 / 72.0) + fig.dpi_scale_trans


def pt_to_fig(fig, x_pt, y_pt):
    """Convert a point-space position to figure fraction, for `fig.text(...)`."""
    w_pt, h_pt = fig.get_size_inches() * 72.0
    return x_pt / w_pt, y_pt / h_pt


def axes_rect_pt(fig, ax):
    """`(x, y, w, h)` of an Axes in POINTS.

    Uses `ax.get_position()` (figure fraction) times the figsize, so it is exact and
    needs no renderer -- unlike `ax.get_window_extent()`, which is in display pixels
    and therefore dpi-dependent.
    """
    w_pt, h_pt = fig.get_size_inches() * 72.0
    b = ax.get_position()
    return b.x0 * w_pt, b.y0 * h_pt, b.width * w_pt, b.height * h_pt


def put_pt(fig, x_pt, y_pt, s, size=8.0, color="#1F2937", weight="normal",
           ha="left", va="baseline", z=5, **kw):
    """`fig.text` positioned in POINTS. Returns the Text."""
    t = fig.text(0, 0, s, fontsize=size, color=color, fontweight=weight, ha=ha,
                 va=va, zorder=z, **kw)
    t.set_transform(pt_transform(fig))
    t.set_position((x_pt, y_pt))
    return t


# --------------------------------------------------------------------------
# 2.  Rounded containers
# --------------------------------------------------------------------------

def rounded_box(fig, x_pt, y_pt, w_pt, h_pt, r_pt=2.5, z=2, **kw):
    """A `FancyBboxPatch` of exactly `w_pt x h_pt` with a true circular corner.

    `boxstyle="round"` multiplies `pad` and `rounding_size` by `mutation_scale`
    (default 1) **in the patch's own coordinate system**.  Drawn in an Axes, that
    system is the data space, which is almost never square -- so a "round" corner
    comes out as an ellipse.  Drawing in point-space fixes both problems at once, and
    `pad=0` makes the patch the size you asked for instead of the size plus padding.

    Tested radii at 5.5 in: 1.5 pt reads as a chamfer, **2.5 pt is the default card
    radius**, 4 pt for an outer grouping ground, `h/2` for a pill.
    """
    p = FancyBboxPatch((x_pt, y_pt), w_pt, h_pt,
                       boxstyle=f"round,pad=0,rounding_size={r_pt}",
                       transform=pt_transform(fig), joinstyle="round", zorder=z, **kw)
    fig.add_artist(p)
    return p


def card(fig, x_pt, y_pt, w_pt, h_pt, r_pt=2.5, fc="#FFFFFF", ec="#C9D4E6",
         lw=0.6, z=2, **kw):
    """A white/tinted card with a hairline stroke.

    0.6 pt is the thinnest stroke that still reads as a deliberate outline at 5.5 in
    (0.3 pt disappears against a tint; 1.0 pt starts to compete with the data).
    """
    return rounded_box(fig, x_pt, y_pt, w_pt, h_pt, r_pt, z,
                       facecolor=fc, edgecolor=ec, linewidth=lw, **kw)


def pill(fig, x_pt, y_pt, w_pt, h_pt, fc="#2E5AA8", ec="none", lw=0.0, z=3, **kw):
    """A stadium-shaped header band: the same box with `r = h/2`."""
    return rounded_box(fig, x_pt, y_pt, w_pt, h_pt, h_pt / 2.0, z,
                       facecolor=fc, edgecolor=ec, linewidth=lw, **kw)


def soft_shadow(fig, x_pt, y_pt, w_pt, h_pt, r_pt=3.0, n=5, dx=0.0, dy=-0.9,
                spread=2.2, alpha=0.055, color="#1B2A41", z=1):
    """A soft drop shadow built from `n` nested alpha patches. 100 % vector.

    Matplotlib has no vector blur.  `patheffects.withSimplePatchShadow` draws ONE
    hard-edged offset copy, which at 5.5 in reads as a smudge under the bottom edge.
    Stacking `n` rounded rects, each grown by `spread*(1-i/n)` pt and each contributing
    `alpha`, gives an opacity ramp from `alpha` at the outer edge to `n*alpha` under
    the card, which is what a blur looks like at this size.

    Verified settings (400 dpi and 900 dpi inspection, 74 x 40 pt card):
      * `n=5, spread=2.2, dy=-0.9, alpha=0.055`  -- the default "raised card"
      * `n=2, spread=1.0, dy=-0.6, alpha=0.09`   -- a tight contact shadow
      * `alpha > 0.08` at n=5 (peak 0.40) starts to look dirty on white
    Cost: 5 extra patches per card, ~0.36 kB each in the PDF.
    """
    tr = pt_transform(fig)
    out = []
    for i in range(n):
        t = (i + 1) / n
        grow = spread * (1.0 - t)
        p = FancyBboxPatch((x_pt - grow + dx * t, y_pt - grow + dy * t),
                           w_pt + 2 * grow, h_pt + 2 * grow,
                           boxstyle=f"round,pad=0,rounding_size={r_pt + grow}",
                           transform=tr, facecolor=color, edgecolor="none",
                           linewidth=0, alpha=alpha, zorder=z)
        fig.add_artist(p)
        out.append(p)
    return out


# --------------------------------------------------------------------------
# 3.  Gradients
# --------------------------------------------------------------------------

def gradient_cmap(colors, name="craft_grad", n=256):
    """A `LinearSegmentedColormap` from an ordered list of hex stops."""
    return LinearSegmentedColormap.from_list(name, list(colors), N=n)


def gradient_box_vector(fig, x_pt, y_pt, w_pt, h_pt, cmap, r_pt=3.0, n=160,
                        vertical=False, ec="none", lw=0.0, z=3):
    """A gradient-filled rounded box made of `n` clipped strips. No raster XObject.

    Measured band visibility on a 78 pt wide box at 400 dpi:
      n=12 (6.5 pt/band) obvious steps; n=32 (2.4 pt) visible; n=80 (0.97 pt) faint;
      **n>=160 (<=0.5 pt/band) indistinguishable from a continuous ramp**.
    Each strip is widened by 3 % so adjacent fills overlap and no hairline seams show.
    Cost: about 5.9 kB and 12 ms per 118 x 28 pt box at n=160.
    """
    tr = pt_transform(fig)
    clip = rounded_box(fig, x_pt, y_pt, w_pt, h_pt, r_pt, z,
                       facecolor="none", edgecolor="none")
    if vertical:
        step = h_pt / n
        for i in range(n):
            c = cmap(i / (n - 1))
            b = plt.Rectangle((x_pt, y_pt + i * step), w_pt, step * 1.03,
                              transform=tr, facecolor=c, edgecolor=c, linewidth=0.0,
                              zorder=z)
            fig.add_artist(b)
            b.set_clip_path(clip)
    else:
        step = w_pt / n
        for i in range(n):
            c = cmap(i / (n - 1))
            b = plt.Rectangle((x_pt + i * step, y_pt), step * 1.03, h_pt,
                              transform=tr, facecolor=c, edgecolor=c, linewidth=0.0,
                              zorder=z)
            fig.add_artist(b)
            b.set_clip_path(clip)
    if ec != "none":
        rounded_box(fig, x_pt, y_pt, w_pt, h_pt, r_pt, z + 1,
                    facecolor="none", edgecolor=ec, linewidth=lw)
    return clip


def gradient_box_raster(fig, x_pt, y_pt, w_pt, h_pt, cmap, r_pt=3.0, ec="none",
                        lw=0.0, z=3, nx=1024):
    """`imshow` + `set_clip_path`.  **This embeds one raster image XObject.**

    Kept for completeness and because it is the standard recipe, but at paper scale
    it is the worse option: the embedded image is resampled to `savefig.dpi` exactly
    (measured: dpi 200 -> 200 ppi, 600 -> 600 ppi, 1200 -> 1200 ppi) and written with
    `/Interpolate false`, so a 200-dpi save prints a visibly coarse ramp.  Prefer
    `gradient_box_vector`.

    The clip patch is built in point-space and handed to `set_clip_path`, which is
    what makes the corners truly circular -- clipping with a patch in axes-fraction
    coordinates produces the same elliptical corners as any other data-unit radius.
    """
    w_fig, h_fig = fig.get_size_inches() * 72.0
    ax = fig.add_axes([x_pt / w_fig, y_pt / h_fig, w_pt / w_fig, h_pt / h_fig],
                      zorder=z)
    ax.set_axis_off()
    ax.patch.set_visible(False)
    im = ax.imshow(np.linspace(0, 1, nx).reshape(1, -1), aspect="auto", cmap=cmap,
                   extent=(0, 1, 0, 1), interpolation="bilinear", zorder=z)
    clip = rounded_box(fig, x_pt, y_pt, w_pt, h_pt, r_pt, z,
                       facecolor="none", edgecolor="none")
    im.set_clip_path(clip)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    if ec != "none":
        rounded_box(fig, x_pt, y_pt, w_pt, h_pt, r_pt, z + 1,
                    facecolor="none", edgecolor=ec, linewidth=lw)
    return ax


# --------------------------------------------------------------------------
# 4.  Grouping grounds  (the round-1 "holes" bug and its two fixes)
# --------------------------------------------------------------------------

def panel_ground(fig, ax, pad_pt=(14, 8, 10, 10), r_pt=4.0, fc="#F5F7FA", ec="none",
                 lw=0.6, z=-1):
    """A pale rounded ground behind an Axes, as a FIGURE-level artist.

    `pad_pt` is `(left, right, bottom, top)` in points, measured out from the Axes
    rectangle -- give the left side room for the tick labels.

    **Call `ax.set_facecolor("none")` (or `ax.patch.set_visible(False)`) first.**  An
    Axes patch is opaque white by default and is drawn after any figure artist at a
    lower zorder, so it punches a white hole exactly the size of the plotting area out
    of the ground.  That was the round-1 bug; there is no zorder value that fixes it,
    because the hole is a fill, not a stacking accident.
    """
    l, rr, b, t = pad_pt
    x, y, w, h = axes_rect_pt(fig, ax)
    return rounded_box(fig, x - l, y - b, w + l + rr, h + b + t, r_pt, z,
                       facecolor=fc, edgecolor=ec, linewidth=lw)


def row_bands(ax, n_rows, color="#EDF2FB", every=2, offset=0, half=0.5, z=0.5):
    """Alternating pale bands behind categorical rows, INSIDE the Axes.

    The second fix for the same problem: `axhspan` draws in data space, so it can
    never be occluded by the Axes patch.  Pair it with `ax.set_axisbelow(True)` so the
    grid and spines still sit above the band but below the data.
    """
    ax.set_axisbelow(True)
    out = []
    for i in range(n_rows):
        if (i + offset) % every == 0:
            out.append(ax.axhspan(i - half, i + half, facecolor=color,
                                  edgecolor="none", zorder=z))
    return out


# --------------------------------------------------------------------------
# 5.  Legends
# --------------------------------------------------------------------------

def style_legend(leg, r_pt=2.5, ec="#9AA5B1", lw=0.7, fc="#FFFFFF", alpha=1.0):
    """Give a legend a designed frame.

    The legend frame is a `FancyBboxPatch` whose `mutation_scale` is the legend's font
    size, so `rounding_size` is in FONT-SIZE units: pass `r_pt / fontsize` to get a
    radius in points.  `pad` is likewise a multiple of the font size and is separate
    from the `borderpad` layout argument -- setting the boxstyle does not move the
    text, only the frame.

    Reference-paper values that read well at 5.5 in: `r_pt=2.5`, `ec="#9AA5B1"` or
    "#7A7A7A", `lw=0.7-0.96`, `fc="#FFFFFF"` at `alpha=0.92`, plus legend kwargs
    `borderpad=0.42, labelspacing=0.30, handlelength=1.5, handletextpad=0.5`.
    """
    fr = leg.get_frame()
    texts = leg.get_texts()
    fs = texts[0].get_fontsize() if texts else 8.0
    fr.set_boxstyle("round", pad=0.35, rounding_size=r_pt / fs)
    fr.set_edgecolor(ec)
    fr.set_linewidth(lw)
    fr.set_facecolor(fc)
    fr.set_alpha(alpha)
    return leg


class HandlerMarkerInterval(HandlerBase):
    """A legend glyph that is a true miniature of a marker-plus-interval data mark.

    `matplotlib.legend_handler.HandlerErrorbar` works and is the right default when
    you can pass the real `ErrorbarContainer` as the handle, but it sizes the glyph in
    fractions of the legend box, so the glyph drifts as the legend is re-laid out.
    This handler places the bar, the two caps and the centre marker in POINTS relative
    to the glyph's centre, so it is identical at any figsize and matches the data mark
    exactly.

        proxy = Line2D([], [], ls="none")
        ax.legend(handles=[proxy], labels=["effect +/- 95% CI"],
                  handler_map={proxy: craft.HandlerMarkerInterval("#2E5AA8")})
    """

    def __init__(self, color, half_len_pt=6.5, cap_pt=2.6, lw=0.9, marker="o",
                 ms=3.2, mfc="white", mew=0.9, **kw):
        self.color = color
        self.half_len_pt = half_len_pt
        self.cap_pt = cap_pt
        self.lw = lw
        self.marker = marker
        self.ms = ms
        self.mfc = mfc
        self.mew = mew
        super().__init__(**kw)

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height,
                       fontsize, trans):
        cx = xdescent + width / 2.0
        cy = ydescent + height / 2.0
        bar = Line2D([cx - self.half_len_pt, cx + self.half_len_pt], [cy, cy],
                     color=self.color, lw=self.lw, solid_capstyle="butt")
        caps = [Line2D([cx + s * self.half_len_pt, cx + s * self.half_len_pt],
                       [cy - self.cap_pt / 2, cy + self.cap_pt / 2],
                       color=self.color, lw=self.lw, solid_capstyle="butt")
                for s in (-1, 1)]
        dot = Line2D([cx], [cy], marker=self.marker, ms=self.ms, mfc=self.mfc,
                     mec=self.color, mew=self.mew, ls="none")
        artists = [bar, *caps, dot]
        for a in artists:
            a.set_transform(trans)
        return artists


# --------------------------------------------------------------------------
# 6.  Connectors
# --------------------------------------------------------------------------

#: `arrowstyle="-|>"` draws a head whose length is `ARROW_HEAD_FRACTION *
#: mutation_scale` points (matplotlib's `ArrowStyle._Curve` defaults head_length=0.4,
#: head_width=0.2).  Used by `dashed_arrow` to size the solid tip.
ARROW_HEAD_FRACTION = 0.4


def arrow(fig, p0_pt, p1_pt, style="-|>", ms=7.0, lw=0.8, color="#3B4554",
          cs="arc3,rad=0", shrinkA=0.0, shrinkB=0.0, z=5, **kw):
    """A `FancyArrowPatch` in point-space.

    `mutation_scale` is in points and is the only thing that sets head size; the head
    is a filled polygon and does not scale with `linewidth`.  Judged at 5.5 in:
    ms=3 is a hairline tick, **ms=5 is the smallest head that still reads as an
    arrow, ms=7 with lw=0.8 is the balanced pair**, ms=9 is emphatic, ms=12 is
    cartoonish.  Shaft weights: 0.5 pt is too faint under a ms=7 head, 0.8-0.9 pt
    balances it, 1.6 pt makes the head look undersized.

    `shrinkA`/`shrinkB` are in points and are the clean way to leave a gap between a
    node's edge and the arrow -- prefer them over shortening the endpoints by hand.

    `connectionstyle` notes: `arc3,rad=0.2..0.3` for a gentle routed edge,
    `angle3,angleA=0,angleB=90` for an L that leaves horizontally and arrives
    vertically, `angle,...,rad=3` for a right-angled elbow with a 3 pt fillet.
    `bar,fraction=...` was tried and rejected for node-to-node routing: on a diagonal
    it produces a kinked path that reads as a mistake.
    """
    a = FancyArrowPatch(p0_pt, p1_pt, transform=pt_transform(fig), arrowstyle=style,
                        mutation_scale=ms, linewidth=lw, color=color,
                        connectionstyle=cs, shrinkA=shrinkA, shrinkB=shrinkB,
                        joinstyle="miter", capstyle="butt", zorder=z, **kw)
    fig.add_artist(a)
    return a


def dashed_arrow(fig, p0_pt, p1_pt, dashes=(3.0, 2.0), ms=7.0, lw=0.8,
                 color="#3B4554", z=5):
    """A dashed shaft with a SOLID head, as two artists.

    Setting `linestyle` on a `FancyArrowPatch` applies the dash pattern to the head's
    outline stroke as well as the shaft, and at 5.5 in that chews the head into a
    ragged blob (verified at 1200 dpi).  There is no flag to exempt the head, so the
    fix is to draw a plain dashed connector that stops `ARROW_HEAD_FRACTION * ms`
    points short, then a solid stub carrying the head.

    Suggested semantics, all at 0.8 pt: solid = a real dependency or data flow;
    `(3.0, 2.0)` = hypothesised or unmeasured; `(0.8, 1.6)` = a reference or grouping
    line, not a flow; `(5.0, 1.6, 1.0, 1.6)` = a control or counterfactual path.
    """
    v = np.asarray(p1_pt, float) - np.asarray(p0_pt, float)
    length = float(np.hypot(*v))
    head = ARROW_HEAD_FRACTION * ms
    q = tuple(np.asarray(p1_pt, float) - v / length * head)
    shaft = FancyArrowPatch(p0_pt, q, transform=pt_transform(fig), arrowstyle="-",
                            mutation_scale=ms, linewidth=lw, color=color,
                            capstyle="butt", zorder=z)
    shaft.set_linestyle((0, tuple(dashes)))
    fig.add_artist(shaft)
    tip = FancyArrowPatch(q, p1_pt, transform=pt_transform(fig), arrowstyle="-|>",
                          mutation_scale=ms, linewidth=lw, color=color,
                          joinstyle="miter", zorder=z)
    fig.add_artist(tip)
    return shaft, tip


def connect_axes(fig, axA, xyA, axB, xyB, ms=7.0, lw=0.8, color="#3B4554",
                 cs="arc3,rad=-0.18", shrinkA=2.0, shrinkB=3.0, style="-|>", z=10):
    """A `ConnectionPatch` between two Axes, anchored in each Axes' DATA coordinates.

    Add it to the FIGURE (not to an Axes): a ConnectionPatch owned by an Axes is
    clipped to that Axes' box, so the half that crosses the gap disappears.
    """
    cp = ConnectionPatch(xyA=xyA, coordsA=axA.transData, xyB=xyB,
                         coordsB=axB.transData, arrowstyle=style, mutation_scale=ms,
                         linewidth=lw, color=color, connectionstyle=cs,
                         shrinkA=shrinkA, shrinkB=shrinkB, zorder=z)
    fig.add_artist(cp)
    return cp


# --------------------------------------------------------------------------
# 7.  Text craft
# --------------------------------------------------------------------------

#: Cap height of Times New Roman, in em.  `y_baseline = y_centre - CAP_HEIGHT_EM*size/2`
#: optically centres a descender-free label in a box.
CAP_HEIGHT_EM = 0.662

#: FIGURE SPACE.  Measured at exactly 0.5000 em in Times New Roman -- the same advance
#: as a digit -- so it is the correct pad character for a signed numeric column.
FIGURE_SPACE = " "


def halo(lw=2.0, fg="white"):
    """Path effects for a haloed label: `[withStroke(lw, fg), Normal()]`.

    `linewidth` is the TOTAL stroke width, half of it outside the glyph, so 2.0 pt
    buys a 1.0 pt halo.  At 8 pt Times over a 1.2 pt rule: 1.2 pt is not enough,
    **2.0 pt is the working value**, 3.0 pt starts to fatten the letterforms.

    Important cost: applying a path effect to a `Text` makes matplotlib emit the
    glyphs as vector OUTLINES.  The PDF then contains no font for that string and
    `pdftotext` extracts nothing -- which will silently defeat a QA gate that reads
    numbers out of the text layer, and roughly doubles those strings' bytes.  It stays
    vector, but for any number a checker must read, use `text_plate()` instead.
    """
    return [_pe.withStroke(linewidth=lw, foreground=fg), _pe.Normal()]


def text_plate(fc="white", alpha=1.0, pad=0.12, r=0.2, ec="none"):
    """A `bbox=` dict giving a label an opaque plate instead of a stroke halo.

    Keeps the glyphs as real, searchable text (verified with `pdftotext`) and costs
    about 0.07 kB per label, versus roughly 0.6 kB for a `withStroke` halo.  `pad` and
    `r` are in font-size units, as everywhere in boxstyle.
    """
    return dict(boxstyle=f"round,pad={pad},rounding_size={r}", facecolor=fc,
                edgecolor=ec, alpha=alpha)


def small_caps(fig, x_pt, y_pt, text, size=8.0, ratio=0.78, tracking=0.30,
               color="#1F2937", weight="normal", z=5, **kw):
    """Faux small caps: word-initial letters at `size`, the rest uppercase at
    `size*ratio`, with `tracking` points of extra letter spacing.

    Times New Roman has no small-caps face and matplotlib has no letter-spacing
    property, so the string is drawn one Text per character with the advance measured
    from the renderer.  Returns `(artists, width_pt)`.

    Inspected at 5.5 in: `ratio=0.78, tracking=0.30` is a quiet label; `ratio=0.72,
    tracking=0.55` with `weight="bold"` is a strong section marker.  Below ratio 0.70
    the small letters drop under the 7.5 pt legibility floor at 8 pt nominal.
    """
    tr = pt_transform(fig)
    renderer = fig.canvas.get_renderer()
    dpi = fig.get_dpi()
    cx = x_pt
    artists = []
    for i, ch in enumerate(text):
        initial = ch.isupper() or i == 0 or text[i - 1] == " "
        t = fig.text(0, 0, ch.upper(), fontsize=size if initial else size * ratio,
                     color=color, fontweight=weight, ha="left", va="baseline",
                     zorder=z, **kw)
        t.set_transform(tr)
        t.set_position((cx, y_pt))
        cx += t.get_window_extent(renderer).width / dpi * 72.0 + tracking
        artists.append(t)
    return artists, cx - x_pt - tracking


def num_cell(fig, x_dec_pt, y_pt, s, size=8.0, color="#1F2937", weight="normal", z=5):
    """Decimal-aligned numeric cell: the part before '.' is right-aligned at
    `x_dec_pt`, the fraction is left-aligned at the same x.

    Times New Roman digits are tabular -- every digit advances exactly 0.5000 em in
    both regular and bold (measured to 4 decimal places), and '.' is 0.2503 em -- so a
    plain `ha="right"` already aligns a column whose entries all carry the same number
    of decimals.  Two things break that: a mixed number of decimals, and the sign.
    U+2212 MINUS is 0.5642 em against the hyphen's 0.3330 em, so a signed column does
    not align on a right edge.  Splitting at the decimal point fixes both.
    """
    tr = pt_transform(fig)
    head, _, tail = str(s).partition(".")
    a = fig.text(x_dec_pt, y_pt, head, fontsize=size, color=color, fontweight=weight,
                 ha="right", va="baseline", zorder=z)
    b = fig.text(x_dec_pt, y_pt, "." + tail, fontsize=size, color=color,
                 fontweight=weight, ha="left", va="baseline", zorder=z)
    a.set_transform(tr)
    b.set_transform(tr)
    return a, b


# --------------------------------------------------------------------------
# 8.  Axes chrome
# --------------------------------------------------------------------------

def row_axes(fig, x_pt, top_pt, w_pt, row_h_pt, n, gap_pt=3.0, spine="#9AA5B1"):
    """`n` stacked miniature Axes in point-space, top-down, sharing one x scale.

    Only the last row keeps a bottom spine and tick labels, which is what makes a
    stack of per-row mini-plots read as one table rather than n charts.  Each Axes is
    created with `facecolor="none"` so a figure-level ground stays visible behind it.
    """
    w_fig, h_fig = fig.get_size_inches() * 72.0
    axes = []
    for i in range(n):
        y = top_pt - (i + 1) * row_h_pt - i * gap_pt
        ax = fig.add_axes([x_pt / w_fig, y / h_fig, w_pt / w_fig, row_h_pt / h_fig])
        ax.set_facecolor("none")
        ax.tick_params(length=2.0, width=0.6, pad=1.5, labelsize=7.0)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_linewidth(0.6)
        ax.spines["bottom"].set_color(spine)
        ax.set_yticks([])
        if i < n - 1:
            ax.spines["bottom"].set_visible(False)
            ax.set_xticks([])
        axes.append(ax)
    return axes


# --------------------------------------------------------------------------
# 9.  Colour
# --------------------------------------------------------------------------

#: A five-role semantic palette.  Roles, not names: pick by what the colour MEANS.
SEMANTIC = {
    "primary": "#2E5AA8",    # the main series
    "contrast": "#C1553C",   # the series it is being compared against
    "support": "#3C8A55",    # agreement / positive control
    "caution": "#C08A2E",    # partial, ambiguous, underpowered
    "neutral": "#6B7280",    # baseline, "everything else"
}


def _hls(c):
    r, g, b = to_rgb(c)
    return colorsys.rgb_to_hls(r, g, b)


def tint(c, amount=0.84):
    """Lighten toward white in HLS.  `L -> L + amount*(1-L)`, saturation damped by
    `0.25*amount` so the tint does not go chalky.  0.84 is the grouping-ground value:
    it lands every hue near 11-13:1 against #1F2937 ink."""
    h, l, s = _hls(c)
    return to_hex(colorsys.hls_to_rgb(h, min(l + amount * (1.0 - l), 0.985),
                                      max(s * (1.0 - 0.25 * amount), 0.0)))


def shade(c, amount=0.34):
    """Darken toward black in HLS.  `L -> L*(1-amount)`, saturation nudged up so the
    dark step keeps its hue.  0.34 gives >= 6:1 against white for every hue in
    `SEMANTIC`, so the dark step is safe for a label or an outline."""
    h, l, s = _hls(c)
    return to_hex(colorsys.hls_to_rgb(h, max(l * (1.0 - amount), 0.03),
                                      min(s * (1.0 + 0.12 * amount), 1.0)))


def ramp(c, tint_amt=0.84, shade_amt=0.34):
    """`(tint, mid, dark)` for one hue -- ground, series, label/outline."""
    return tint(c, tint_amt), to_hex(to_rgb(c)), shade(c, shade_amt)


def relative_luminance(c):
    """WCAG 2.1 relative luminance of a colour."""
    def lin(u):
        return u / 12.92 if u <= 0.04045 else ((u + 0.055) / 1.055) ** 2.4
    r, g, b = (lin(v) for v in to_rgb(c))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(c1, c2):
    """WCAG contrast ratio in [1, 21].  Targets for figure work: >= 4.5 for a label on
    a fill, >= 3.0 for a 1 pt or thicker graphical mark, >= 10 for small ink on a
    pale tint."""
    l1, l2 = relative_luminance(c1), relative_luminance(c2)
    hi, lo = max(l1, l2), min(l1, l2)
    return (hi + 0.05) / (lo + 0.05)


def luma(c):
    """sRGB luma (0.299R + 0.587G + 0.114B) -- what a B/W printer sees."""
    r, g, b = to_rgb(c)
    return 0.299 * r + 0.587 * g + 0.114 * b


def grayscale(c):
    """The colour's luma as a gray hex, for a side-by-side B/W check."""
    y = luma(c)
    return to_hex((y, y, y))


def gray_gap(colors):
    """Minimum pairwise luma gap across a set of colours.  Aim for >= 0.10; below
    that the hues merge into one another in greyscale print."""
    ys = sorted(luma(c) for c in colors)
    return min(b - a for a, b in zip(ys, ys[1:])) if len(ys) > 1 else 1.0


def set_luma(c, target, iters=40):
    """Bisect a colour's HLS lightness until its luma equals `target`.  Hue and
    saturation are untouched, so the colour keeps its identity."""
    h, _, s = _hls(c)
    lo, hi = 0.0, 1.0
    for _ in range(iters):
        mid = (lo + hi) / 2
        rgb = colorsys.hls_to_rgb(h, mid, s)
        y = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        if y < target:
            lo = mid
        else:
            hi = mid
    return to_hex(colorsys.hls_to_rgb(h, (lo + hi) / 2, s))


def spread_luma(colors, lo=0.30, hi=0.68):
    """Re-space a set of hues evenly in luma so they stay separable in B/W.

    A palette chosen purely on hue is very likely to collapse in greyscale: the five
    `SEMANTIC` mids as designed have a minimum pairwise luma gap of **0.004**, i.e.
    they are effectively the same grey.  `spread_luma` raises that to 0.093 while
    keeping every hue and saturation, at the cost of shifting the lighter members
    noticeably lighter -- so re-check `contrast_ratio` against white afterwards.
    """
    order = np.argsort([luma(c) for c in colors])
    targets = np.linspace(lo, hi, len(colors))
    out = list(colors)
    for rank, idx in enumerate(order):
        out[idx] = set_luma(colors[idx], float(targets[rank]))
    return out


def palette_report(palette=None, ink="#1F2937"):
    """A printable audit of a palette: the three-step ramp, the contrast ratios that
    matter, and the greyscale gap.  Returns a list of dicts; prints nothing."""
    palette = SEMANTIC if palette is None else palette
    rows = []
    for name, base_c in palette.items():
        lt, mid, dk = ramp(base_c)
        rows.append(dict(
            role=name, tint=lt, mid=mid, dark=dk,
            ink_on_tint=round(contrast_ratio(lt, ink), 2),
            white_on_mid=round(contrast_ratio(mid, "#FFFFFF"), 2),
            white_on_dark=round(contrast_ratio(dk, "#FFFFFF"), 2),
            luma_mid=round(luma(mid), 3),
        ))
    return rows
