"""Numerical figure renderer from the archived paper build."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REDESIGN = HERE
ROOT = HERE
sys.path.insert(0, str(REDESIGN / "research2"))
sys.path.insert(0, str(REDESIGN / "qa"))

import craft                                    # noqa: E402  (tested helpers)

import matplotlib                               # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt                 # noqa: E402
from matplotlib.legend_handler import HandlerBase   # noqa: E402
from matplotlib.lines import Line2D             # noqa: E402
from matplotlib.patches import Rectangle        # noqa: E402

# ---------------------------------------------------------------------------
# 1.  Frozen inputs -- SHA-256 asserted at load (copied from build_fig3_v3.py)
# ---------------------------------------------------------------------------

SOURCE = "inputs/wide_d"
PRIMARY_PATH = f"{SOURCE}/source_reports/reporting/wide_d_nine_model_v1.json"
SECONDARY_PATH = f"{SOURCE}/contrasts/secondary_contrasts.json"
# The T contrast.  Already SHA-pinned in research/figure1_values.json; this is the
# same file and the same digest, asserted here so the strip cannot drift.
T_CONTRAST_PATH = ("inputs/abc/c/"
                   "cross_checkpoint_contrasts_v1.json")
ORIGINAL_ARCHIVE_SHA256 = {
    PRIMARY_PATH: "6b1ccd80ecd58d01025ee41f3b2c7be274875a831509018244045019fa98e14b",
    SECONDARY_PATH: "9b0924d092c1e6ea2577b8e305d8eebf44016f269b693c8b620f9ae48ea9250b",
    T_CONTRAST_PATH: "e66d80a32b44c8245c88180f7fafc1b4656b7be5ca9ce7ca67e40b18c0735a56",
}  # Historical identities.
SOURCE_SHA256 = {
    PRIMARY_PATH: "8d09caf9edcd496cbc05971d26e1bedea0635a1aca164871c933807155479568",
    SECONDARY_PATH: "1bce16d6011bb19eaf5aadb444e4f8b3638dbef23f648915072dd58605ca9539",
    T_CONTRAST_PATH: "c3c43955882664630a6c36321e14b99eb13f6e41add958b4f980e3df119f35cd",
}  # Current public input bytes.

T_CONTRAST_SELECTOR = dict(estimand="cross_checkpoint_T_difference",
                           metric="mean_logprob", arm="all_router",
                           window="full_step", control="paired",
                           left="gpt_oss_120b", right="gpt_oss_20b")
CONTRACT = HERE / "inputs" / "figure3_values.json"

MODELS = [
    ("qwen36_moe", "Qwen3.6"),
    ("qwen3_moe", "Qwen3"),
    ("glm4_moe_lite", "GLM-4.7-Flash"),
    ("nemotron35_lightning", "Lightning"),
    ("nemotron3_nano", "Nano"),
    ("gemma4_moe", "Gemma4"),
    ("olmoe", "OLMoE"),
    ("gpt_oss_20b", "GPT-OSS-20B"),
    ("gpt_oss_120b", "GPT-OSS-120B"),
]
ROLES = ["D_clean", "D_error", "D_common"]
ROLE_TITLES = ["Clean receiver", "Error receiver", "Receiver average"]
ROLE_MARKERS = ["o", "s", "D"]
CONTRACT_Y_OFFSETS = [-0.24, 0.0, 0.24]        # frozen record, asserted
ROLE_Y_OFFSETS = [-0.30, 0.0, 0.30]            # display (change 3)
ENDPOINT = "primary_mean_logprob"
FIELDS = ["mean", "ci_low", "ci_high", "bootstrap_standard_error"]

XLIM = (-0.25, 0.75)
XTICKS = [-0.2, 0, 0.2, 0.4, 0.6]
XTICKLABELS = ["−0.2", "0", "0.2", "0.4", "0.6"]
YLIM = (8.55, -0.55)

# ---------------------------------------------------------------------------
# 2.  Palette and tokens -- PALETTE_SPEC.md §3, §6.2.  ONE HUE.
# ---------------------------------------------------------------------------
# Hue names the patched interface and nothing else.  Every series in this figure is
# measured at ONE interface (ENDPOINT below is full-step all-router), so this figure
# has no interface to vary: it is a one-hue figure.  The donor role is carried by a
# TONE step inside the teal family plus the marker shapes the figure already had.
# Crimson means the post-MoE residual interface, which this figure does not contain,
# so crimson leaves Figure 3 entirely.

ALLROUTER = "#087F8C"      # L* 48.41 -- THE ROUTER-LOGIT INTERFACE, all-layer arm.
                           # Here: the clean receiver, the lighter step of the pair.
ALLROUTER_DEEP = "#05535C"  # L* 31.79, same hue to 0.3 deg -- the SAME interface, its
                           # darker step.  Here: the error receiver.  Dark-is-more:
                           # the error condition is the one with something in it.
INK = "#11181F"            # L*  7.9 -- glyphs, and the receiver-average point
RULE = "#6B7580"           # L* 48.8 -- frames, ticks, rail headers
HAIR = "#98A2AB"           # L* 66.1 -- the zero rule, and nothing else
GRID = "#D3D9DE"           # L* 86.4 -- gridlines on white rows
GRID_ON_BAND = "#C4C9CF"   # L* 80.7 -- gridlines on BAND rows (equal perceived wt)
BAND = "#E8EFF7"           # L* 94.14 -- per-model row grounds, re-tinted from
                           # #EDEFF2 at the same lightness (dRGB 5 -> 15), so the
                           # band reads as tinted paper rather than as a grey.

# Plan A had to ADD one neutral to VISUAL_LANGUAGE §2.2's ramp (#AEB7BF) to fill the
# average's slab.  The fix deletes the slab, so that added neutral is deleted with it
# and the figure is back inside the published ramp exactly: the receiver average is
# INK, the same near-black as its own diamond, and every other colour is a token.
PALETTE = {ALLROUTER, ALLROUTER_DEEP, INK, RULE, HAIR,
           GRID, GRID_ON_BAND, BAND, "#FFFFFF", "white", "none"}

ROLE_COLOR = {"D_clean": ALLROUTER, "D_error": ALLROUTER_DEEP, "D_common": INK}
# PALETTE_SPEC §3.4: no coloured legend labels.  The four dark steps the old scheme
# keyed the legend with span 0.16 L* and are dE00 2.06 apart for a protanope -- in
# greyscale and for the commonest deficiency that key is one colour, i.e. not a key.
# The marker miniature carries the colour; the label is INK.
ROLE_LABEL_COLOR = {"D_clean": INK, "D_error": INK, "D_common": INK}

LW_GRID = 0.50
LW_ZERO = 0.64
LW_FRAME = 0.80
LW_DATA = 1.00
STROKE_LADDER = {0.50, 0.60, 0.64, 0.80, 0.90, 1.00}

DASH_GRID = (2.4, 2.0)
DASH_ZERO = (2.55, 1.91)

SZ = 8.0                   # one tier: ticks, names, rail, headers, axis title
SIZE_LADDER = {8.0, 8.5, 9.0}

R_CHIP = 2.5
TICK_PAD = 4.5

# Marks.  The two receiver roles keep the round-3 equal-ink sizes and the white
# moat (they cross the zero rule and the gridlines).  The average's diamond carries
# no moat: it sits on its own INK rule, and a white ring would both cut that rule in
# two and eat the clearance to the end caps on the shortest interval (Nano, 4.85 pt
# against a 3.8 pt diamond) -- see gate_occlusion.  The diamond is 3.8 pt, a hair
# larger than the circle (3.77) and the square (3.4), because it is the summary of
# the two of them; the shape, not the size, is what names it.
MS_BY_ROLE = {"D_clean": 3.77, "D_error": 3.4, "D_common": 3.8}
MEW_ROLE = 0.6
MEW_AVG = 0.0
CAPSIZE = 1.5              # role caps: 3.0 pt tall
CAPTHICK = 1.0
AVG_CAP_H_PT = 5.6         # the receiver-average end caps: 5.6 pt tall, so the
                           # average's endpoints are the tallest data marks in the
                           # panel and its interval reads at least as loudly as its
                           # mean -- without any fill, and therefore without a bar.

# ---------------------------------------------------------------------------
# 3.  Canvas geometry -- points from the lower-left corner
# ---------------------------------------------------------------------------

FIG_W_PT, FIG_H_PT = 396.0, 250.0              # 5.5 x 3.4722 in; 222 + the 28 pt strip
NAME_RIGHT_PT = 60.0                           # 'GPT-OSS-120B' is 53.59 pt at 8 pt bold
AX_X0, AX_X1 = 66.0, 300.0                     # 234 pt for the 1.0-unit span (was 240)
AX_Y0, AX_Y1 = 64.0, 238.0                     # panel height 174 pt, unchanged
RAIL_DEC_X = 316.0                             # mean, decimal-aligned      (was 322)
RAIL_CI_X = 334.0                              # '[lo, hi]', left-aligned   (was 340)
RAIL_MEAN_HEADER_X = 330.0                     # right edge of the mean column
RAIL_HEADER_BASELINE = 241.5
AXIS_TITLE_X = 183.0                           # the axes rect's centre
AXIS_TITLE_BASELINE = 42.0
BAND_X0, BAND_X1 = AX_X0, AX_X1                # row bands are clipped to the ruler
BLOCK_X0, BLOCK_X1 = 6.0, 390.0

# --- the matched-pair contrast strip -----------------------------------------
# A stroke-free BAND ground the width of the whole block, so it cannot be read as a
# tenth row: every row band stops at the ruler's edges, this one does not, and it
# sits below the axis title with no axis, tick or zero rule anywhere near it.
STRIP_X0, STRIP_X1 = BLOCK_X0, BLOCK_X1
STRIP_Y0, STRIP_Y1 = 8.0, 34.0
STRIP_LABEL_X = 12.0                           # 6 pt of inset, VISUAL_LANGUAGE 3.4
STRIP_BASELINES = {"T": 24.8, "D": 13.8}       # 11.0 pt pitch, block centred
                                               # in the 26 pt ground (3.1 pt each side)
STRIP_ROWS = [
    dict(key="T",
         label=r"$T$, GPT-OSS-120B $-$ 20B, full step all-router"),
    dict(key="D",
         label=r"$D_{\rm common}$, GPT-OSS-120B $-$ 20B, post hoc secondary"),
]

OUT_DIR = HERE
OUT_PDF = OUT_DIR / "fig3.pdf"
OUT_PNG = OUT_DIR / "fig3.png"
OUT_INVENTORY = OUT_DIR / "fig3_inventory.json"
OUT_GEOMETRY = OUT_DIR / "fig3_geometry.json"
OUT_CAPTION = OUT_DIR / "captions" / "fig3.tex"


# ---------------------------------------------------------------------------
# 4.  Data loading -- verbatim from build_fig3_v3.py
# ---------------------------------------------------------------------------

def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def estimate(source: dict) -> dict:
    result = {key: source[key] for key in FIELDS}
    assert all(math.isfinite(v) for v in result.values())
    assert result["ci_low"] <= result["mean"] <= result["ci_high"]
    assert result["bootstrap_standard_error"] > 0
    return result


def load_rows() -> tuple[list[dict], dict]:
    for path, expected in SOURCE_SHA256.items():
        actual = sha((ROOT / path).read_bytes())
        assert actual == expected, f"Changed accepted input: {path} ({actual})"

    primary = json.loads((ROOT / PRIMARY_PATH).read_text())
    secondary = json.loads((ROOT / SECONDARY_PATH).read_text())
    assert primary["status"] == "COMPLETE" and not primary["missing_models"]
    models = {row["model"]: row for row in primary["models"]}
    assert set(models) == {key for key, _ in MODELS}
    assert set(secondary["role_contrasts"]) == set(models)
    bootstrap = secondary["bootstrap"]
    assert bootstrap["n_pairs"] == 290 and bootstrap["n_clusters"] == 235
    assert bootstrap["n_resamples"] == 50000 and bootstrap["seed"] == 20260908
    assert bootstrap["donor_assignment_fixed"]
    assert secondary["analysis_status"] == "posthoc_secondary"

    rows = []
    for alias, label in MODELS:
        row = models[alias]
        assert (row["N"], row["K"], row["unique_fixed_donor_pairs"]) == (290, 235, 204)
        cells = {}
        for role in ROLES:
            values = estimate(row[ENDPOINT][role])
            assert values == estimate(
                secondary["reconstructed_original_endpoints"][alias][ENDPOINT][role]
            )
            cells[role] = values
        rows.append(dict(alias=alias, label=label, N=row["N"], K=row["K"],
                         cells=cells,
                         contrast=estimate(secondary["role_contrasts"][alias])))
    return rows, secondary


def load_contrasts(secondary: dict) -> dict:
    """The two matched-pair direct contrasts printed in the strip.

    Returns both forms for the D contrast.  `stored` is what the frozen file holds
    (20B - 120B); `oriented` is what the strip prints (120B - 20B) -- the estimate
    negated and the interval endpoints swapped, which is exact for a percentile
    interval and is an ORIENTATION, not a new estimate.  Nothing here recomputes a
    bootstrap; the negation is asserted to be exact in both directions.
    """
    records = json.loads((ROOT / T_CONTRAST_PATH).read_text())
    hits = [r for r in records
            if all(r.get(k) == v for k, v in T_CONTRAST_SELECTOR.items())]
    assert len(hits) == 1, f"T contrast selector matched {len(hits)} rows"
    t = hits[0]
    assert (t["N"], t["K"]) == (290, 235)
    assert t["bootstrap_draws"] == 50000 and t["seed"] == 20260908
    assert t["interval_type"] == "matched_receiver_component"
    assert t["ci_low"] <= t["mean"] <= t["ci_high"]
    assert t["ci_low"] > 0.0, "the T contrast is stated as resolved and positive"

    d = secondary["gpt20_minus_gpt120_D_common"]
    assert d["ci_low"] <= d["mean"] <= d["ci_high"]
    assert d["ci_high"] < 0.0 or d["ci_low"] > 0.0
    boot = secondary["bootstrap"]
    # What makes the two rows a legitimate side-by-side pair: identical resampling
    # size, draw count and seed.  Only the resampling unit and the protocol differ.
    assert (boot["n_pairs"], boot["n_clusters"]) == (t["N"], t["K"]) == (290, 235)
    assert boot["n_resamples"] == t["bootstrap_draws"] == 50000
    assert boot["seed"] == t["seed"] == 20260908
    assert boot["unit"] == "receiver_source_component"

    oriented = dict(mean=-d["mean"], ci_low=-d["ci_high"], ci_high=-d["ci_low"])
    assert oriented["ci_low"] <= oriented["mean"] <= oriented["ci_high"]
    assert -oriented["mean"] == d["mean"] and -oriented["ci_low"] == d["ci_high"]
    assert -oriented["ci_high"] == d["ci_low"]
    # The whole reason the strip exists: one orientation, opposite signs.
    assert t["mean"] > 0.0 > oriented["mean"]
    assert t["ci_low"] > 0.0 and oriented["ci_high"] < 0.0

    return dict(
        T=dict(stored=dict(mean=t["mean"], ci_low=t["ci_low"],
                           ci_high=t["ci_high"]),
               oriented=dict(mean=t["mean"], ci_low=t["ci_low"],
                             ci_high=t["ci_high"]),
               stored_direction="gpt_oss_120b - gpt_oss_20b",
               printed_direction="gpt_oss_120b - gpt_oss_20b",
               negated=False, N=t["N"], K=t["K"],
               source_path=T_CONTRAST_PATH,
               source_locator=("[] where estimand==cross_checkpoint_T_difference "
                               "& metric==mean_logprob & arm==all_router "
                               "& window==full_step & control==paired "
                               "(left=gpt_oss_120b, right=gpt_oss_20b)"),
               interval_type=t["interval_type"],
               analysis_status="primary_direct_contrast"),
        D=dict(stored=dict(mean=d["mean"], ci_low=d["ci_low"],
                           ci_high=d["ci_high"]),
               oriented=oriented,
               stored_direction="gpt_oss_20b - gpt_oss_120b",
               printed_direction="gpt_oss_120b - gpt_oss_20b",
               negated=True, N=boot["n_pairs"], K=boot["n_clusters"],
               source_path=SECONDARY_PATH,
               source_locator="gpt20_minus_gpt120_D_common",
               interval_type=boot["unit"],
               analysis_status=secondary["analysis_status"],
               bootstrap_standard_error=d["bootstrap_standard_error"]),
    )


def assert_frozen_structure(rows: list[dict]) -> None:
    contract = json.loads(CONTRACT.read_text())
    assert contract["endpoint"] == ENDPOINT
    assert tuple(contract["xlim"]) == XLIM
    assert contract["xticks"] == XTICKS
    assert tuple(contract["ylim"]) == YLIM
    assert contract["role_order"] == ROLES
    assert contract["role_titles"] == ROLE_TITLES
    assert contract["role_markers"] == ROLE_MARKERS
    # The frozen record is asserted; the DISPLAY offsets are wider (change 3) and
    # are recorded in the geometry JSON, never silently substituted.
    assert contract["role_y_offsets"] == CONTRACT_Y_OFFSETS
    assert [r["alias"] for r in contract["rows"]] == [a for a, _ in MODELS]
    assert [r["label"] for r in contract["rows"]] == [b for _, b in MODELS]
    for got, want in zip(rows, contract["rows"]):
        assert got["alias"] == want["alias"] and got["label"] == want["label"]
        assert (got["N"], got["K"]) == (want["N"], want["K"]) == (290, 235)
        for role in ROLES:
            for field in FIELDS:
                assert got["cells"][role][field] == want[role][field]

    # Sign partitions, read from the frozen floats and never from a formatted string.
    excludes_zero = [(r["label"], role)
                     for r in rows for role in ROLES
                     if r["cells"][role]["ci_low"] > 0 or r["cells"][role]["ci_high"] < 0]
    assert excludes_zero == [
        ("GLM-4.7-Flash", "D_error"), ("Gemma4", "D_clean"),
        ("GPT-OSS-20B", "D_clean"), ("GPT-OSS-20B", "D_common"),
    ], excludes_zero
    assert len(excludes_zero) == 4 and 27 - len(excludes_zero) == 23
    lightning = rows[3]
    assert lightning["label"] == "Lightning"
    assert lightning["cells"]["D_clean"]["ci_low"] == -0.0004819306006960866 < 0
    glm = rows[2]
    assert glm["cells"]["D_error"]["ci_high"] == -0.0011054407199225964 < 0


def fmt(value: float) -> str:
    return f"{value:.3f}".replace("-", "−")


def fmt_signed(value: float) -> str:
    """The strip prints an explicit '+' on the positive contrast.  The sign crossing
    IS the reading, so it must not be carried by the absence of a glyph; and Times
    sets '+' and U+2212 at the same advance, so the decimal column still aligns."""
    return ("+" if value > 0 else "") + fmt(value)


# ---------------------------------------------------------------------------
# 5.  The figure
# ---------------------------------------------------------------------------

def row_centre_pt(i: int) -> float:
    span = YLIM[0] - YLIM[1]
    return AX_Y0 + (AX_Y1 - AX_Y0) * (YLIM[0] - i) / span


def band_pitch_pt() -> float:
    return (AX_Y1 - AX_Y0) / (YLIM[0] - YLIM[1])


def pt_to_data_y(y_pt: float) -> float:
    span = YLIM[0] - YLIM[1]
    return YLIM[0] - (y_pt - AX_Y0) * span / (AX_Y1 - AX_Y0)


def grid_segments() -> list[tuple[float, float, bool]]:
    """The axes interior split into (y0_pt, y1_pt, on_band) runs, bottom-up."""
    pitch = band_pitch_pt()
    runs: list[tuple[float, float, bool]] = []
    edges: list[tuple[float, float, bool]] = []
    for i in range(9):
        y0 = row_centre_pt(i) - pitch / 2.0
        edges.append((y0, y0 + pitch, i % 2 == 0))
    edges.sort()
    cursor = AX_Y0
    for y0, y1, on_band in edges:
        if y0 > cursor + 1e-9:
            runs.append((cursor, y0, False))
        runs.append((max(y0, cursor), y1, on_band))
        cursor = y1
    if cursor < AX_Y1 - 1e-9:
        runs.append((cursor, AX_Y1, False))
    assert abs(runs[0][0] - AX_Y0) < 1e-9 and abs(runs[-1][1] - AX_Y1) < 1e-9
    return runs


class HandlerCappedPoint(HandlerBase):
    """The receiver-average legend glyph: a true miniature of the drawn object -- an
    INK rule with a full-height cap at each end and the real INK diamond on it."""

    def __init__(self, colour, half_len_pt=6.5, cap_h_pt=AVG_CAP_H_PT,
                 lw=LW_DATA, marker="D", ms=3.8, **kw):
        self.colour = colour
        self.half_len_pt = half_len_pt
        self.cap_h_pt = cap_h_pt
        self.lw = lw
        self.marker = marker
        self.ms = ms
        self.dot = None
        super().__init__(**kw)

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height,
                       fontsize, trans):
        cx = xdescent + width / 2.0
        cy = ydescent + height / 2.0
        x0, x1 = cx - self.half_len_pt, cx + self.half_len_pt
        rule = Line2D([x0, x1], [cy, cy], color=self.colour, linewidth=self.lw,
                      solid_capstyle="butt")
        caps = [Line2D([x, x], [cy - self.cap_h_pt / 2.0, cy + self.cap_h_pt / 2.0],
                       color=self.colour, linewidth=self.lw, solid_capstyle="butt")
                for x in (x0, x1)]
        dot = Line2D([cx], [cy], marker=self.marker, ms=self.ms,
                     mfc=self.colour, mec=self.colour, mew=MEW_AVG,
                     color=self.colour, ls="none")
        self.dot = dot
        artists = [rule, *caps, dot]
        for a in artists:
            a.set_transform(trans)
        return artists


class MiniatureHandler(craft.HandlerMarkerInterval):
    """`craft.HandlerMarkerInterval` hardcodes `mec=self.color`; the data marks carry
    the white moat, so the miniature must too."""

    def __init__(self, *args, mec="white", **kwargs):
        super().__init__(*args, **kwargs)
        self.mec = mec
        self.dot = None

    def create_artists(self, *args, **kwargs):
        artists = super().create_artists(*args, **kwargs)
        dot = artists[-1]
        dot.set_color(self.color)
        dot.set_markeredgecolor(self.mec)
        dot.set_markerfacecolor(self.mfc)
        dot.set_markeredgewidth(self.mew)
        self.dot = dot
        return artists


def build_figure(rows: list[dict], contrasts: dict):
    craft.apply_paper_rc(times_math=True, extra={
        "axes.unicode_minus": True,
        "text.color": INK,
        "savefig.facecolor": "white",
        "savefig.pad_inches": 0.0,
        "figure.dpi": 220,
        "savefig.dpi": 220,
    })

    fig = plt.figure(figsize=(FIG_W_PT / 72.0, FIG_H_PT / 72.0))
    assert fig.get_size_inches()[0] == 5.5
    ax = fig.add_axes([AX_X0 / FIG_W_PT, AX_Y0 / FIG_H_PT,
                       (AX_X1 - AX_X0) / FIG_W_PT, (AX_Y1 - AX_Y0) / FIG_H_PT])
    ax.set_facecolor("none")
    ax.set_zorder(2.0)
    ax.patch.set_visible(False)
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_xticks(XTICKS, XTICKLABELS)
    ax.set_yticks([])
    ax.set_autoscale_on(False)
    assert [t.get_text() for t in ax.get_xticklabels()] == XTICKLABELS
    assert list(ax.get_xticks()) == XTICKS
    assert ax.get_xlim() == XLIM and ax.get_ylim() == YLIM

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(RULE)
        spine.set_linewidth(LW_FRAME)
    ax.tick_params(axis="x", direction="in", length=2.0, width=0.6, color=RULE,
                   labelsize=SZ, pad=TICK_PAD, labelcolor=INK, top=False)
    ax.tick_params(axis="y", length=0, labelleft=False)

    pitch = band_pitch_pt()
    for i in range(0, 9, 2):
        y0 = row_centre_pt(i) - pitch / 2.0
        fig.add_artist(Rectangle(
            (BAND_X0, y0), BAND_X1 - BAND_X0, pitch,
            transform=craft.pt_transform(fig), facecolor=BAND,
            edgecolor="none", linewidth=0, zorder=0.2))
    assert ax.patch.get_facecolor()[3] == 0

    # Two-tone gridlines, so the ruler reads at one perceived weight everywhere.
    ax.xaxis.grid(False)
    period = DASH_GRID[0] + DASH_GRID[1]
    runs = grid_segments()
    gridlines = []
    for tick in XTICKS:
        for y0, y1, on_band in runs:
            line = Line2D([tick, tick], [pt_to_data_y(y0), pt_to_data_y(y1)],
                          color=GRID_ON_BAND if on_band else GRID,
                          linewidth=LW_GRID, solid_capstyle="butt", zorder=0.5)
            line.set_linestyle(((y0 - AX_Y0) % period, DASH_GRID))
            ax.add_line(line)
            gridlines.append(line)
    assert len(gridlines) == len(XTICKS) * len(runs)

    zero = ax.axvline(0.0, color=HAIR, linewidth=LW_ZERO, zorder=1.5)
    zero.set_dashes(DASH_ZERO)

    # --- the 27 estimates -------------------------------------------------
    # Two grammars, one per information tier.  Both are applied uniformly: every
    # clean/error interval is drawn by one rule and every average by another, and
    # nothing about an interval's relation to zero alters either.
    cap_h_rows = AVG_CAP_H_PT / pitch
    inventory: list[dict] = []
    containers = []
    avg_rules = []
    avg_caps = []
    avg_points = []
    for i, row in enumerate(rows):
        for role, dy, marker in zip(ROLES, ROLE_Y_OFFSETS, ROLE_MARKERS):
            cell = row["cells"][role]
            colour = ROLE_COLOR[role]
            y = i + dy
            if role == "D_common":
                rule = Line2D([cell["ci_low"], cell["ci_high"]], [y, y],
                              color=colour, linewidth=LW_DATA,
                              solid_capstyle="butt", zorder=2.6)
                ax.add_line(rule)
                avg_rules.append(rule)
                for end in (cell["ci_low"], cell["ci_high"]):
                    cap = Line2D([end, end],
                                 [y - cap_h_rows / 2.0, y + cap_h_rows / 2.0],
                                 color=colour, linewidth=LW_DATA,
                                 solid_capstyle="butt", zorder=2.8)
                    ax.add_line(cap)
                    avg_caps.append(cap)
                point = Line2D([cell["mean"]], [y], marker=marker,
                               ms=MS_BY_ROLE[role], mfc=colour, mec=colour,
                               mew=MEW_AVG, color=colour, ls="none", zorder=3.2)
                ax.add_line(point)
                avg_points.append(point)
            else:
                cont = ax.errorbar(
                    [cell["mean"]], [y],
                    xerr=[[cell["mean"] - cell["ci_low"]],
                          [cell["ci_high"] - cell["mean"]]],
                    fmt=marker, ms=MS_BY_ROLE[role], mfc=colour, mec="white",
                    mew=MEW_ROLE, color=colour, ecolor=colour, elinewidth=LW_DATA,
                    capsize=CAPSIZE, capthick=CAPTHICK, zorder=3.0,
                )
                containers.append(cont)
                for cap in cont[1]:
                    cap.set_markeredgewidth(CAPTHICK)
                    cap.set_color(colour)
            inventory.append(dict(
                panel="roles", model=row["label"], alias=row["alias"],
                series=role, role_title=ROLE_TITLES[ROLES.index(role)],
                row=i, y=y, marker=marker, color=colour,
                form=("capped-rule+diamond" if role == "D_common"
                      else "interval+marker"),
                marker_size_pt=MS_BY_ROLE[role],
                mean=cell["mean"], ci_low=cell["ci_low"], ci_high=cell["ci_high"],
                bootstrap_standard_error=cell["bootstrap_standard_error"],
                N=row["N"], K=row["K"], endpoint=ENDPOINT,
                source_path=PRIMARY_PATH,
                source_locator=f"models[model={row['alias']}].{ENDPOINT}.{role}",
                plotted=True,
                shown_in_numeric_rail=(role == "D_common"),
            ))
    assert len(containers) == 18 and len(avg_rules) == 9 and len(avg_caps) == 18
    assert len(avg_points) == 9 and len(inventory) == 27

    # Contract rows this layout does not plot.
    for i, row in enumerate(rows):
        c = row["contrast"]
        inventory.append(dict(
            panel="not-plotted", model=row["label"], alias=row["alias"],
            series="D_clean - D_error", row=i,
            mean=c["mean"], ci_low=c["ci_low"], ci_high=c["ci_high"],
            bootstrap_standard_error=c["bootstrap_standard_error"],
            N=row["N"], K=row["K"], endpoint=ENDPOINT,
            source_path=SECONDARY_PATH,
            source_locator=f"role_contrasts.{row['alias']}",
            plotted=False,
            reason="Figure 3 plots the three receiver roles, not their difference; "
                   "the contrast is reported in the wide-D table.",
        ))

    # --- legend, inside the axes, upper right ------------------------------
    proxies = [Line2D([], [], linestyle="none") for _ in ROLES]
    handlers = {}
    for p, role, marker in zip(proxies, ROLES, ROLE_MARKERS):
        if role == "D_common":
            handlers[p] = HandlerCappedPoint(ROLE_COLOR[role], half_len_pt=6.5,
                                             marker=marker, ms=MS_BY_ROLE[role])
        else:
            handlers[p] = MiniatureHandler(ROLE_COLOR[role], half_len_pt=6.5,
                                           cap_pt=3.0, lw=LW_DATA, marker=marker,
                                           ms=MS_BY_ROLE[role],
                                           mfc=ROLE_COLOR[role], mew=MEW_ROLE,
                                           mec="white")
    leg = ax.legend(proxies, ROLE_TITLES, loc="upper right", handler_map=handlers,
                    fontsize=SZ, borderpad=0.42, labelspacing=0.34,
                    handlelength=13.0 / SZ, handletextpad=0.5,
                    borderaxespad=0.35 + 4.0 / SZ,
                    frameon=True, fancybox=False, framealpha=1.0)
    craft.style_legend(leg, r_pt=R_CHIP, ec=RULE, lw=LW_FRAME, fc="#FFFFFF",
                       alpha=1.0)
    leg.set_zorder(6.0)
    leg.get_frame().set_zorder(6.0)
    for text, role in zip(leg.get_texts(), ROLES):
        text.set_color(ROLE_LABEL_COLOR[role])
        text.set_fontweight("normal")
        text.set_fontsize(SZ)

    # --- model-name column --------------------------------------------------
    for i, row in enumerate(rows):
        craft.put_pt(fig, NAME_RIGHT_PT, row_centre_pt(i), row["label"],
                     size=SZ, color=INK, weight="bold", ha="right",
                     va="center_baseline", z=5)

    # --- numeric rail: the mean AND the interval ---------------------------
    headers = [
        craft.put_pt(fig, RAIL_MEAN_HEADER_X, RAIL_HEADER_BASELINE,
                     r"$D_{\rm common}$", size=SZ, color=RULE, ha="right",
                     va="baseline", z=5),
        craft.put_pt(fig, RAIL_CI_X, RAIL_HEADER_BASELINE, "95% CI", size=SZ,
                     color=RULE, ha="left", va="baseline", z=5),
    ]
    rail_parts = []
    for i, row in enumerate(rows):
        cell = row["cells"]["D_common"]
        for part in craft.num_cell(fig, RAIL_DEC_X, row_centre_pt(i),
                                   fmt(cell["mean"]), size=SZ, color=INK, z=5):
            part.set_va("center_baseline")
            rail_parts.append(part)
        rail_parts.append(craft.put_pt(
            fig, RAIL_CI_X, row_centre_pt(i),
            f"[{fmt(cell['ci_low'])}, {fmt(cell['ci_high'])}]",
            size=SZ, color=RULE, ha="left", va="center_baseline", z=5))

    craft.put_pt(fig, AXIS_TITLE_X, AXIS_TITLE_BASELINE,
                 r"$D$ (mean-logprob margin)", size=SZ, color=INK,
                 ha="center", va="baseline", z=5)

    # --- the matched-pair contrast strip -----------------------------------
    # Ground first.  Stroke-free, no shadow, the published BAND token -- the same
    # ground the row groups use, but the full block width instead of the ruler's,
    # which is the signal that it is not a row.  Nothing is drawn on it but text:
    # no axis, no tick, no zero rule, no mark.  T and D therefore share no ruler.
    fig.add_artist(Rectangle(
        (STRIP_X0, STRIP_Y0), STRIP_X1 - STRIP_X0, STRIP_Y1 - STRIP_Y0,
        transform=craft.pt_transform(fig), facecolor=BAND,
        edgecolor="none", linewidth=0, zorder=0.2))

    strip_parts = []
    for spec in STRIP_ROWS:
        row = contrasts[spec["key"]]
        shown = row["oriented"]
        baseline = STRIP_BASELINES[spec["key"]]
        strip_parts.append(craft.put_pt(
            fig, STRIP_LABEL_X, baseline, spec["label"],
            size=SZ, color=INK, ha="left", va="baseline", z=5))
        # Same two columns as the nine receiver averages above: one numeric rail
        # for every number in the figure.
        for part in craft.num_cell(fig, RAIL_DEC_X, baseline,
                                   fmt_signed(shown["mean"]), size=SZ,
                                   color=INK, z=5):
            strip_parts.append(part)
        strip_parts.append(craft.put_pt(
            fig, RAIL_CI_X, baseline,
            f"[{fmt(shown['ci_low'])}, {fmt(shown['ci_high'])}]",
            size=SZ, color=RULE, ha="left", va="baseline", z=5))

        inventory.append(dict(
            panel="contrast-strip", model="GPT-OSS-120B minus GPT-OSS-20B",
            alias="gpt_oss_120b_minus_gpt_oss_20b",
            series=f"{spec['key']}_contrast",
            role_title=spec["label"],
            form="numeral (no axis, no mark, no shared ruler)",
            # mean/ci_low/ci_high are the STORED values, so `figqa numbers` traces
            # them to a frozen leaf by identity.  oriented_* is what is printed.
            mean=row["stored"]["mean"], ci_low=row["stored"]["ci_low"],
            ci_high=row["stored"]["ci_high"],
            stored_direction=row["stored_direction"],
            oriented_mean=row["oriented"]["mean"],
            oriented_ci_low=row["oriented"]["ci_low"],
            oriented_ci_high=row["oriented"]["ci_high"],
            printed_direction=row["printed_direction"],
            negated_from_stored=row["negated"],
            orientation_note=(
                "Printed in the 120B - 20B orientation. The stored record is "
                + row["stored_direction"]
                + (
                    "; the printed form negates the estimate and swaps the "
                    "interval endpoints, which is exact for a percentile "
                    "interval. It is an orientation, not a new estimate."
                    if row["negated"] else
                    "; stored and printed directions agree, so no negation was "
                    "applied and oriented_* equals mean/ci_low/ci_high."
                )),
            printed_estimate=fmt_signed(row["oriented"]["mean"]),
            printed_interval=(f"[{fmt(row['oriented']['ci_low'])}, "
                              f"{fmt(row['oriented']['ci_high'])}]"),
            N=row["N"], K=row["K"], endpoint=ENDPOINT,
            interval_type=row["interval_type"],
            analysis_status=row["analysis_status"],
            source_path=row["source_path"],
            source_locator=row["source_locator"],
            plotted=False, printed_in_figure=True,
            shown_in_numeric_rail=True,
            reason="H2's home: the only place in the paper where the T contrast "
                   "and the D contrast stand in one field of view. Numerals in "
                   "the rail, not marks on an axis -- method.tex:53-54.",
        ))
    assert len(strip_parts) == len(STRIP_ROWS) * 4

    drawn = dict(fig=fig, ax=ax, inventory=inventory, containers=containers,
                 avg_rules=avg_rules, avg_caps=avg_caps, avg_points=avg_points,
                 leg=leg, handlers=handlers, headers=headers, rail=rail_parts,
                 strip=strip_parts, rows=rows, contrasts=contrasts)
    return drawn


# ---------------------------------------------------------------------------
# 6.  Self-audit and the round's objective checks
# ---------------------------------------------------------------------------

def audit(d) -> dict:
    fig, ax = d["fig"], d["ax"]
    texts = [t for t in fig.findobj(matplotlib.text.Text)
             if t.get_visible() and t.get_text().strip()]
    sizes = sorted({round(float(t.get_fontsize()), 2) for t in texts})
    bold = sorted({t.get_text() for t in texts
                   if str(t.get_fontweight()) in ("bold", "700")})
    bold_sizes = sorted({round(float(t.get_fontsize()), 2) for t in texts
                         if str(t.get_fontweight()) in ("bold", "700")})
    assert set(sizes) <= SIZE_LADDER, sizes
    assert set(bold_sizes) <= {8.0}, bold_sizes
    assert set(bold) == {label for _, label in MODELS}, bold

    widths = set()
    for artist in fig.findobj(Line2D):
        if artist.get_visible() and artist.get_linestyle() != "None":
            widths.add(round(float(artist.get_linewidth()), 2))
    for spine in ax.spines.values():
        widths.add(round(float(spine.get_linewidth()), 2))
    widths.discard(0.0)
    assert widths <= STROKE_LADDER, widths
    assert max(w for w in widths if w < LW_DATA) < LW_DATA

    caps = [c for cont in d["containers"] for c in cont[1]]
    assert len(caps) == 36, len(caps)
    assert all(round(float(c.get_markeredgewidth()), 2) == CAPTHICK for c in caps)

    from matplotlib.patches import FancyBboxPatch
    assert len([p for p in fig.findobj(FancyBboxPatch)]) == 1   # the legend frame

    allowed = {matplotlib.colors.to_hex(c).upper() for c in PALETTE if c != "none"}
    used = set()
    for artist in fig.findobj():
        if not artist.get_visible():
            continue
        if artist is fig.patch or artist is ax.patch:
            continue
        if isinstance(artist, matplotlib.text.Text) and not artist.get_text().strip():
            continue
        if isinstance(artist, Line2D):
            if len(artist.get_xdata()) == 0:
                continue
            if (artist.get_linestyle() in ("None", " ", "")
                    and artist.get_marker() in ("None", "", " ")):
                continue
        for getter in ("get_color", "get_facecolor", "get_edgecolor",
                       "get_markerfacecolor", "get_markeredgecolor"):
            fn = getattr(artist, getter, None)
            if fn is None:
                continue
            try:
                value = fn()
            except Exception:                    # noqa: BLE001
                continue
            if value is None or (isinstance(value, str) and value == "none"):
                continue
            try:
                arr = matplotlib.colors.to_rgba_array(value)
            except (ValueError, TypeError):
                continue
            for rgba in arr:
                if rgba[3] == 0.0:
                    continue
                used.add(matplotlib.colors.to_hex(rgba[:3]).upper())
    assert used <= allowed, sorted(used - allowed)

    coloured = {t.get_text(): matplotlib.colors.to_hex(t.get_color()).upper()
                for t in texts
                if matplotlib.colors.to_hex(t.get_color()).upper() != INK.upper()}
    # PALETTE_SPEC §3.4: no glyph in this figure carries a hue.  Everything that is
    # not INK is RULE, the neutral for secondary numerals and headers.
    assert set(coloured.values()) <= {RULE.upper()}, coloured
    assert all(matplotlib.colors.to_hex(t.get_color()).upper()
               not in {ALLROUTER.upper(), ALLROUTER_DEEP.upper()}
               for t in texts), "a hue reached a glyph"

    frame = leg_frame = d["leg"].get_frame()
    r_pt = float(frame.get_boxstyle().rounding_size) * float(
        leg_frame.get_mutation_scale())

    return dict(distinct_font_sizes=sizes, bold_strings=bold,
                bold_sizes=bold_sizes, text_count=len(texts),
                min_font_pt=min(sizes), stroke_widths=sorted(widths),
                radii_pt=[round(r_pt, 3)], shadow_artists=0,
                saturated_protagonist_cards=0, colours_used=sorted(used),
                coloured_strings=coloured)


def gate_occlusion(inventory) -> dict:
    """CHECK 4.  For every plotted interval: its rendered length in pt against the
    diameter of the marker that sits on it.  `before` is the shipped FINAL v3
    geometry (axes 276 pt wide, the same 27 intervals, line-and-marker throughout);
    `after` is this layout."""
    span_after = AX_X1 - AX_X0
    span_before = 348.0 - 72.0           # FINAL/generators/build_fig3_v3.py
    unit = XLIM[1] - XLIM[0]

    def survey(span):
        short, lengths = [], []
        for r in inventory:
            if not r["plotted"]:
                continue
            length = (r["ci_high"] - r["ci_low"]) * span / unit
            ms = MS_BY_ROLE[r["series"]]
            lengths.append((round(length, 2), r["model"], r["series"]))
            if length < ms:
                short.append(dict(model=r["model"], series=r["series"],
                                  length_pt=round(length, 2), marker_pt=ms))
        lengths.sort()
        return short, lengths

    short_after, lengths_after = survey(span_after)
    short_before, _ = survey(span_before)
    return dict(
        n_intervals=27,
        before=dict(source="FINAL/generators/build_fig3_v3.py, axes 276 pt",
                    n_shorter_than_marker=len(short_before), list=short_before),
        after=dict(source="round4/merged/build_fig3.py, axes 240 pt",
                   n_shorter_than_marker=len(short_after), list=short_after),
        all_lengths_pt=lengths_after,
        shortest_five_pt=lengths_after[:5],
        note="Figure 2's single-layer router column is the 8-of-9 case the check "
             "was written for; Figure 3 has never had one.  The receiver average's "
             "shortest interval (Nano, 4.85 pt) is now a 4.85 pt rule between two "
             "5.6 pt end caps with a 3.8 pt diamond on it, so the interval is still "
             "1.05 pt longer than the mark that sits on it.",
    )


def gate_marker_clearance() -> dict:
    pitch = band_pitch_pt()
    pairs = {}
    for a, b in zip(ROLES, ROLES[1:]):
        gap = abs(ROLE_Y_OFFSETS[ROLES.index(b)] - ROLE_Y_OFFSETS[ROLES.index(a)])
        ha = AVG_CAP_H_PT if a == "D_common" else MS_BY_ROLE[a]
        hb = AVG_CAP_H_PT if b == "D_common" else MS_BY_ROLE[b]
        pairs[f"{a}|{b}"] = round(gap * pitch - (ha + hb) / 2.0, 3)
    band_edge = {}
    for role in ROLES:
        h = AVG_CAP_H_PT if role == "D_common" else MS_BY_ROLE[role]
        dy = ROLE_Y_OFFSETS[ROLES.index(role)]
        band_edge[role] = round(pitch / 2.0 - abs(dy) * pitch - h / 2.0, 3)
    assert min(pairs.values()) >= 1.0, pairs
    assert min(band_edge.values()) >= 1.0, band_edge
    return dict(adjacent_role_clearance_pt=pairs,
                clearance_to_row_edge_pt=band_edge,
                row_pitch_pt=round(pitch, 4))


CAPTION = (
    r"Full-step all-router donor-role effects on fixed receivers, on the primary "
    r"mean-token-log-probability margin; checkpoint order as in "
    r"Figure~\ref{fig:abc-windows}, which carries $T$. Positive $D$ means the error "
    r"donor produces a more error-favoring margin than the clean donor on the same "
    r"receiver. $D_{\rm common}$ averages the two receiver roles: a capped rule over "
    r"its interval, its mean a diamond. One interface, so one hue: the darker step "
    r"is the error receiver. Pointwise 95\% receiver-source-component "
    r"percentile intervals (50,000 draws, seed 20260908; $N=290$, $K=235$), "
    r"conditional on fixed donors. The axis carries $D$ only; it shows role effects, "
    r"not a ranking. The strip below prints both matched-pair direct contrasts as "
    r"numerals on no axis; the $D_{\rm common}$ contrast negates the stored "
    r"20B$-$120B estimate and swaps its endpoints, exact for a percentile interval. "
    r"Their protocols differ: no difference or ratio."
)




