"""Rendering functions used by the paper figures; no model execution."""
import copy
import numpy as np
from matplotlib.colors import to_hex
from matplotlib.patches import Patch,Rectangle
from matplotlib.text import Text
from matplotlib.lines import Line2D

def rounded(value):
    return np.asarray(value, dtype=float).round(9).tolist()

def apply_palette(m, n, p, measurements):
    neutral = {"INK": "ink", "MUTED": "muted", "RULE": "rule", "HAIR": "hair",
               "GRID": "grid", "BAND": "band", "WHITE": "paper", "WINDOW_FILL": "window_fill"}
    for name, key in neutral.items():
        setattr(m, name, p[key])
    if n == 2:
        m.ROUTER, m.ROUTER_DARK = p["single_router"], p["single_router_edge"]
        m.ALLROUTER, m.ALLROUTER_DARK = p["router"], p["router_deep"]
        m.RESIDUAL, m.RESIDUAL_DARK = p["residual"], p["residual_deep"]
        for arm, token, edge in [("router_logits", "single_router", "single_router_edge"),
                                 ("all_router", "router", "router_deep"),
                                 ("post_moe_residual", "residual", "residual_deep")]:
            m.STYLE[arm].update(color=p[token], dark=p[edge],
                mfc=p["paper"] if m.STYLE[arm]["open"] else p[token])
        m.D_STYLE.update(color=p["ink"], mfc=p["ink"])
        # Python bound this default to historical INK when the module was imported.
        m.num_cell.__defaults__ = (m.SZ_BODY, p["ink"], "normal", 5)

        def new_palette_check():
            assert m.STYLE["router_logits"]["color"] == p["single_router"]
            assert m.STYLE["router_logits"]["dark"] == p["single_router_edge"]
            assert m.STYLE["all_router"]["color"] == p["router"]
            assert m.STYLE["post_moe_residual"]["color"] == p["residual"]
            assert measurements["contrast"]["ink"]["window_fill"] >= 7.0
            assert measurements["contrast"]["muted"]["window_fill"] >= 4.5
        m.palette_ladder = new_palette_check
    else:
        # Historical global names were used for the two receiver roles. In this
        # wrapper they are compatibility aliases for the explicit role colors;
        # their old same-hue/lightness interpretation is intentionally retired.
        m.ALLROUTER, m.ALLROUTER_DEEP = p["clean_fill"], p["error_fill"]
        # Retain separate grid segments/geometry, using one soft grid token on both grounds.
        m.GRID_ON_BAND = p["grid"]
        m.ROLE_COLOR = {"D_clean": p["clean_fill"], "D_error": p["error_fill"], "D_common": p["ink"]}
        m.ROLE_LABEL_COLOR = dict.fromkeys(m.ROLES, p["ink"])
    m.PALETTE = {v for v in p.values() if isinstance(v, str) and v.startswith("#")}
    m.PALETTE.update({"white", "none"} if n == 3 else set())

def apply_surfaces(fig, m, n, p, drawn):
    """Add only named, stroke-free reading surfaces; preserve all original artists.

    New rectangles are returned by identity, not found by a broad type or colour
    filter. Only these exact new objects may be omitted from the before/after
    geometry census. Recoloured original patches and text remain in that census.
    Each allowed colour mutation records its exact object, channel and old/new
    value so the full original colour-channel census can be restored and checked.
    """
    additions, recoloured, report = [], [], []

    def ground(name, bounds, token, z):
        x, y, width, height = bounds
        assert width > 0 and height > 0 and z < 1.0
        rect = Rectangle((x, y), width, height, transform=m.craft.pt_transform(fig),
                         facecolor=p[token], edgecolor="none", linewidth=0, zorder=z)
        rect.set_gid(f"rv-depth-background-fig{n}-{name}")
        fig.add_artist(rect)
        additions.append(rect)
        report.append({"name": name, "kind": "new_non_data_background",
                       "bounds_pt": list(bounds), "token": token, "zorder": z})

    def recolour(artist, channel, token, name):
        assert artist not in additions
        assert ((isinstance(artist, Patch) and channel in ("facecolor", "edgecolor"))
                or (isinstance(artist, Text) and channel == "color"))
        before = copy.deepcopy(getattr(artist, "get_" + channel)())
        getattr(artist, "set_" + channel)(p[token])
        assert to_hex(getattr(artist, "get_" + channel)()).upper() == p[token].upper()
        recoloured.append((artist, channel, before, p[token]))
        report.append({"name": name, "kind": "existing_text_colour_only" if isinstance(artist, Text)
                       else "existing_patch_colour_only", "channel": channel, "token": token,
                       "before": to_hex(before).upper(), "after": p[token].upper()})

    def unique_text(label):
        matches = [a for a in fig.findobj(Text) if a.get_text() == label and a.get_visible()]
        assert len(matches) == 1, f"expected one label: {label!r}, got {len(matches)}"
        return matches[0]

    if n == 2:
        ground("schematic-floor", (6, m.WIN_LABEL_BASE - 4, 384,
               m.H_PT - 6 - (m.WIN_LABEL_BASE - 4)), "panel_fill", 0.05)
        ground("model-row-floor", (6, m.BLOCK_BOT - m.AX_PAD, 384,
               m.BLOCK_TOP - m.BLOCK_BOT + 2 * m.AX_PAD), "panel_fill", 0.05)
        for i in range(0, 9, 2):
            ground(f"model-row-{i}-band", (6, m.BLOCK_TOP - (i + 1) * m.PITCH,
                   384, m.PITCH), "band", 0.15)
        windows = [a for a in fig.artists if isinstance(a, Patch) and
                   to_hex(a.get_facecolor()).upper() == p["window_fill"].upper()]
        assert len(windows) == 2, "the two original equal-coverage windows must be identifiable"
        windows.sort(key=lambda a: a.get_path().transformed(a.get_transform()).get_extents().x0)
        cell_w = (m.STRIP[1] - m.STRIP[0]) / m.N_CELLS
        first_token_bounds = [m.STRIP[0] + m.DIV_CELL * cell_w,
                              m.STRIP_Y0 + 0.45, cell_w,
                              m.STRIP_Y1 - m.STRIP_Y0 - 0.9]
        first_token = [a for a in fig.artists if isinstance(a, Rectangle)
                       and a not in additions
                       and np.allclose([a.get_x(), a.get_y(), a.get_width(), a.get_height()],
                                       first_token_bounds, atol=1e-9, rtol=0)]
        assert len(first_token) == 1, "first-divergence token must be identified by its actual cell bounds"
        recolour(first_token[0], "facecolor", "divergence_marker_fill", "first-divergence-token-fill")
        for window, bounds, label, key in zip(windows, (m.DIV_WIN, m.SUF_WIN),
                ("divergence-8", "suffix-8"), ("window_div", "window_suffix")):
            extents_pt = window.get_path().transformed(window.get_transform()).get_extents().extents * 72 / fig.dpi
            lo, hi = bounds
            assert hi - lo == 8
            assert np.allclose(extents_pt, [m.STRIP[0] + lo * cell_w, m.STRIP_Y0,
                m.STRIP[0] + hi * cell_w, m.STRIP_Y1], atol=1e-9, rtol=0), \
                f"window identity does not match its token positions: {label}"
            recolour(window, "facecolor", key + "_fill", label + "-fill")
            recolour(window, "edgecolor", key + "_edge", label + "-outline")
            recolour(unique_text(label), "color", key + "_text", label + "-label")
        for label, key in (("Single-layer router", "single_router_edge"),
                           ("All-router", "router_deep"),
                           ("Single-layer residual", "residual_deep")):
            recolour(unique_text(label), "color", key, "legend-" + label)
        # Actual window path extents, including the outline's rounded path.
        widths = [a.get_path().transformed(a.get_transform()).get_extents().width
                  for a in windows]
        assert np.isclose(widths[0], widths[1], atol=1e-9, rtol=0)
        assert all(ax.patch.get_alpha() == 0.0 for ax in fig.axes), "axes must not hide row surfaces"
    else:
        ground("model-row-floor", (m.BLOCK_X0, m.AX_Y0, m.BLOCK_X1 - m.BLOCK_X0,
               m.AX_Y1 - m.AX_Y0), "panel_fill", 0.05)
        pitch = m.band_pitch_pt()
        for i in range(0, 9, 2):
            ground(f"model-row-{i}-full-width-band", (m.BLOCK_X0,
                   m.row_centre_pt(i) - pitch / 2, m.BLOCK_X1 - m.BLOCK_X0,
                   pitch), "band", 0.15)
        recolour(drawn["leg"].get_frame(), "facecolor", "legend_fill", "legend-ground")
        strips = [a for a in fig.artists if isinstance(a, Rectangle) and
                  a not in additions and np.isclose(a.get_x(), m.STRIP_X0) and
                  np.isclose(a.get_y(), m.STRIP_Y0) and
                  np.isclose(a.get_width(), m.STRIP_X1 - m.STRIP_X0)]
        assert len(strips) == 1, "the original matched-pair strip must be uniquely identified"
        recolour(strips[0], "facecolor", "paper", "matched-pair-contrast-ground")
        assert not drawn["ax"].patch.get_visible(), "axes must not hide row surfaces"
    assert len({a.get_gid() for a in additions}) == len(additions)
    for patch in additions:
        assert patch.get_linewidth() == 0 and patch.get_edgecolor()[3] == 0
    return additions, recoloured, report

def apply_final_layout(fig, m, n):
    """Move only the six exact title texts and three complete legend groups.

    Every mutation records the exact artist and coordinates and is reversed for
    the strict pre/post census. Scientific axes, data artists and other text are
    never broadly excluded or baselined anew.
    """
    changes, report = [], []
    if n != 2:
        return changes, report
    titles = {"B", "Full-step $T$", "C", "Residual − all-router", "D", "Gap shift"}
    legend_labels = {m.STYLE[a]["label"] for a in m.ARMS_TOP_TO_BOTTOM}
    title_dy = (m.BLOCK_TOP + 10.0) - m.LETTER_BASE
    legend_dy = (m.BLOCK_TOP + 25.0) - m.LEGEND_CY
    for label in sorted(titles | legend_labels):
        matches = [a for a in fig.findobj(Text) if a.get_visible() and a.get_text() == label]
        assert len(matches) == 1, f"layout text identity is ambiguous: {label!r}"
        a = matches[0]
        before = tuple(a.get_position())
        dy = title_dy if label in titles else legend_dy
        expected_y = m.LETTER_BASE if label in titles else m.LEGEND_CY - m.CAP * m.SZ_BODY / 2.0
        assert np.isclose(before[1], expected_y, atol=1e-9, rtol=0)
        after = (before[0], before[1] + dy)
        a.set_position(after)
        changes.append((a, "position", before, after))
        report.append({"kind": "panel_title_text" if label in titles else "legend_text",
                       "text": label, "before_pt": list(before), "after_pt": list(after)})
    handles = [a for a in fig.artists if isinstance(a, Line2D)
               and a.get_zorder() in (4.5, 4.7)
               and np.all(np.abs(np.asarray(a.get_ydata(), dtype=float) - m.LEGEND_CY) <= 1.5)]
    assert len(handles) == 12, "three legend intervals must each retain line, two caps and marker"
    for a in handles:
        before = np.asarray(a.get_ydata(), dtype=float).copy()
        after = before + legend_dy
        a.set_ydata(after)
        changes.append((a, "ydata", before, after))
        report.append({"kind": "legend_mark", "marker": str(a.get_marker()),
                       "x_pt": rounded(a.get_xdata()), "before_y_pt": rounded(before),
                       "after_y_pt": rounded(after)})
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = lambda label: next(a for a in fig.findobj(Text) if a.get_text() == label).get_window_extent(renderer).extents * 72 / fig.dpi
    title_boxes = [bounds(label) for label in titles]
    legend_boxes = [bounds(label) for label in legend_labels]
    gap = min(b[1] for b in legend_boxes) - max(b[3] for b in title_boxes)
    assert gap >= 3.0, f"legend/title whitespace must be at least 3 pt, got {gap}"
    assert min(b[1] for b in title_boxes) - (m.BLOCK_TOP + m.AX_PAD) >= 3.0
    for label, x in (("B", m.B_AX[0]), ("C", m.C_LEFT), ("D", m.D_AX[0])):
        assert np.isclose(next(a for a in fig.findobj(Text) if a.get_text() == label).get_position()[0], x)
    report.append({"kind": "layout_clearance", "legend_above_titles_gap_pt": float(gap),
                   "title_baseline_pt": m.BLOCK_TOP + 10.0,
                   "legend_centre_pt": m.BLOCK_TOP + 25.0})
    return changes, report
