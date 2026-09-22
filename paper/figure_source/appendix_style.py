"""Rendering functions used by the appendix figures."""
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import patch
from matplotlib.figure import Figure
from matplotlib.artist import Artist
from matplotlib.colors import to_hex,to_rgba
from matplotlib.collections import Collection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text
from matplotlib.markers import MarkerStyle

def capture(call):
    """Run just a figure function, intercepting all its save/close operations."""
    captured = []
    def collect(fig, *_args, **_kwargs):
        if not captured:
            captured.append(fig)
        assert captured[0] is fig
    with patch.object(Figure, "savefig", collect), patch.object(plt, "close", lambda *_a, **_k: None):
        call()
    assert len(captured) == 1
    return captured[0]

def artist_census(fig):
    fig.canvas.draw()
    def array(a):
        return np.asarray(a).tolist()
    axes = []
    for ax in fig.axes:
        collections = []
        for c in ax.collections:
            row = {"class": type(c).__name__, "offsets": array(c.get_offsets())}
            if hasattr(c, "get_segments"):
                row["segments"] = [array(s) for s in c.get_segments()]
            if hasattr(c, "get_sizes"):
                row["sizes"] = array(c.get_sizes())
            collections.append(row)
        axes.append({"position": list(ax.get_position().bounds), "xlim": list(ax.get_xlim()), "ylim": list(ax.get_ylim()),
                     "xscale": ax.get_xscale(), "yscale": ax.get_yscale(),
                     "xticks": array(ax.get_xticks()), "yticks": array(ax.get_yticks()),
                     "lines": [{"x": array(l.get_xdata()), "y": array(l.get_ydata()), "marker": str(l.get_marker()), "markersize": l.get_markersize(), "lw": l.get_linewidth()} for l in ax.lines],
                     "collections": collections})
    texts = [{"text": t.get_text(), "position": list(t.get_position()), "fontsize": t.get_fontsize(), "fontfamily": t.get_fontfamily(), "weight": t.get_fontweight()} for t in fig.findobj(Text) if t.get_visible() and t.get_text()]
    return {"size_inches": list(fig.get_size_inches()), "axes": axes, "texts": texts}

def recolour(fig, mapping, palette, *, single_router_old=None):
    def changed(c):
        try:
            rgba = to_rgba(c)
        except (ValueError, TypeError):
            return c
        if rgba[3] == 0:
            return c
        new = mapping.get(to_hex(rgba).lower())
        return (*to_rgba(new)[:3], rgba[3]) if new else c
    for obj in fig.findobj(Artist):
        if isinstance(obj, Text):
            obj.set_color(palette["muted"] if to_hex(obj.get_color()).lower() == "#54616b" else palette["ink"])
        elif isinstance(obj, Line2D):
            old = to_hex(obj.get_color()).lower()
            obj.set_color(changed(obj.get_color()))
            obj.set_markerfacecolor(changed(obj.get_markerfacecolor()))
            obj.set_markeredgecolor(changed(obj.get_markeredgecolor()))
            if single_router_old and old == single_router_old and obj.get_marker() == "o":
                obj.set_markerfacecolor(palette["single_router"])
                obj.set_markeredgecolor(palette["single_router_edge"])
        elif isinstance(obj, Collection):
            for getter, setter in [(obj.get_facecolors, obj.set_facecolors), (obj.get_edgecolors, obj.set_edgecolors)]:
                colors = getter()
                if len(colors):
                    setter([changed(c) for c in colors])
        elif isinstance(obj, Patch):
            obj.set_facecolor(changed(obj.get_facecolor()))
            obj.set_edgecolor(changed(obj.get_edgecolor()))
    fig.patch.set_facecolor(palette["paper"])
    for ax in fig.axes:
        ax.set_facecolor(palette["panel_fill"])
        for s in ax.spines.values():
            s.set_edgecolor(palette["rule"])
        ax.tick_params(axis="both", color=palette["rule"], labelcolor=palette["ink"])

def typography(fig, stem):
    """Register only the final text settings; no data artist moves."""
    from matplotlib.font_manager import findfont
    font_path = findfont("Times New Roman", fallback_to_default=False)
    titles = {id(t) for ax in fig.axes for t in (ax.title, ax._left_title, ax._right_title)
              if t.get_visible() and t.get_text()}
    totals = {"Four windows, four intervention arms", "ProcessBench: correct-to-error residual transfer"}
    changes = []
    for t in fig.findobj(Text):
        if not t.get_visible() or not t.get_text():
            continue
        old = (t.get_fontproperties().copy(), t.get_position(), t.get_ha())
        title = id(t) in titles or t.get_text() in totals
        t.set_fontfamily("Times New Roman")
        t.set_fontsize(9 if title else 8)
        t.set_fontweight("bold" if title else old[0].get_weight())
        if stem == "processbench_receiver_state" and id(t) in titles:
            # Attach each title to the left edge of its own axes.
            t.set_position((0, t.get_position()[1]))
            t.set_ha("left")
        new = (t.get_fontproperties().copy(), t.get_position(), t.get_ha())
        changes.append((t, old, new, {"text": t.get_text(),
            "font_before": old[0].get_family(), "font_after": new[0].get_family(),
            "size_before": old[0].get_size_in_points(), "size_after": new[0].get_size_in_points(),
            "weight_before": old[0].get_weight(), "weight_after": new[0].get_weight(),
            "position_before": list(old[1]), "position_after": list(new[1]),
            "ha_before": old[2], "ha_after": new[2]}))
    return changes, font_path

def set_registered_typography(changes, *, changed):
    for t, old, new, _record in changes:
        properties, position, ha = new if changed else old
        t.set_fontproperties(properties.copy())
        t.set_position(position)
        t.set_ha(ha)

def data_census(fig):
    """Freeze all axes/data geometry, line styles and marker paths separately from text."""
    result = artist_census(fig)
    result.pop("texts")
    for ax, entry in zip(fig.axes, result["axes"], strict=True):
        for line, row in zip(ax.lines, entry["lines"], strict=True):
            row["linestyle"] = line.get_linestyle()
            row["markerfacecolor"] = str(line.get_markerfacecolor())
            row["markeredgecolor"] = str(line.get_markeredgecolor())
        for collection, row in zip(ax.collections, entry["collections"], strict=True):
            row["paths"] = [{"vertices": p.vertices.tolist(), "codes": None if p.codes is None else p.codes.tolist()}
                            for p in collection.get_paths()]
    return result

def layout_census(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # Matplotlib keeps visible=True on tick labels outside the final limits,
    # but Axis.draw suppresses these ticks. Check the actual drawn labels.
    skipped = set()
    for ax in fig.axes:
        for axis, limits in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            lo, hi = sorted(limits)
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                if not lo - 1e-12 <= tick.get_loc() <= hi + 1e-12:
                    skipped.update((id(tick.label1), id(tick.label2)))
    visible = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text() and id(t) not in skipped]
    boxes = [(t, t.get_window_extent(renderer)) for t in visible]
    outside = [t.get_text() for t,b in boxes if b.x0 < -0.5 or b.y0 < -0.5
               or b.x1 > fig.bbox.x1 + 0.5 or b.y1 > fig.bbox.y1 + 0.5]
    overlaps = []
    for i, (a, ba) in enumerate(boxes):
        for b, bb in boxes[i + 1:]:
            w = min(ba.x1, bb.x1) - max(ba.x0, bb.x0)
            h = min(ba.y1, bb.y1) - max(ba.y0, bb.y0)
            if w > 0.5 and h > 0.5:
                overlaps.append([a.get_text(), b.get_text(), w, h])
    assert not outside, outside
    assert not overlaps, overlaps
    assert set(t.get_fontsize() for t in visible) == {8.0, 9.0}
    assert all(t.get_fontfamily() == ["Times New Roman"] for t in visible)
    return {"canvas_width_pt": fig.get_figwidth() * 72, "canvas_height_pt": fig.get_figheight() * 72,
            "paper_scale": 1.0, "font_sizes_pt": [8, 9], "font_family": "Times New Roman",
            "out_of_canvas": outside, "independent_text_overlaps": overlaps,
            "text_count": len(visible)}

def receiver_status_markers(fig, palette):
    """Register the five explicit circle-to-triangle changes, retaining size.

    Only the missed-error scatter, its legend proxy, and three matching
    interval center markers change. No data coordinate, interval or cap moves.
    Restoring these exact objects must recover the frozen PDF drawing stream.
    """
    assert len(fig.axes) == 2
    scatter = [c for c in fig.axes[0].collections if c.get_label() == "Error missed"]
    assert len(scatter) == 1
    legend = fig.axes[0].get_legend()
    assert [t.get_text() for t in legend.get_texts()] == ["Error detected", "Error missed"]
    handles = getattr(legend, "legend_handles", getattr(legend, "legendHandles", None))
    assert len(handles) == 2
    changes = []
    marker = MarkerStyle(palette["pb_missed_marker"])
    triangle = marker.get_path().transformed(marker.get_transform())
    for identifier, obj in (("missed-scatter", scatter[0]), ("missed-legend", handles[1])):
        original = list(obj.get_paths())
        assert len(original) == 1
        old_bounds, new_bounds = original[0].get_extents(), triangle.get_extents()
        assert new_bounds.width <= old_bounds.width and new_bounds.height <= old_bounds.height
        changes.append((obj, "paths", original, [triangle], {
            "id": identifier, "from": "o", "to": "^",
            "size_points_squared": obj.get_sizes().tolist(),
            "number_of_points": len(obj.get_offsets()),
            "old_marker_bounds": list(old_bounds.bounds),
            "new_marker_bounds": list(new_bounds.bounds),
            "marker_width_and_height_not_increased": True,
        }))
    intervals = fig.axes[1].containers
    assert len(intervals) == 6
    for index in (1, 3, 5):
        obj = intervals[index].lines[0]
        assert obj.get_marker() == "o"
        assert to_hex(obj.get_color()).lower() == palette["pb_missed"].lower()
        changes.append((obj, "marker", obj.get_marker(), "^", {
            "id": f"missed-interval-{index}", "from": "o", "to": "^",
            "size_points": obj.get_markersize(),
            "coordinates": [list(obj.get_xdata()), list(obj.get_ydata())],
        }))
    assert all(change[4].get("size_points_squared", [17]) == [17] for change in changes)
    assert all(change[4].get("size_points", 4) == 4 for change in changes)
    set_registered_markers(changes, changed=True)
    return changes

def set_registered_markers(changes, *, changed):
    for obj, kind, old, new, _record in changes:
        getattr(obj, "set_" + kind)(new if changed else old)
