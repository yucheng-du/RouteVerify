"""The archived ProcessBench figure function; CPU rendering only."""
import numpy as np
RUNS={"gpt_oss_20b":None,"gemma4":None,"nemotron3_nano":None}
NAMES={"gpt_oss_20b":"GPT-OSS","gemma4":"Gemma 4","nemotron3_nano":"Nano"}

def figures(output, pairs, subgroup):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 3.05), gridspec_kw={"width_ratios": [1, 1.12]})
    fig.subplots_adjust(left=0.11, right=0.97, bottom=0.19, top=0.80, wspace=0.68)
    colors = {True: "#196C9F", False: "#C77728"}
    gp = [
        r
        for r in pairs
        if r["model_id"] == "gemma4"
        and r["site_family"] == "post_moe_residual"
        and r["direction"] == "clean_to_corrupt"
    ]
    gp.sort(key=lambda r: r["delta"])
    for correct in (True, False):
        axes[0].scatter(
            [i + 1 for i, r in enumerate(gp) if r["receiver_judgment_correct"] == correct],
            [r["delta"] for r in gp if r["receiver_judgment_correct"] == correct],
            s=17,
            c=colors[correct],
            linewidths=0,
            label="Error detected" if correct else "Error missed",
        )
    axes[0].axhline(0, color=".5", lw=0.7)
    axes[0].axhline(np.mean([r["delta"] for r in gp]), color=".3", lw=1, linestyle="--")
    axes[0].set(
        xlabel="Pair rank (all 64)",
        ylabel="Residual effect on margin",
        title="(a) Gemma pair effects",
    )
    axes[0].legend(loc="lower right", fontsize=8, frameon=False, handletextpad=0.4)
    axes[0].set_xticks([1, 16, 32, 48, 64])
    y = 5
    ticks, labels = [], []
    for model in RUNS:
        for correct in (True, False):
            r = next(
                r
                for r in subgroup
                if r["model_id"] == model
                and r["site_family"] == "post_moe_residual"
                and r["direction"] == "clean_to_corrupt"
                and r["receiver_judgment_correct"] == correct
            )
            axes[1].errorbar(
                r["estimate"],
                y,
                xerr=[[r["estimate"] - r["ci_low"]], [r["ci_high"] - r["estimate"]]],
                color=colors[correct],
                fmt="o",
                markersize=4,
                capsize=2,
                lw=1,
            )
            ticks.append(y)
            labels.append(f"{NAMES[model]} ({r['n_pairs']})")
            y -= 1
        y -= 0.6
    axes[1].axvline(0, color=".5", lw=0.7)
    axes[1].set_yticks(ticks, labels)
    axes[1].set(xlabel="Mean effect (95% CI)", title="(b) Error detection state")
    axes[1].set_ylim(y + 0.1, 5.8)
    axes[1].set_xlim(-0.20, 0.63)
    axes[1].set_xticks([-0.2, 0, 0.3, 0.6])
    # Target-size artwork: 5.5 in wide at scale 1; no bbox cropping or font downscaling.
    fig.suptitle("ProcessBench: correct-to-error residual transfer", y=0.98, fontsize=10)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside = []
    for text in fig.findobj(matplotlib.text.Text):
        if text.get_visible() and text.get_text():
            bbox = text.get_window_extent(renderer)
            if (
                bbox.x0 < -1
                or bbox.y0 < -1
                or bbox.x1 > fig.bbox.x1 + 1
                or bbox.y1 > fig.bbox.y1 + 1
            ):
                outside.append(text.get_text())
    assert not outside, outside
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output / "receiver_state_explanation.pdf", metadata={"CreationDate": None, "ModDate": None}
    )
    fig.savefig(output / "receiver_state_explanation.png", dpi=240)
    plt.close(fig)
    return {
        "width_inches": 5.5,
        "height_inches": 3.05,
        "intended_latex_width": "5.5in = manuscript textwidth at scale 1",
        "smallest_font_points": 8,
        "all_text_within_canvas": True,
        "final_manuscript_embedding_verified": False,
        "matplotlib_version": matplotlib.__version__,
    }
