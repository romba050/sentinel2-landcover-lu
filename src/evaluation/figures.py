"""Result figures for the write-up.

Chart choices follow one rule: the encoding is picked from the data's job, and
colour is assigned last.

*   Confusion matrix -> magnitude, so a single-hue sequential ramp (never a
    rainbow, which invents category boundaries where the data has none).
*   Accuracy vs boundary distance -> one series, so one hue and no legend; the
    title names it. Sample counts are annotated as text rather than put on a
    second y-axis, because a dual-axis chart lets the reader infer a crossing
    that is purely an artefact of two arbitrary scales.
*   Leakage sweep -> two series that must stay distinguishable, so a
    CVD-validated blue/orange pair plus distinct markers and direct labels, so
    identity never rests on colour alone.
*   Land-cover maps -> categorical, using the CORINE colour convention, because
    a domain-standard palette beats a prettier invented one.

    uv run python -m src.evaluation.figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import rasterio  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from src.ingest.preview import stretch  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS = PROJECT_ROOT / "data" / "results"
FIGURES = RESULTS / "figures"

# Validated categorical pair (light surface): worst-pair CVD dE 24.7,
# normal-vision dE 33.6, both well clear of the floors.
SERIES_1 = "#2a78d6"  # blue
SERIES_2 = "#eb6834"  # orange
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#dcdcd8"
SEQUENTIAL = "Blues"


def _style(ax, *, xlabel="", ylabel="", title=""):
    """Recessive axes: the data should be the most prominent thing on the page."""
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_MUTED, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_MUTED, fontsize=10)
    if title:
        ax.set_title(title, color=INK, fontsize=12, loc="left", pad=12)
    return ax


# --------------------------------------------------------------------------- #


def fig_confusion(summary: dict, out: Path):
    report = summary["spatial_block_split"]
    cm = np.array(report["confusion_matrix"], dtype=float)
    names = report["class_names"]
    # Row-normalised: each row reads as "of the pixels that truly are X, what
    # fraction went where". Raw counts would just re-draw the class imbalance.
    recall = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)

    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(recall, cmap=SEQUENTIAL, vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=35, ha="right")
    ax.set_yticks(range(len(names)), names)
    for i in range(len(names)):
        for j in range(len(names)):
            v = recall[i, j]
            if v < 0.005:
                continue
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.5,
                    color="white" if v > 0.55 else INK)
    _style(ax, xlabel="predicted", ylabel="CORINE label",
           title="Confusion matrix, spatially-blocked split (row-normalised)")
    ax.set_xticks(np.arange(-0.5, len(names)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(names)), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)
    fig.colorbar(im, ax=ax, fraction=0.045, label="fraction of true class")
    fig.text(0.01, 0.01,
             f"overall accuracy {report['overall_accuracy']:.3f}  |  "
             f"macro-F1 {report['macro_f1']:.3f}  |  "
             f"kappa {report['cohen_kappa']:.3f}  |  "
             f"{report['n_test_pixels']:,} held-out pixels",
             fontsize=8.5, color=INK_MUTED)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def fig_boundary_distance(summary: dict, out: Path):
    rows = summary["accuracy_by_boundary_distance"]
    labels = [
        f"{r['min_distance_m']:.0f}-{r['max_distance_m']:.0f}"
        if r["max_distance_m"] else f">{r['min_distance_m']:.0f}"
        for r in rows
    ]
    acc = [r["accuracy"] for r in rows]
    n = [r["n_pixels"] for r in rows]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(range(len(rows)), acc, color=SERIES_1, linewidth=2,
            marker="o", markersize=8, markerfacecolor=SERIES_1,
            markeredgecolor="white", markeredgewidth=2, zorder=3)
    for i, (a, count) in enumerate(zip(acc, n, strict=True)):
        ax.annotate(f"{a:.2f}", (i, a), textcoords="offset points", xytext=(0, 12),
                    ha="center", fontsize=9, color=INK)
        # Sample size as text, not a second axis.
        ax.annotate(f"n={count / 1000:.0f}k", (i, a), textcoords="offset points",
                    xytext=(0, -18), ha="center", fontsize=8, color=INK_MUTED)
    ax.set_xticks(range(len(rows)), labels)
    ax.set_ylim(0.35, 0.92)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    _style(ax, xlabel="distance from the nearest CORINE class boundary (m)",
           ylabel="overall accuracy",
           title="Model accuracy rises steeply away from CORINE polygon edges")
    fig.text(0.01, -0.02,
             "CORINE is photo-interpreted at 1:100 000, so its boundaries are only "
             "accurate to ~100 m. Much of the apparent\nmodel error near an edge is "
             "the label being wrong, not the prediction.",
             fontsize=9, color=INK_MUTED)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def fig_leakage(summary: dict, out: Path):
    rows = summary.get("leakage_vs_training_density") or []
    if not rows:
        return None
    x = [r["n_train_pixels"] for r in rows]
    blocked = [r["spatial_accuracy"] for r in rows]
    random_ = [r["random_accuracy"] for r in rows]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(x, random_, color=SERIES_2, linewidth=2, marker="s", markersize=8,
            markeredgecolor="white", markeredgewidth=2, label="random pixel split",
            zorder=3)
    ax.plot(x, blocked, color=SERIES_1, linewidth=2, marker="o", markersize=8,
            markeredgecolor="white", markeredgewidth=2, label="spatially-blocked split",
            zorder=3)
    ax.fill_between(x, blocked, random_, color=SERIES_2, alpha=0.10, zorder=1)

    # Direct labels so identity never depends on colour alone.
    ax.annotate("random pixel split", (x[-1], random_[-1]), textcoords="offset points",
                xytext=(-8, 12), ha="right", fontsize=10, color=SERIES_2, weight="bold")
    ax.annotate("spatially-blocked split", (x[-1], blocked[-1]),
                textcoords="offset points", xytext=(-8, -20), ha="right",
                fontsize=10, color=SERIES_1, weight="bold")
    mid = len(rows) // 2
    ax.annotate(
        f"gap = {rows[-1]['accuracy_inflation']:+.3f}",
        (x[mid], (blocked[mid] + random_[mid]) / 2),
        fontsize=9, color=INK_MUTED, ha="center",
    )

    ax.set_xscale("log")
    ax.set_xticks(x, [f"{v // 1000:,}k" for v in x])
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left", fontsize=9, labelcolor=INK_MUTED)
    _style(ax, xlabel="training pixels (log scale)", ylabel="overall accuracy",
           title="A random split's advantage is leakage, and it grows with training density")
    fig.text(0.01, -0.04,
             "Adding 64x more training data barely moves the blocked score: the extra pixels "
             "teach the model nothing new\nabout the landscape. The random split keeps "
             "climbing because those pixels are near-duplicates of its test pixels.",
             fontsize=9, color=INK_MUTED)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def fig_feature_importance(summary: dict, out: Path):
    rows = summary["feature_importance"]
    names = [r["feature"] for r in rows][::-1]
    vals = [r["importance"] for r in rows][::-1]

    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.barh(names, vals, color=SERIES_1, height=0.62)
    for i, v in enumerate(vals):
        ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=8.5, color=INK_MUTED)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(0, max(vals) * 1.18)
    _style(ax, xlabel="Gini importance", title="Which features the forest actually uses")
    fig.text(0.01, -0.02,
             "SWIR (B11/B12) and the SWIR-based NBR2 dominate; NDVI barely registers. In a "
             "late-August scene almost\neverything vegetated has saturated NDVI, so it "
             "cannot separate forest from pasture from crops.",
             fontsize=9, color=INK_MUTED)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def fig_maps(summary: dict, ingest: dict, labels_meta: dict, out: Path):
    with rasterio.open(PROCESSED / ingest["files"]["stack"]) as src:
        names = list(src.descriptions)
        rgb = np.dstack([stretch(src.read(names.index(b) + 1)) for b in ("B04", "B03", "B02")])
    with rasterio.open(PROCESSED / labels_meta["files"]["labels"]) as src:
        truth = src.read(1)
    with rasterio.open(RESULTS / summary["files"]["prediction"]) as src:
        pred = src.read(1)
    with rasterio.open(PROCESSED / labels_meta["files"]["boundary_distance"]) as src:
        dist = src.read(1)

    kept = [c for c in labels_meta["class_scheme"]["classes"] if c["kept"]]
    ids = [0] + [c["class_id"] for c in kept]
    cmap = ListedColormap(["#ffffff"] + [c["colour"] for c in kept])
    norm = BoundaryNorm(ids + [max(ids) + 1], cmap.N)

    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("Sentinel-2 true colour, 2024-08-24", color=INK, fontsize=12)

    axes[0, 1].imshow(truth, cmap=cmap, norm=norm, interpolation="nearest")
    axes[0, 1].set_title("CORINE 2018 (label)", color=INK, fontsize=12)
    axes[0, 1].legend(
        handles=[Patch(fc=c["colour"], ec="k", lw=0.4, label=c["name"]) for c in kept],
        loc="lower right", fontsize=8, framealpha=0.92,
    )

    axes[1, 0].imshow(pred, cmap=cmap, norm=norm, interpolation="nearest")
    axes[1, 0].set_title(
        f"Random Forest prediction (blocked-split accuracy "
        f"{summary['spatial_block_split']['overall_accuracy']:.2f})",
        color=INK, fontsize=12,
    )

    # Disagreement, split by whether the label can be trusted there.
    labelled = truth != 0
    wrong = labelled & (pred != truth)
    near = wrong & (dist < 100)
    far = wrong & (dist >= 100)
    disagreement = np.zeros(truth.shape, dtype=np.uint8)
    disagreement[near] = 1
    disagreement[far] = 2
    axes[1, 1].imshow(rgb)
    axes[1, 1].imshow(
        np.ma.masked_where(disagreement == 0, disagreement),
        cmap=ListedColormap([SERIES_1, SERIES_2]), vmin=1, vmax=2, alpha=0.75,
        interpolation="nearest",
    )
    axes[1, 1].set_title("Where model and CORINE disagree", color=INK, fontsize=12)
    near_share, far_share = near.sum() / wrong.sum(), far.sum() / wrong.sum()
    axes[1, 1].legend(
        handles=[
            Patch(fc=SERIES_1, label=f"within 100 m of a boundary ({near_share:.0%})"),
            Patch(fc=SERIES_2, label=f"deep inside a polygon ({far_share:.0%})"),
        ],
        loc="lower right", fontsize=9, framealpha=0.92,
    )

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out, dpi=100, bbox_inches="tight", facecolor="white",
                pil_kwargs={"quality": 88})
    plt.close(fig)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out-dir", type=Path, default=FIGURES)
    args = p.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary = json.loads((RESULTS / "latest_rf.json").read_text())
    ingest = json.loads((PROCESSED / "latest_ingest.json").read_text())
    labels_meta = json.loads((PROCESSED / "latest_labels.json").read_text())

    written = [
        fig_confusion(summary, args.out_dir / "fig1_confusion_matrix.png"),
        fig_boundary_distance(summary, args.out_dir / "fig2_accuracy_vs_boundary_distance.png"),
        fig_leakage(summary, args.out_dir / "fig3_leakage_vs_training_density.png"),
        fig_feature_importance(summary, args.out_dir / "fig4_feature_importance.png"),
        fig_maps(summary, ingest, labels_meta, args.out_dir / "fig5_prediction_maps.jpg"),
    ]
    for path in filter(None, written):
        print(f"wrote {path.relative_to(PROJECT_ROOT)} ({path.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
