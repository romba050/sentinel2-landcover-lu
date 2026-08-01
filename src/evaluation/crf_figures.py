"""Figures for the CRF stage.

Fig 6 -- boundary-distance curves, unary vs CRF. The instrument that separates
"the map got tidier" from "the map got righter": spatial context should help
most deep inside CORINE polygons, where the label is trustworthy and speckle
is pure error, and do little near boundaries, where the label itself is soft.

Fig 7 -- the theta sweep on a crop containing the airport runway, motorways
and settlement fingers. Thin connected structures are what over-smoothing
destroys first (and the closest visual analogue to the retinal vessels the
thesis method segmented), so the sweep is shown rather than asserted: roads
surviving at the chosen theta, dissolving at the high one.

    uv run python -m src.evaluation.crf_figures
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

from src.evaluation.figures import GRID, INK, INK_MUTED, SERIES_1, SERIES_2, _style  # noqa: E402
from src.ingest.preview import stretch  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS = PROJECT_ROOT / "data" / "results"
FIGURES = RESULTS / "figures"

#: Crop for the road-survival panel (row/col slices on the 10 m grid):
#: Kirchberg, Findel airport and the motorway junctions east of the city.
CROP = (slice(350, 1100), slice(950, 1850))


def _band_labels(rows):
    return [
        f"{r['min_distance_m']:.0f}-{r['max_distance_m']:.0f}"
        if r["max_distance_m"] else f">{r['min_distance_m']:.0f}"
        for r in rows
    ]


def fig_boundary(summary: dict, out: Path):
    chosen = f"{summary['method']['theta_chosen']:g}"
    unary = summary["results"]["unary"]["accuracy_by_boundary_distance"]
    crf_rows = summary["results"][chosen]["accuracy_by_boundary_distance"]
    x = range(len(unary))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for rows, colour, marker, label in (
        (unary, SERIES_1, "o", "Random Forest (unary)"),
        (crf_rows, SERIES_2, "s", f"+ mean-field CRF (θ={chosen})"),
    ):
        acc = [r["accuracy"] for r in rows]
        ax.plot(x, acc, color=colour, linewidth=2, marker=marker, markersize=8,
                markeredgecolor="white", markeredgewidth=2, label=label, zorder=3)

    for i, (u, c) in enumerate(zip(unary, crf_rows, strict=True)):
        gain = c["accuracy"] - u["accuracy"]
        ax.annotate(f"{gain:+.3f}", (i, max(u["accuracy"], c["accuracy"])),
                    textcoords="offset points", xytext=(0, 14), ha="center",
                    fontsize=8.5, color=INK_MUTED)

    ax.set_xticks(list(x), _band_labels(unary))
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left", fontsize=9.5, labelcolor=INK_MUTED)
    _style(ax, xlabel="distance from the nearest CORINE class boundary (m)",
           ylabel="accuracy (all labelled pixels, out-of-fold)",
           title="CORINE agreement barely moves — the CRF's gain is structural, not score")
    fig.text(0.01, -0.06,
             "Both curves come from the identical out-of-fold Random Forest; the only "
             "difference is mean-field smoothing.\nThe expected pattern was gains "
             "concentrating deep inside polygons, where labels are trustworthy; the "
             "measured gain is a\nuniform ~+0.005. The CRF's real effect is in map "
             "structure (speckle and patch metrics), which this instrument cannot see.",
             fontsize=9, color=INK_MUTED)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def fig_sweep(summary: dict, labels_meta: dict, ingest: dict, out: Path):
    kept = [c for c in labels_meta["class_scheme"]["classes"] if c["kept"]]
    ids = [0] + [c["class_id"] for c in kept]
    cmap = ListedColormap(["#ffffff"] + [c["colour"] for c in kept])
    norm = BoundaryNorm(ids + [max(ids) + 1], cmap.N)

    with rasterio.open(PROCESSED / ingest["files"]["stack"]) as src:
        names = list(src.descriptions)
        rgb = np.dstack([
            stretch(src.read(names.index(b) + 1))[CROP] for b in ("B04", "B03", "B02")
        ])

    def read_crop(name):
        with rasterio.open(RESULTS / name) as src:
            return src.read(1)[CROP]

    chosen = float(summary["method"]["theta_chosen"])
    sweep = sorted(float(t) for t in summary["files"]["sweep"])
    # Five map panels tell the story: what the ground looks like, what the
    # unary produces, a gentle theta, the chosen theta, and the bulldozer
    # counterfactual -- same chosen theta with the contrast term removed.
    mild = min(sweep)
    panels = [("Sentinel-2 true colour", rgb, None),
              ("Random Forest (unary, out-of-fold)",
               read_crop(summary["files"]["unary"]), "unary"),
              (f"θ = {mild:g}",
               read_crop(summary["files"]["sweep"][f"{mild:g}"]), f"{mild:g}"),
              (f"θ = {chosen:g}  — chosen",
               read_crop(summary["files"]["crf"]), f"{chosen:g}"),
              (f"θ = {chosen:g}, contrast term removed — the bulldozer case",
               read_crop(summary["files"]["ablation"]), "ablation")]

    fig, axes = plt.subplots(2, 3, figsize=(19, 10.5))
    for ax, (title, img, key) in zip(axes.ravel(), panels, strict=False):
        if img.ndim == 3:
            ax.imshow(img)
        else:
            ax.imshow(img, cmap=cmap, norm=norm, interpolation="nearest")
        subtitle = ""
        if key is not None:
            r = summary["results"][key]
            subtitle = (f"\nspecks {r['fragmentation']['tiny_patch_pixel_share']:.2%} · "
                        f"thin-road survival {r['thin_artificial_retention']:.0%}")
        ax.set_title(title + subtitle, color=INK, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    # last axis: legend
    ax = axes.ravel()[-1]
    if len(panels) < 6:
        ax.axis("off")
        ax.legend(
            handles=[Patch(fc=c["colour"], ec="k", lw=0.4, label=c["name"]) for c in kept],
            loc="center", fontsize=11, frameon=False,
        )

    fig.suptitle(
        "Road survival across the pairwise-strength sweep — thin structures are what "
        "over-smoothing destroys first",
        fontsize=13, color=INK,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight", facecolor="white",
                pil_kwargs={"quality": 88})
    plt.close(fig)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out-dir", type=Path, default=FIGURES)
    args = p.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary = json.loads((RESULTS / "latest_crf.json").read_text())
    labels_meta = json.loads((PROCESSED / "latest_labels.json").read_text())
    ingest = json.loads((PROCESSED / "latest_ingest.json").read_text())

    written = [
        fig_boundary(summary, args.out_dir / "fig6_crf_boundary_distance.png"),
        fig_sweep(summary, labels_meta, ingest, args.out_dir / "fig7_crf_theta_sweep.jpg"),
    ]
    for path in written:
        print(f"wrote {path.relative_to(PROJECT_ROOT)} ({path.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
