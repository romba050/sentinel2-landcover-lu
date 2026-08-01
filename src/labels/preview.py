"""Visual check that the CORINE labels line up with the Sentinel-2 imagery.

Statistics cannot catch a CRS or rasterisation bug -- a label raster that is
shifted, flipped or in the wrong projection still produces a perfectly sensible
class histogram. Only looking at it does.

    uv run python -m src.labels.preview
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
FIGURES = PROJECT_ROOT / "data" / "results" / "figures"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--processed-dir", type=Path, default=PROCESSED)
    p.add_argument("--out-dir", type=Path, default=FIGURES)
    args = p.parse_args(argv)

    ingest = json.loads((args.processed_dir / "latest_ingest.json").read_text())
    labels_meta = json.loads((args.processed_dir / "latest_labels.json").read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(args.processed_dir / ingest["files"]["stack"]) as src:
        names = list(src.descriptions)
        rgb = np.dstack([
            stretch(src.read(names.index(b) + 1)) for b in ("B04", "B03", "B02")
        ])
    with rasterio.open(args.processed_dir / labels_meta["files"]["labels"]) as src:
        labels = src.read(1)
    with rasterio.open(args.processed_dir / labels_meta["files"]["boundary_distance"]) as src:
        dist = src.read(1)

    kept = [c for c in labels_meta["class_scheme"]["classes"] if c["kept"]]
    ids = [0] + [c["class_id"] for c in kept]
    cmap = ListedColormap(["#ffffff"] + [c["colour"] for c in kept])
    norm = BoundaryNorm(list(ids) + [max(ids) + 1], cmap.N)

    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    fig.suptitle(
        f"CORINE 2018 labels on the Sentinel-2 grid  |  "
        f"{ingest['stac']['item_id']}  |  EPSG:{labels_meta['grid']['epsg']}",
        fontsize=13,
    )

    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("Sentinel-2 true colour")

    axes[0, 1].imshow(labels, cmap=cmap, norm=norm, interpolation="nearest")
    axes[0, 1].set_title("CORINE labels, aggregated to 7 classes")

    # The overlay is the actual alignment test: class edges must follow the
    # image, not float somewhere near it.
    axes[1, 0].imshow(rgb)
    axes[1, 0].imshow(labels, cmap=cmap, norm=norm, interpolation="nearest", alpha=0.45)
    axes[1, 0].set_title("Overlay (labels at 45% opacity)")

    buffer_m = labels_meta["limitations"]["boundary_buffer_m"]
    frac = labels_meta["limitations"]["fraction_within_boundary_buffer"]
    im = axes[1, 1].imshow(np.where(labels == 0, np.nan, dist), cmap="magma", vmax=400)
    axes[1, 1].set_title(
        f"Distance to class boundary (m)\n{frac:.0%} of pixels are within {buffer_m:.0f} m"
    )
    fig.colorbar(im, ax=axes[1, 1], fraction=0.046, label="m")

    handles = [Patch(fc=c["colour"], ec="k", lw=0.4,
                     label=f"{c['class_id']} {c['name']} ({c['share']:.1%})") for c in kept]
    axes[0, 1].legend(handles=handles, loc="lower right", fontsize=7.5, framealpha=0.92)

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    out = args.out_dir / "corine_labels_overlay.jpg"
    fig.tight_layout()
    fig.savefig(out, dpi=100, bbox_inches="tight", pil_kwargs={"quality": 88})
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
