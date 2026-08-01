"""Quick-look PNGs from an ingested Sentinel-2 stack.

Serves two purposes: a sanity check that the ingest is georeferenced and scaled
correctly (does it actually look like Luxembourg City?), and figures for the
project write-up.

    uv run python -m src.ingest.preview
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
from matplotlib.colors import ListedColormap  # noqa: E402

from src.ingest.sentinel2 import SCL_CLASSES  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"
FIGURES = PROJECT_ROOT / "data" / "results" / "figures"

#: Roughly the ESA SCL palette, so the QA image reads the same as in SNAP/QGIS.
SCL_COLOURS = {
    0: "#000000", 1: "#ff0000", 2: "#2f2f2f", 3: "#643200", 4: "#00a000",
    5: "#ffe65a", 6: "#0000ff", 7: "#808080", 8: "#c0c0c0", 9: "#ffffff",
    10: "#64c8ff", 11: "#ff96ff",
}


def stretch(band: np.ndarray, lo: float = 2, hi: float = 98) -> np.ndarray:
    """Percentile contrast stretch to [0, 1], ignoring NaN."""
    finite = band[np.isfinite(band)]
    p_lo, p_hi = np.percentile(finite, [lo, hi])
    return np.clip((band - p_lo) / (p_hi - p_lo), 0, 1)


def load(meta_path: Path):
    meta = json.loads(meta_path.read_text())
    src = rasterio.open(meta_path.parent / meta["files"]["stack"])
    names = list(src.descriptions)
    return meta, src, names


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--meta", type=Path, default=PROCESSED / "latest_ingest.json")
    p.add_argument("--out-dir", type=Path, default=FIGURES)
    args = p.parse_args(argv)

    meta, src, names = load(args.meta)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(meta["files"]["stack"]).stem.replace("_stack", "")
    title = (
        f"{meta['stac']['item_id']}  |  {meta['stac']['datetime'][:10]}"
        f"  |  EPSG:{meta['grid']['epsg']}"
    )

    band = {n: src.read(i + 1) for i, n in enumerate(names)}
    scl = rasterio.open(args.meta.parent / meta["files"]["scl"]).read(1)

    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    fig.suptitle(title, fontsize=13)

    # True colour: what a person would see.
    rgb = np.dstack([stretch(band["B04"]), stretch(band["B03"]), stretch(band["B02"])])
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("True colour (B04/B03/B02)")

    # False colour: vegetation red, water black, built-up cyan-grey.
    fcc = np.dstack([stretch(band["B08"]), stretch(band["B04"]), stretch(band["B03"])])
    axes[0, 1].imshow(fcc)
    axes[0, 1].set_title("False colour infrared (B08/B04/B03)")

    im = axes[1, 0].imshow(band["NDVI"], cmap="RdYlGn", vmin=-0.2, vmax=1.0)
    axes[1, 0].set_title("NDVI")
    fig.colorbar(im, ax=axes[1, 0], fraction=0.046)

    present = sorted(int(c) for c in np.unique(scl))
    lut = np.zeros(max(present) + 1, dtype=int)
    for i, c in enumerate(present):
        lut[c] = i
    cmap = ListedColormap([SCL_COLOURS[c] for c in present])
    axes[1, 1].imshow(lut[scl], cmap=cmap, vmin=0, vmax=len(present) - 1)
    axes[1, 1].set_title("Scene Classification Layer (SCL)")
    handles = [
        plt.Rectangle((0, 0), 1, 1, fc=SCL_COLOURS[c], ec="k", lw=0.4) for c in present
    ]
    axes[1, 1].legend(
        handles,
        [f"{c} {SCL_CLASSES[c]}" for c in present],
        loc="lower right", fontsize=7, framealpha=0.9,
    )

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    out = args.out_dir / f"{stem}_quicklook.jpg"
    fig.tight_layout()
    # JPEG, not PNG: this is a photographic quicklook, and lossless encoding of
    # 4 x 2100 px of satellite imagery costs ~5 MB in a repo meant to be cloned.
    fig.savefig(out, dpi=100, bbox_inches="tight", pil_kwargs={"quality": 88})
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
