"""Out-of-fold class probabilities for the whole scene.

Why this exists
---------------
The CRF stage (``src/postprocessing/crf.py``) smooths each pixel toward its
neighbours' class probabilities. If those probabilities came from the model
that *trained* on the surrounding block, they are in-sample -- optimistically
sharp and optimistically correct -- and the CRF would appear to help more than
it does, because test pixels near a block edge get pulled toward fitted values.

The fix is out-of-fold probabilities everywhere: a spatially-blocked K-fold in
which each fold's pixels are predicted by a model that never saw that fold's
blocks. The assembled raster has no in-sample optimism anywhere, so the CRF
runs over the scene exactly as it would at real inference time -- and, as a
bonus, *every* labelled pixel becomes a fair test pixel for the unary-vs-CRF
comparison.

The RF hyperparameters are copied from the baseline on purpose. The CRF must
be a pure post-processing stage over the identical model family, or nothing in
the before/after comparison means anything.

    uv run python -m src.models.oof
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from sklearn.ensemble import RandomForestClassifier

from src.evaluation.spatial_split import DEFAULT_BLOCK_SIZE_M, spatial_block_folds, subsample
from src.models.random_forest import (
    DEFAULT_MAX_TRAIN_PIXELS,
    DEFAULT_MIN_SAMPLES_LEAF,
    DEFAULT_PROCESSED,
    DEFAULT_TREES,
    PREDICT_CHUNK,
    Dataset,
    load_dataset,
    matrix,
)

log = logging.getLogger(__name__)


def oof_probabilities(
    ds: Dataset,
    n_folds: int = 5,
    n_trees: int = DEFAULT_TREES,
    block_size_m: float = DEFAULT_BLOCK_SIZE_M,
    max_train_pixels: int | None = DEFAULT_MAX_TRAIN_PIXELS,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (probabilities (C,H,W) float32, fold raster (H,W) int8).

    Every *valid* pixel is predicted by the model of its own spatial fold,
    trained only on eligible pixels of the other folds.
    """
    folds = spatial_block_folds(
        ds.eligible, ds.transform, n_folds, block_size_m, seed, assign=ds.valid
    )
    n_classes = len(ds.class_ids)
    proba = np.zeros((n_classes, *ds.labels.shape), dtype=np.float32)
    flat_features = ds.features.reshape(ds.features.shape[0], -1)

    for f in range(n_folds):
        train = subsample(ds.eligible & (folds != f) & (folds >= 0),
                          max_train_pixels, seed=seed + f)
        clf = RandomForestClassifier(
            n_estimators=n_trees,
            min_samples_leaf=DEFAULT_MIN_SAMPLES_LEAF,
            max_features="sqrt",
            n_jobs=-1,
            random_state=seed,
            class_weight=None,
        )
        t0 = time.perf_counter()
        clf.fit(matrix(ds.features, train), ds.labels[train])

        # Every fold's training half must contain every class, or the proba
        # bands would mean different classes in different regions of the map.
        if list(clf.classes_) != ds.class_ids:
            raise RuntimeError(
                f"fold {f}: training data lost classes "
                f"{set(ds.class_ids) - set(clf.classes_)}; use a smaller block size"
            )

        target = np.flatnonzero((ds.valid & (folds == f)).ravel())
        for start in range(0, target.size, PREDICT_CHUNK):
            chunk = target[start : start + PREDICT_CHUNK]
            p = clf.predict_proba(flat_features[:, chunk].T).astype(np.float32)
            proba.reshape(n_classes, -1)[:, chunk] = p.T
        log.info(
            "fold %d/%d: %s train px -> %s predicted px in %.0f s",
            f + 1, n_folds, f"{train.sum():,}", f"{target.size:,}",
            time.perf_counter() - t0,
        )

    return proba, folds


def run(
    processed_dir: Path = DEFAULT_PROCESSED,
    n_folds: int = 5,
    n_trees: int = DEFAULT_TREES,
    block_size_m: float = DEFAULT_BLOCK_SIZE_M,
    max_train_pixels: int | None = DEFAULT_MAX_TRAIN_PIXELS,
    seed: int = 42,
) -> dict:
    ds = load_dataset(processed_dir)
    proba, folds = oof_probabilities(
        ds, n_folds, n_trees, block_size_m, max_train_pixels, seed
    )

    # Sanity check the assembly: the OOF argmax accuracy over all labelled
    # pixels should sit close to the single blocked-split figure (~0.62).
    # A number near the random-split figure would mean leakage crept in.
    unary = np.zeros(ds.labels.shape, dtype=np.uint8)
    unary[ds.valid] = np.array(ds.class_ids, dtype=np.uint8)[
        np.argmax(proba[:, ds.valid], axis=0)
    ]
    acc = float((unary[ds.eligible] == ds.labels[ds.eligible]).mean())
    log.info("OOF unary accuracy over all %s labelled pixels: %.4f",
             f"{ds.eligible.sum():,}", acc)

    proba_path = processed_dir / f"{ds.stem}_rf_proba_oof.tif"
    with rasterio.open(
        proba_path, "w", driver="GTiff", height=proba.shape[1], width=proba.shape[2],
        count=proba.shape[0], dtype="float32", crs=ds.crs, transform=ds.transform,
        nodata=None, tiled=True, blockxsize=512, blockysize=512,
        compress="deflate", predictor=3, BIGTIFF="IF_SAFER",
    ) as dst:
        dst.write(proba)
        for i, (cid, name) in enumerate(
            zip(ds.class_ids, ds.class_names, strict=True), start=1
        ):
            dst.set_band_description(i, f"p_class_{cid} ({name})")
    log.info("wrote %s (%.1f MB)", proba_path.name, proba_path.stat().st_size / 1e6)

    meta = {
        "proba_raster": proba_path.name,
        "class_ids": ds.class_ids,
        "class_names": ds.class_names,
        "n_folds": n_folds,
        "block_size_m": block_size_m,
        "n_trees": n_trees,
        "min_samples_leaf": DEFAULT_MIN_SAMPLES_LEAF,
        "max_train_pixels": max_train_pixels,
        "seed": seed,
        "oof_unary_accuracy_all_labelled": acc,
    }
    (processed_dir / "latest_oof.json").write_text(json.dumps(meta, indent=2))
    return meta


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--trees", type=int, default=DEFAULT_TREES)
    p.add_argument("--block-size", type=float, default=DEFAULT_BLOCK_SIZE_M)
    p.add_argument("--max-train-pixels", type=int, default=DEFAULT_MAX_TRAIN_PIXELS)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S",
    )
    logging.getLogger("rasterio").setLevel(logging.WARNING)

    meta = run(
        processed_dir=args.processed_dir,
        n_folds=args.folds,
        n_trees=args.trees,
        block_size_m=args.block_size,
        max_train_pixels=args.max_train_pixels,
        seed=args.seed,
    )
    print(f"\nOOF probabilities ready: {meta['proba_raster']}")
    print(f"  unary accuracy on all labelled pixels: "
          f"{meta['oof_unary_accuracy_all_labelled']:.4f} "
          f"(should sit near the blocked-split 0.624, NOT the random-split 0.647)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
