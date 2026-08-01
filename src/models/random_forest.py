"""Random Forest land-cover baseline with spatially-blocked validation.

This is deliberately a *per-pixel spectral* classifier: each pixel is described
only by its own 10 reflectance bands and 5 spectral indices, with no
neighbourhood context whatsoever. That is the honest baseline for the question
"how much land cover can you read straight out of the spectrum?", and it leaves
a clean gap for a convolutional model to close by adding spatial context.

The headline experiment is the split comparison. The same model, the same
features and the same hyperparameters are evaluated twice: once with a random
pixel split and once with a spatially-blocked split. The difference between the
two numbers is the amount by which the conventional random split flatters the
model, and it is the single most useful thing this script reports.

    uv run python -m src.models.random_forest
    uv run python -m src.models.random_forest --block-size 4000 --trees 500
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import rasterio
from sklearn.ensemble import RandomForestClassifier

from src.evaluation import metrics as M
from src.evaluation.spatial_split import (
    DEFAULT_BLOCK_SIZE_M,
    Split,
    class_coverage,
    random_pixel_split,
    spatial_block_split,
    subsample,
)

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROCESSED = PROJECT_ROOT / "data" / "processed"
DEFAULT_RESULTS = PROJECT_ROOT / "data" / "results"
DEFAULT_MODELS = PROJECT_ROOT / "models"

#: Cap on training pixels. The AOI holds ~4.4 M labelled pixels; a Random
#: Forest saturates long before that, and the extra samples are mostly
#: near-duplicates of their neighbours anyway.
DEFAULT_MAX_TRAIN_PIXELS = 300_000
DEFAULT_TREES = 300
#: Minimum samples per leaf. Not cosmetic: at min_samples_leaf=5 a 300-tree
#: forest over 300 k pixels grows ~36 M nodes and serialises to 1.7 GB, because
#: every node carries a per-class probability vector. Twenty costs almost no
#: accuracy on spatially-correlated data and shrinks the model roughly 4x.
DEFAULT_MIN_SAMPLES_LEAF = 20
#: Prediction is chunked so a full-scene inference never materialises more than
#: this many rows of float32 features at once.
PREDICT_CHUNK = 500_000
#: Training-set sizes for the leakage sweep. Leakage from a random split is not
#: a fixed quantity: it grows with how densely the training pixels sample the
#: scene, because a denser sample puts a training neighbour ever closer to each
#: test pixel. A single number would misrepresent that.
LEAKAGE_SWEEP_SIZES = (25_000, 100_000, 400_000, 1_600_000)
LEAKAGE_SWEEP_TREES = 100


@dataclass
class Dataset:
    features: np.ndarray  # (n_bands, H, W) float32
    labels: np.ndarray  # (H, W) uint8, 0 = nodata
    valid: np.ndarray  # (H, W) bool, cloud-free
    boundary_distance: np.ndarray  # (H, W) float32, metres
    feature_names: list[str]
    class_ids: list[int]
    class_names: list[str]
    transform: rasterio.Affine
    crs: rasterio.crs.CRS
    stem: str

    @property
    def eligible(self) -> np.ndarray:
        """Pixels usable for training or testing: cloud-free and labelled."""
        return self.valid & (self.labels != 0)


def load_dataset(processed_dir: Path = DEFAULT_PROCESSED) -> Dataset:
    ingest = json.loads((processed_dir / "latest_ingest.json").read_text())
    labels_meta = json.loads((processed_dir / "latest_labels.json").read_text())

    with rasterio.open(processed_dir / ingest["files"]["stack"]) as src:
        features = src.read().astype(np.float32)
        feature_names = list(src.descriptions)
        transform, crs = src.transform, src.crs
    with rasterio.open(processed_dir / ingest["files"]["valid"]) as src:
        valid = src.read(1).astype(bool)
    with rasterio.open(processed_dir / labels_meta["files"]["labels"]) as src:
        labels = src.read(1)
    with rasterio.open(processed_dir / labels_meta["files"]["boundary_distance"]) as src:
        boundary_distance = src.read(1)

    kept = [c for c in labels_meta["class_scheme"]["classes"] if c["kept"]]
    ds = Dataset(
        features=features,
        labels=labels,
        valid=valid,
        boundary_distance=boundary_distance,
        feature_names=feature_names,
        class_ids=[c["class_id"] for c in kept],
        class_names=[c["name"] for c in kept],
        transform=transform,
        crs=crs,
        stem=Path(ingest["files"]["stack"]).stem.replace("_stack", ""),
    )
    log.info(
        "Loaded %d features over %dx%d px; %d eligible pixels across %d classes",
        len(feature_names), labels.shape[1], labels.shape[0],
        ds.eligible.sum(), len(kept),
    )
    return ds


def matrix(features: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Extract an (n_pixels, n_features) design matrix for the masked pixels."""
    return features[:, mask].T


def train_and_evaluate(
    ds: Dataset,
    split: Split,
    n_trees: int,
    max_train_pixels: int | None,
    seed: int,
    n_jobs: int = -1,
    min_samples_leaf: int = DEFAULT_MIN_SAMPLES_LEAF,
    class_weight: str | None = None,
) -> tuple[RandomForestClassifier, M.ClassificationReport, dict]:
    train_mask = subsample(split.train, max_train_pixels, seed=seed)
    x_train = matrix(ds.features, train_mask)
    y_train = ds.labels[train_mask]

    log.info("Training RF: %d trees on %s samples x %d features",
             n_trees, f"{x_train.shape[0]:,}", x_train.shape[1])
    started = time.perf_counter()
    clf = RandomForestClassifier(
        n_estimators=n_trees,
        # Depth is left unbounded but leaves are floored: with millions of
        # correlated pixels an unconstrained forest will happily grow leaves
        # holding a single pixel and memorise noise.
        min_samples_leaf=min_samples_leaf,
        max_features="sqrt",
        n_jobs=n_jobs,
        random_state=seed,
        # Priors are left alone by default. Balancing makes the rare forest
        # classes look better but distorts predicted *areas*, and area per
        # class is what the PostGIS stage actually reports. The effect is
        # measured explicitly rather than assumed -- see the balanced variant.
        class_weight=class_weight,
    )
    clf.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - started
    log.info("  fitted in %.1f s", fit_seconds)

    x_test = matrix(ds.features, split.test)
    y_test = ds.labels[split.test]
    y_pred = clf.predict(x_test)

    report = M.evaluate(y_test, y_pred, ds.class_ids, ds.class_names)
    report.extra = {
        "split": {"kind": split.kind, **split.detail},
        "n_train_pixels": int(train_mask.sum()),
        "fit_seconds": fit_seconds,
        "class_weight": class_weight,
    }

    boundary = M.accuracy_by_boundary_distance(
        y_test, y_pred, ds.boundary_distance[split.test]
    )
    return clf, report, {"accuracy_by_boundary_distance": boundary}


def leakage_sweep(
    ds: Dataset,
    block_size_m: float,
    test_fraction: float,
    seed: int,
    sizes: tuple[int, ...] = LEAKAGE_SWEEP_SIZES,
    n_trees: int = LEAKAGE_SWEEP_TREES,
) -> list[dict]:
    """Measure how random-split optimism grows with training density.

    Quoting one leakage figure is misleading, because the figure depends
    entirely on how densely you sample. Sparse training pixels sit far from the
    test pixels even under a random split, so little leaks; sample densely and
    almost every test pixel gains an immediate neighbour in training, and the
    random split's score detaches from reality. The sweep shows the trend
    instead of a single point on it.
    """
    eligible = ds.eligible
    spatial = spatial_block_split(eligible, ds.transform, test_fraction, block_size_m, seed)
    random_ = random_pixel_split(eligible, test_fraction, seed)

    rows = []
    for n in sizes:
        if n > spatial.train.sum():
            log.info("Skipping density %s: more than the %s available training pixels",
                     f"{n:,}", f"{spatial.train.sum():,}")
            continue
        _, sp_report, _ = train_and_evaluate(ds, spatial, n_trees, n, seed)
        _, rd_report, _ = train_and_evaluate(ds, random_, n_trees, n, seed)
        row = {
            "n_train_pixels": n,
            "train_density": n / float(eligible.sum()),
            "spatial_accuracy": sp_report.overall_accuracy,
            "random_accuracy": rd_report.overall_accuracy,
            "accuracy_inflation": rd_report.overall_accuracy - sp_report.overall_accuracy,
            "spatial_macro_f1": sp_report.macro_f1,
            "random_macro_f1": rd_report.macro_f1,
            "macro_f1_inflation": rd_report.macro_f1 - sp_report.macro_f1,
        }
        rows.append(row)
        log.info(
            "  density %9s (%.1f%% of AOI): blocked=%.3f random=%.3f inflation=%+.3f",
            f"{n:,}", 100 * row["train_density"],
            row["spatial_accuracy"], row["random_accuracy"], row["accuracy_inflation"],
        )
    return rows


def predict_raster(clf: RandomForestClassifier, ds: Dataset) -> np.ndarray:
    """Classify every cloud-free pixel; masked pixels stay 0."""
    out = np.zeros(ds.labels.shape, dtype=np.uint8)
    idx = np.flatnonzero(ds.valid.ravel())
    flat_features = ds.features.reshape(ds.features.shape[0], -1)
    log.info("Predicting %s pixels", f"{idx.size:,}")

    predictions = np.empty(idx.size, dtype=np.uint8)
    for start in range(0, idx.size, PREDICT_CHUNK):
        chunk = idx[start : start + PREDICT_CHUNK]
        predictions[start : start + chunk.size] = clf.predict(flat_features[:, chunk].T)
    out.ravel()[idx] = predictions
    return out


def feature_importance(clf: RandomForestClassifier, names: list[str]) -> list[dict]:
    order = np.argsort(clf.feature_importances_)[::-1]
    return [
        {"feature": names[i], "importance": float(clf.feature_importances_[i])}
        for i in order
    ]


def run(
    processed_dir: Path = DEFAULT_PROCESSED,
    results_dir: Path = DEFAULT_RESULTS,
    models_dir: Path = DEFAULT_MODELS,
    n_trees: int = DEFAULT_TREES,
    block_size_m: float = DEFAULT_BLOCK_SIZE_M,
    test_fraction: float = 0.3,
    max_train_pixels: int | None = DEFAULT_MAX_TRAIN_PIXELS,
    seed: int = 42,
    run_sweep: bool = True,
) -> dict:
    ds = load_dataset(processed_dir)
    eligible = ds.eligible

    log.info("--- Spatially-blocked split (the honest one) ---")
    spatial = spatial_block_split(eligible, ds.transform, test_fraction, block_size_m, seed)
    coverage = class_coverage(ds.labels, spatial)
    clf, spatial_report, spatial_extra = train_and_evaluate(
        ds, spatial, n_trees, max_train_pixels, seed
    )
    print("\n=== Spatially-blocked split ===")
    print(spatial_report.text())

    log.info("--- Random pixel split (baseline, expected to be optimistic) ---")
    random_split = random_pixel_split(eligible, test_fraction, seed)
    _, random_report, _ = train_and_evaluate(
        ds, random_split, n_trees, max_train_pixels, seed
    )
    print("\n=== Random pixel split (leaky) ===")
    print(random_report.text())

    comparison = M.compare_splits(spatial_report, random_report)
    inflation = comparison["inflation"]
    print(
        f"\nAt {max_train_pixels:,} training pixels a random split overstates "
        f"overall accuracy by {inflation['overall_accuracy']:+.1%} and macro-F1 "
        f"by {inflation['macro_f1']:+.3f}."
    )

    log.info("--- Class-balance sensitivity ---")
    _, balanced_report, _ = train_and_evaluate(
        ds, spatial, n_trees, max_train_pixels, seed, class_weight="balanced"
    )
    print("\n=== Spatially-blocked split, class_weight='balanced' ===")
    print(balanced_report.text())
    print(
        "  Balancing trades overall accuracy "
        f"({spatial_report.overall_accuracy:.3f} -> {balanced_report.overall_accuracy:.3f}) "
        f"for macro-F1 ({spatial_report.macro_f1:.3f} -> {balanced_report.macro_f1:.3f}). "
        "The unbalanced model is kept because predicted class *areas* -- what "
        "the PostGIS stage reports -- stay closer to the truth."
    )

    sweep: list[dict] = []
    if run_sweep:
        log.info("--- Leakage vs training density ---")
        sweep = leakage_sweep(ds, block_size_m, test_fraction, seed)
        print("\n=== How much a random split inflates accuracy, by training density ===")
        print(f"  {'train px':>10} {'% of AOI':>9} {'blocked':>9} {'random':>8} {'inflation':>10}")
        for row in sweep:
            print(f"  {row['n_train_pixels']:>10,} {row['train_density']:>8.1%} "
                  f"{row['spatial_accuracy']:>9.3f} {row['random_accuracy']:>8.3f} "
                  f"{row['accuracy_inflation']:>+10.3f}")

    print("\n=== Accuracy vs distance to nearest CORINE boundary (blocked split) ===")
    for row in spatial_extra["accuracy_by_boundary_distance"]:
        hi, lo = row["max_distance_m"], row["min_distance_m"]
        band = f"{lo:.0f}-{hi:.0f} m" if hi else f">{lo:.0f} m"
        print(f"  {band:>12}  n={row['n_pixels']:>9,}  accuracy={row['accuracy']:.3f}")

    importance = feature_importance(clf, ds.feature_names)
    print("\n=== Feature importance (top 8) ===")
    for row in importance[:8]:
        print(f"  {row['feature']:<6} {row['importance']:.4f}")

    # Full-scene prediction from the spatially-blocked model.
    prediction = predict_raster(clf, ds)
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    pred_path = results_dir / f"{ds.stem}_rf_prediction.tif"
    with rasterio.open(
        pred_path, "w", driver="GTiff", height=ds.labels.shape[0],
        width=ds.labels.shape[1], count=1, dtype="uint8", crs=ds.crs,
        transform=ds.transform, nodata=0, tiled=True, blockxsize=512,
        blockysize=512, compress="deflate", predictor=2,
    ) as dst:
        dst.write(prediction, 1)
        dst.set_band_description(1, "class_id")
    log.info("wrote %s", pred_path.name)

    model_path = models_dir / f"{ds.stem}_rf.joblib"
    joblib.dump(
        {"model": clf, "feature_names": ds.feature_names,
         "class_ids": ds.class_ids, "class_names": ds.class_names},
        model_path,
        compress=3,
    )
    log.info("wrote %s (%.1f MB)", model_path.name, model_path.stat().st_size / 1e6)

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": {
            "type": "RandomForestClassifier",
            "n_estimators": n_trees,
            "min_samples_leaf": DEFAULT_MIN_SAMPLES_LEAF,
            "max_features": "sqrt",
            "class_weight": None,
            "features": ds.feature_names,
            "context": "none (per-pixel spectral only)",
            "seed": seed,
        },
        "spatial_block_split": spatial_report.to_dict(),
        "random_pixel_split": random_report.to_dict(),
        "balanced_class_weight_variant": balanced_report.to_dict(),
        "split_comparison": comparison,
        "leakage_vs_training_density": sweep,
        "class_coverage": {str(k): v for k, v in coverage.items()},
        "feature_importance": importance,
        **spatial_extra,
        "files": {
            "prediction": pred_path.name,
            "model": str(model_path.relative_to(PROJECT_ROOT)),
        },
    }
    summary_path = results_dir / f"{ds.stem}_rf_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    (results_dir / "latest_rf.json").write_text(json.dumps(summary, indent=2))
    log.info("wrote %s", summary_path.name)
    return summary


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED)
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    p.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS)
    p.add_argument("--trees", type=int, default=DEFAULT_TREES)
    p.add_argument("--block-size", type=float, default=DEFAULT_BLOCK_SIZE_M,
                   help="Spatial block edge length in metres.")
    p.add_argument("--test-fraction", type=float, default=0.3)
    p.add_argument("--max-train-pixels", type=int, default=DEFAULT_MAX_TRAIN_PIXELS)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-sweep", action="store_true",
                   help="Skip the leakage-vs-training-density sweep (saves ~4 min).")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S",
    )
    logging.getLogger("rasterio").setLevel(logging.WARNING)

    run(
        processed_dir=args.processed_dir,
        results_dir=args.results_dir,
        models_dir=args.models_dir,
        n_trees=args.trees,
        block_size_m=args.block_size,
        test_fraction=args.test_fraction,
        max_train_pixels=args.max_train_pixels,
        seed=args.seed,
        run_sweep=not args.no_sweep,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
