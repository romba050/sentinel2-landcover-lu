"""Run the mean-field CRF over the OOF probabilities and evaluate it honestly.

Prerequisite: ``uv run python -m src.models.oof`` (the out-of-fold probability
raster -- see that module for why in-sample probabilities would rig the
comparison in the CRF's favour).

What "honestly" means here, concretely
--------------------------------------
theta (the pairwise strength) is deliberately NOT tuned against overall
accuracy: CORINE's 25 ha polygons reward agreeing with big blobs, so accuracy
tuning selects over-smoothing. The contrast-ablation column of the report
demonstrates that failure mode live -- remove the contrast term and every
CORINE-derived number *improves* while the road network is erased. Instead the
sweep reports every instrument at every theta, and the chosen theta is picked
by an explicit, non-CORINE, two-part rule:

    keep thetas whose map retains >= THIN_RETENTION of the unary map's THIN
    artificial pixels (a 3x3-opening residue: roads, railways, settlement
    fingers), then among those take the theta that minimises the area lost to
    the >= 2000 m2 speckle filter at the vectorisation stage.

Thin structures, not class area, because uniform smoothing *grows* the
artificial class (urban interiors consolidate) at the same time as it erases
roads -- an area test is blind to the destruction. Roads are also the closest
visual analogue to the retinal vasculature the thesis method was built on.

Instruments, per theta:
1. accuracy stratified by distance to the CORINE boundary (the whole curve --
   spatial context should help most deep inside polygons, where the label is
   trustworthy and speckle is pure error);
2. fragmentation: patch count, mean patch size, edge density, share of pixels
   in sub-5-px specks;
3. the downstream area effect: how much area the >= 2000 m2 vectorisation
   filter no longer has to throw away before the PostGIS stage;
4. thin-structure (road) survival, as above.

    uv run python -m src.postprocessing.crf_pipeline
    uv run python -m src.postprocessing.crf_pipeline --thetas 0.3 0.8 2.0 5.0
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio

from src.evaluation import metrics as M
from src.postprocessing import crf
from src.postprocessing.to_postgis import MIN_POLYGON_AREA_M2, vectorise_array

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS = PROJECT_ROOT / "data" / "results"

#: Feature bands driving the contrast term. SWIR + NBR2 carry the class signal
#: in this scene (see the feature-importance figure); NDVI does not.
CONTRAST_BANDS = ("B11", "B12", "NBR2")
#: The sweep must bracket the over-smoothing regime: the top value is meant to
#: fail the road-retention rule so that the chosen theta is a real interior
#: optimum, not just "the largest value tried".
DEFAULT_THETAS = (0.3, 0.8, 2.0, 5.0, 12.0)
DEFAULT_ITERS = 8
#: Road-survival rule: chosen theta must keep at least this fraction of the
#: unary map's THIN artificial pixels (those removed by a 3x3 morphological
#: opening -- structures under ~3 px across: roads, railways, thin settlement
#: fingers). Total class area is deliberately NOT the criterion: uniform
#: smoothing *grows* artificial area by consolidating urban interiors while it
#: erases the road network, so an area test cannot see the destruction.
THIN_RETENTION = 0.75
BUILT_ID = 1


def load_inputs(processed_dir: Path):
    oof = json.loads((processed_dir / "latest_oof.json").read_text())
    ingest = json.loads((processed_dir / "latest_ingest.json").read_text())
    labels_meta = json.loads((processed_dir / "latest_labels.json").read_text())

    with rasterio.open(processed_dir / oof["proba_raster"]) as src:
        proba = src.read().astype(np.float32)
        transform, crs = src.transform, src.crs
    with rasterio.open(processed_dir / ingest["files"]["stack"]) as src:
        names = list(src.descriptions)
        features = np.stack([src.read(names.index(b) + 1) for b in CONTRAST_BANDS])
    with rasterio.open(processed_dir / ingest["files"]["valid"]) as src:
        valid = src.read(1).astype(bool)
    with rasterio.open(processed_dir / labels_meta["files"]["labels"]) as src:
        labels = src.read(1)
    with rasterio.open(processed_dir / labels_meta["files"]["boundary_distance"]) as src:
        boundary = src.read(1)

    return oof, labels_meta, proba, features, valid, labels, boundary, transform, crs


def thin_structure_mask(pred: np.ndarray, class_id: int = BUILT_ID) -> np.ndarray:
    """Pixels of ``class_id`` sitting in structures thinner than ~3 px.

    A 3x3 binary opening removes anything that cannot contain a 3x3 square --
    which is precisely the roads, railways and settlement fingers at 10 m.
    What the opening removes is the thin-structure mask.
    """
    from scipy import ndimage

    mask = pred == class_id
    return mask & ~ndimage.binary_opening(mask, structure=np.ones((3, 3), dtype=bool))


def evaluate_map(
    pred: np.ndarray,
    labels: np.ndarray,
    valid: np.ndarray,
    boundary: np.ndarray,
    class_ids: list[int],
    class_names: list[str],
    transform,
    crs,
    thin_mask: np.ndarray | None = None,
) -> dict:
    """All instruments for one candidate map. Test set = every labelled valid
    pixel; with OOF probabilities they are all out-of-sample."""
    eligible = valid & (labels != 0)
    y_true, y_pred = labels[eligible], pred[eligible]
    report = M.evaluate(y_true, y_pred, class_ids, class_names)

    gdf = vectorise_array(
        pred, transform, crs, dict(zip(class_ids, class_names, strict=True)),
        MIN_POLYGON_AREA_M2,
    )
    built_km2 = float((pred == BUILT_ID).sum()) * abs(transform.a * transform.e) / 1e6

    return {
        "overall_accuracy": report.overall_accuracy,
        "macro_f1": report.macro_f1,
        "kappa": report.kappa,
        "per_class": report.per_class,
        "accuracy_by_boundary_distance": M.accuracy_by_boundary_distance(
            y_true, y_pred, boundary[eligible]
        ),
        "fragmentation": M.fragmentation(np.where(valid, pred, 0)),
        "vectorisation": {
            "n_polygons": int(len(gdf)),
            "dropped_speck_area_km2": gdf.attrs["dropped_speck_area_km2"],
            "dropped_speck_count": gdf.attrs["dropped_speck_count"],
        },
        "artificial_km2": built_km2,
        "thin_artificial_retention": (
            float((pred[thin_mask] == BUILT_ID).mean()) if thin_mask is not None else None
        ),
    }


def _write_map(path: Path, data: np.ndarray, transform, crs):
    with rasterio.open(
        path, "w", driver="GTiff", height=data.shape[0], width=data.shape[1],
        count=1, dtype="uint8", crs=crs, transform=transform, nodata=0,
        tiled=True, blockxsize=512, blockysize=512, compress="deflate", predictor=2,
    ) as dst:
        dst.write(data, 1)
        dst.set_band_description(1, "class_id")
    log.info("wrote %s", path.name)


def run(
    processed_dir: Path = PROCESSED,
    results_dir: Path = RESULTS,
    thetas: tuple[float, ...] = DEFAULT_THETAS,
    n_iters: int = DEFAULT_ITERS,
) -> dict:
    (oof, labels_meta, proba, features, valid, labels, boundary,
     transform, crs) = load_inputs(processed_dir)
    class_ids, class_names = oof["class_ids"], oof["class_names"]
    stem = oof["proba_raster"].replace("_rf_proba_oof.tif", "")

    feats = crf.standardise(features, valid)
    sigma = crf.estimate_sigma(feats, valid)
    log.info("contrast features %s, sigma = %.3f (median 4-neighbour distance)",
             CONTRAST_BANDS, sigma)
    kernels = crf.contrast_kernels(feats, valid, sigma)

    unary_map = crf.map_labels(proba, class_ids, valid)
    thin_mask = thin_structure_mask(unary_map)
    log.info("evaluating unary (theta = 0) baseline; %s thin artificial px "
             "(roads/rails/settlement fingers) tracked for survival",
             f"{thin_mask.sum():,}")
    results = {"unary": evaluate_map(unary_map, labels, valid, boundary,
                                     class_ids, class_names, transform, crs,
                                     thin_mask)}

    maps = {"unary": unary_map}
    for theta in thetas:
        t0 = time.perf_counter()
        Q, deltas = crf.mean_field(proba, kernels, theta, valid, n_iters)
        crf_map = crf.map_labels(Q, class_ids, valid)
        maps[theta] = crf_map
        log.info("theta = %.2f: %d iterations in %.0f s, mean |dQ| %s",
                 theta, n_iters, time.perf_counter() - t0,
                 " -> ".join(f"{d:.4f}" for d in deltas))
        res = evaluate_map(crf_map, labels, valid, boundary,
                           class_ids, class_names, transform, crs, thin_mask)
        res["convergence_mean_abs_dQ"] = deltas
        results[theta] = res

    # Selection rule -- explicit and non-CORINE, see docstring. Thin-structure
    # survival is the constraint; among the road-safe thetas, pick the one
    # that minimises the area the vectorisation filter throws away (the
    # downstream instrument). Neither criterion consults CORINE agreement:
    # accuracy tuning would select the bulldozer, because CORINE's 25 ha
    # polygons reward exactly the blobs that over-smoothing produces -- the
    # ablation column of the report demonstrates that failure mode live.
    surviving = [t for t in thetas
                 if results[t]["thin_artificial_retention"] >= THIN_RETENTION]
    pool = surviving or [min(thetas)]
    chosen = min(pool, key=lambda t: (
        results[t]["vectorisation"]["dropped_speck_area_km2"], t
    ))
    log.info(
        "chosen theta = %g (thin-structure retention >= %.0f%% held by %s; "
        "minimum speckle-area loss among them)",
        chosen, 100 * THIN_RETENTION, [f"{t:g}" for t in surviving],
    )

    # The contrast-ablation counterfactual at the chosen theta: identical
    # smoothing strength, no ability to tell a real edge from noise.
    log.info("running contrast ablation (uniform kernels) at theta = %g", chosen)
    Q_abl, _ = crf.mean_field(proba, crf.uniform_kernels(valid), chosen, valid, n_iters)
    ablation_map = crf.map_labels(Q_abl, class_ids, valid)
    maps["ablation"] = ablation_map
    results["ablation"] = evaluate_map(ablation_map, labels, valid, boundary,
                                       class_ids, class_names, transform, crs,
                                       thin_mask)

    results_dir.mkdir(parents=True, exist_ok=True)
    unary_path = results_dir / f"{stem}_unary_oof.tif"
    crf_path = results_dir / f"{stem}_crf_labels.tif"
    _write_map(unary_path, unary_map, transform, crs)
    _write_map(crf_path, maps[chosen], transform, crs)
    sweep_files = {}
    for theta in thetas:
        p = results_dir / f"{stem}_crf_theta_{theta:g}.tif"
        _write_map(p, maps[theta], transform, crs)
        sweep_files[f"{theta:g}"] = p.name
    ablation_path = results_dir / f"{stem}_crf_ablation.tif"
    _write_map(ablation_path, maps["ablation"], transform, crs)

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "method": {
            "unary": "RandomForest predict_proba, out-of-fold (see latest_oof.json)",
            "pairwise": "contrast-sensitive Potts, 8-neighbour grid",
            "contrast_bands": list(CONTRAST_BANDS),
            "sigma": sigma,
            "n_iters": n_iters,
            "theta_sweep": list(thetas),
            "theta_chosen": chosen,
            "selection_rule": (
                f"road-safe thetas (>= {THIN_RETENTION:.0%} of the unary map's "
                f"thin artificial pixels surviving) ranked by minimum "
                f"speckle-area loss at the vectorisation stage; NOT accuracy-tuned"
            ),
            "ablation": "uniform kernels (contrast term removed) at chosen theta",
        },
        "results": {
            (k if isinstance(k, str) else f"{k:g}"): v for k, v in results.items()
        },
        "files": {"unary": unary_path.name, "crf": crf_path.name,
                  "sweep": sweep_files, "ablation": ablation_path.name},
        "oof": oof,
    }
    (results_dir / f"{stem}_crf_summary.json").write_text(json.dumps(summary, indent=2))
    (results_dir / "latest_crf.json").write_text(json.dumps(summary, indent=2))
    log.info("wrote %s", f"{stem}_crf_summary.json")
    return summary


def print_report(summary: dict) -> None:
    res = summary["results"]
    thetas = [k for k in res if k not in ("unary", "ablation")]
    cols = ["unary"] + thetas + (["ablation"] if "ablation" in res else [])

    def row(label, fn, fmt):
        cells = "".join(f"{fmt.format(fn(res[c])):>12}" for c in cols)
        print(f"  {label:<34}{cells}")

    def col_name(c):
        if c == "unary":
            return "unary"
        if c == "ablation":
            return "no-contrast"
        return "θ=" + c

    header = "".join(f"{col_name(c):>12}" for c in cols)
    print(f"\n=== CRF sweep (chosen θ = {summary['method']['theta_chosen']}) ===")
    print(f"  {'':<34}{header}")
    row("overall accuracy (all OOF px)", lambda r: r["overall_accuracy"], "{:.4f}")
    row("macro F1", lambda r: r["macro_f1"], "{:.4f}")
    row("accuracy > 400 m from boundary",
        lambda r: r["accuracy_by_boundary_distance"][-1]["accuracy"], "{:.4f}")
    row("accuracy 0-20 m from boundary",
        lambda r: r["accuracy_by_boundary_distance"][0]["accuracy"], "{:.4f}")
    row("patches", lambda r: r["fragmentation"]["n_patches"], "{:,}")
    row("mean patch (ha)", lambda r: r["fragmentation"]["mean_patch_ha"], "{:.2f}")
    row("edge density (km/km²)",
        lambda r: r["fragmentation"]["edge_density_km_per_km2"], "{:.2f}")
    row("pixels in sub-5px specks",
        lambda r: r["fragmentation"]["tiny_patch_pixel_share"], "{:.3%}")
    row("polygons after 2000 m² filter",
        lambda r: r["vectorisation"]["n_polygons"], "{:,}")
    row("area lost to speckle (km²)",
        lambda r: r["vectorisation"]["dropped_speck_area_km2"], "{:.2f}")
    row("artificial surfaces (km²)", lambda r: r["artificial_km2"], "{:.2f}")
    row("thin-structure (road) survival",
        lambda r: r["thin_artificial_retention"], "{:.1%}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--processed-dir", type=Path, default=PROCESSED)
    p.add_argument("--results-dir", type=Path, default=RESULTS)
    p.add_argument("--thetas", type=float, nargs="+", default=list(DEFAULT_THETAS))
    p.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S",
    )
    logging.getLogger("rasterio").setLevel(logging.WARNING)

    summary = run(
        processed_dir=args.processed_dir,
        results_dir=args.results_dir,
        thetas=tuple(args.thetas),
        n_iters=args.iters,
    )
    print_report(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
