"""Export the pipeline results as a self-contained interactive web map.

Produces everything ``web/index.html`` needs into ``web/``:

``truecolor.webp``       the Sentinel-2 scene, stretched, warped to EPSG:3857
``classification.png``   the RF prediction, class colours, transparent nodata
``data.js``              bounds, class legend, headline metrics and a
                         per-commune GeoJSON, embedded as one JS constant

Everything is static -- the finished page needs no server, no database and no
external requests, so it can be dropped onto any web host (or opened by
double-click) and cannot break during a demo.

Why the rasters are re-warped to EPSG:3857
------------------------------------------
Leaflet drapes an image linearly across its corner coordinates in Web-Mercator
screen space. Our rasters are in UTM 31N, whose grid is rotated ~2.4 degrees
against the Mercator graticule at this longitude (grid convergence: the AOI
sits 3 degrees east of the zone's central meridian). Overlaying the raw UTM
pixels would therefore shift the corners by hundreds of metres and nothing
would line up with the (correctly projected) commune boundaries. Reprojection
to EPSG:3857 happens once, here, at export -- the analysis rasters stay
untouched in their native CRS.

Why the commune stats are recomputed here rather than read from PostGIS
-----------------------------------------------------------------------
So the web export works on a fresh clone without a running database. The
numbers are the same zonal statistics the SQL produces -- pixel counting on
the prediction grid is exact for this purpose -- and the page states its own
provenance in the footer.

    uv run python -m src.postprocessing.web_export
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.warp import calculate_default_transform, reproject, transform_bounds
from shapely.geometry import box

from src.ingest.preview import stretch

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS = PROJECT_ROOT / "data" / "results"
RAW = PROJECT_ROOT / "data" / "raw"
WEB = PROJECT_ROOT / "web"

WEB_CRS = "EPSG:3857"
#: Douglas-Peucker tolerance for the commune outlines, in metres. 15 m is
#: invisible at the zoom levels the page allows but cuts the GeoJSON ~5x.
SIMPLIFY_M = 15.0
#: Communes covered less than this are shown but flagged as partially imaged.
FULL_COVERAGE = 0.90
FOREST_IDS = frozenset({6, 7, 8})
BUILT_ID = 1


# --------------------------------------------------------------------------- #
# Raster exports
# --------------------------------------------------------------------------- #


def _mercator_grid(src) -> tuple:
    transform, width, height = calculate_default_transform(
        src.crs, WEB_CRS, src.width, src.height, *src.bounds
    )
    return transform, width, height


def _warp(band: np.ndarray, src, dst_shape, dst_transform, resampling, nodata):
    out = np.full(dst_shape, nodata, dtype=band.dtype)
    reproject(
        band,
        out,
        src_transform=src.transform,
        src_crs=src.crs,
        src_nodata=nodata,
        dst_transform=dst_transform,
        dst_crs=WEB_CRS,
        dst_nodata=nodata,
        resampling=resampling,
    )
    return out


def export_classification(pred_path: Path, classes: list[dict], out: Path) -> tuple:
    """Warp the class raster to Mercator and colour it; nodata -> transparent."""
    with rasterio.open(pred_path) as src:
        dst_transform, width, height = _mercator_grid(src)
        warped = _warp(
            src.read(1), src, (height, width), dst_transform, Resampling.nearest, 0
        )

    lut = np.zeros((256, 4), dtype=np.uint8)
    for c in classes:
        h = c["colour"].lstrip("#")
        lut[c["class_id"]] = [int(h[i : i + 2], 16) for i in (0, 2, 4)] + [255]
    rgba = lut[warped]

    Image.fromarray(rgba, "RGBA").save(out, optimize=True)
    log.info("wrote %s (%.2f MB, %dx%d)", out.name, out.stat().st_size / 1e6, width, height)

    merc_bounds = rasterio.transform.array_bounds(height, width, dst_transform)
    return transform_bounds(WEB_CRS, "EPSG:4326", *merc_bounds)


def export_truecolor(stack_path: Path, out: Path, quality: int = 82) -> None:
    """True-colour composite, stretched then warped, alpha over nodata edges.

    The warp rotates the footprint inside its Mercator bounding box, so the
    corners are empty; WebP keeps the alpha channel (JPEG could not) at a
    fraction of PNG's size for photographic content.
    """
    with rasterio.open(stack_path) as src:
        names = list(src.descriptions)
        dst_transform, width, height = _mercator_grid(src)

        channels = []
        valid = None
        for band_name in ("B04", "B03", "B02"):
            data = src.read(names.index(band_name) + 1)
            finite = np.isfinite(data)
            valid = finite if valid is None else (valid & finite)
            # Stretch in the native grid (statistics of real pixels only),
            # reserving 0 for nodata so the warp's nodata handling stays clean.
            scaled = np.where(finite, stretch(data) * 254 + 1, 0).astype(np.uint8)
            channels.append(
                _warp(scaled, src, (height, width), dst_transform, Resampling.bilinear, 0)
            )
        alpha = _warp(
            (valid * np.uint8(255)), src, (height, width), dst_transform,
            Resampling.nearest, 0,
        )

    rgba = np.dstack(channels + [alpha])
    Image.fromarray(rgba, "RGBA").save(out, format="WEBP", quality=quality, method=5)
    log.info("wrote %s (%.2f MB, %dx%d)", out.name, out.stat().st_size / 1e6, width, height)


# --------------------------------------------------------------------------- #
# Commune statistics
# --------------------------------------------------------------------------- #


def commune_stats(pred_path: Path, communes_path: Path) -> gpd.GeoDataFrame:
    """Zonal per-class pixel counts for every commune the scene touches."""
    with rasterio.open(pred_path) as src:
        pred = src.read(1)
        grid_box = box(*src.bounds)
        crs, transform = src.crs, src.transform

    communes = gpd.read_file(communes_path).rename(columns=str.lower).to_crs(crs)
    communes = communes[communes.intersects(grid_box)].reset_index(drop=True)

    # Burn commune index+1 onto the prediction grid; 0 stays "no commune".
    zones = rasterize(
        ((geom, i + 1) for i, geom in enumerate(communes.geometry)),
        out_shape=pred.shape,
        transform=transform,
        fill=0,
        all_touched=False,
        dtype="int32",
    )

    n_classes = int(pred.max()) + 1
    labelled = (pred > 0) & (zones > 0)
    counts = np.bincount(
        (zones[labelled] - 1) * n_classes + pred[labelled],
        minlength=len(communes) * n_classes,
    ).reshape(len(communes), n_classes)

    px_km2 = abs(transform.a * transform.e) / 1e6
    in_grid = np.bincount(zones[zones > 0] - 1, minlength=len(communes))

    total = counts.sum(axis=1) * px_km2
    forest = counts[:, sorted(FOREST_IDS)].sum(axis=1) * px_km2
    built = counts[:, BUILT_ID] * px_km2
    with np.errstate(invalid="ignore", divide="ignore"):
        communes["forest_pct"] = np.round(100 * forest / total, 1)
        communes["built_pct"] = np.round(100 * built / total, 1)
    communes["classified_km2"] = np.round(total, 2)
    communes["covered"] = np.round(
        in_grid * px_km2 * 1e6 / communes.geometry.area, 3
    ).clip(0, 1)

    communes["geometry"] = communes.geometry.simplify(SIMPLIFY_M)
    out = communes[
        ["commune", "lau2", "forest_pct", "built_pct", "classified_km2", "covered",
         "geometry"]
    ].to_crs("EPSG:4326")
    log.info(
        "commune stats: %d communes intersect the scene, %d fully covered",
        len(out), int((out["covered"] >= FULL_COVERAGE).sum()),
    )
    return out


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #


def build(out_dir: Path = WEB) -> dict:
    ingest = json.loads((PROCESSED / "latest_ingest.json").read_text())
    labels_meta = json.loads((PROCESSED / "latest_labels.json").read_text())
    rf = json.loads((RESULTS / "latest_rf.json").read_text())
    crf = json.loads((RESULTS / "latest_crf.json").read_text())

    kept = [c for c in labels_meta["class_scheme"]["classes"] if c["kept"]]
    stack_path = PROCESSED / ingest["files"]["stack"]
    # The displayed pair is out-of-fold unary vs the CRF on top of it -- the
    # identical model underneath, so every visible difference is the smoothing.
    unary_path = RESULTS / crf["files"]["unary"]
    crf_path = RESULTS / crf["files"]["crf"]
    out_dir.mkdir(parents=True, exist_ok=True)

    bounds4326 = export_classification(unary_path, kept, out_dir / "classification_rf.png")
    export_classification(crf_path, kept, out_dir / "classification_crf.png")
    # The reference itself, same palette and grid: seeing CORINE's >= 25 ha
    # polygons next to the 10 m ML maps is the clearest way to grasp both what
    # the model learned from and what the reference cannot represent.
    export_classification(
        PROCESSED / labels_meta["files"]["labels"], kept,
        out_dir / "classification_corine.png",
    )
    export_truecolor(stack_path, out_dir / "truecolor.webp")
    # Commune statistics from the CRF map: it is the end product a per-commune
    # report would be built from (and loses half as much area to speckle).
    communes = commune_stats(crf_path, RAW / "communes4326.geojson")

    west, south, east, north = bounds4326
    blocked = rf["spatial_block_split"]
    random_ = rf["random_pixel_split"]
    boundary = rf["accuracy_by_boundary_distance"]
    theta = crf["method"]["theta_chosen"]
    crf_res = crf["results"][f"{theta:g}"]
    unary_res = crf["results"]["unary"]

    data = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        # Leaflet wants [[south, west], [north, east]].
        "bounds": [[south, west], [north, east]],
        "scene": {
            "id": ingest["stac"]["item_id"],
            "date": ingest["stac"]["datetime"][:10],
            "epsg": ingest["grid"]["epsg"],
            "cloud_pct": round(ingest["stac"]["scene_cloud_cover_pct"], 2),
        },
        "classes": [
            {
                "id": c["class_id"],
                "name": c["name"],
                "colour": c["colour"],
                "share": round(c["share"], 4),
            }
            for c in kept
        ],
        "metrics": {
            "blocked_accuracy": round(blocked["overall_accuracy"], 3),
            "random_accuracy": round(random_["overall_accuracy"], 3),
            "kappa": round(blocked["cohen_kappa"], 3),
            "macro_f1": round(blocked["macro_f1"], 3),
            "n_test": blocked["n_test_pixels"],
            "boundary_near": round(boundary[0]["accuracy"], 3),
            "boundary_far": round(boundary[-1]["accuracy"], 3),
            "boundary_within_100m_pct": round(
                100 * labels_meta["limitations"]["fraction_within_boundary_buffer"], 1
            ),
            "crf": {
                "theta": theta,
                "unary_accuracy": round(unary_res["overall_accuracy"], 3),
                "crf_accuracy": round(crf_res["overall_accuracy"], 3),
                "speckle_before_km2": round(
                    unary_res["vectorisation"]["dropped_speck_area_km2"], 1
                ),
                "speckle_after_km2": round(
                    crf_res["vectorisation"]["dropped_speck_area_km2"], 1
                ),
                "road_survival": round(crf_res["thin_artificial_retention"], 3),
                "patches_before": unary_res["fragmentation"]["n_patches"],
                "patches_after": crf_res["fragmentation"]["n_patches"],
            },
        },
        "full_coverage": FULL_COVERAGE,
        "layers": {
            "truecolor": "truecolor.webp",
            "rf": "classification_rf.png",
            "crf": "classification_crf.png",
            "corine": "classification_corine.png",
        },
        "communes": json.loads(communes.to_json(to_wgs84=False)),
    }

    js = "// Generated by src/postprocessing/web_export.py -- do not edit.\n"
    js += "const APP_DATA = " + json.dumps(data, separators=(",", ":")) + ";\n"
    (out_dir / "data.js").write_text(js)
    log.info("wrote data.js (%.2f MB)", (out_dir / "data.js").stat().st_size / 1e6)
    return data


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out-dir", type=Path, default=WEB)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S",
    )
    logging.getLogger("rasterio").setLevel(logging.WARNING)

    data = build(args.out_dir)
    total = sum(f.stat().st_size for f in args.out_dir.rglob("*") if f.is_file())
    print(f"\nweb/ ready: {len(data['communes']['features'])} communes, "
          f"{len(data['classes'])} classes, {total / 1e6:.1f} MB total")
    return 0


if __name__ == "__main__":
    sys.exit(main())
