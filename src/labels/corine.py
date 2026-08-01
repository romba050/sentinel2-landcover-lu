"""CORINE Land Cover 2018 labels, rasterised onto the Sentinel-2 grid.

Why CORINE
----------
CLC is the European Environment Agency's pan-European land-cover inventory. It
is the natural label source for an EEA-facing project, and its weaknesses are
as instructive as its strengths (see "Known label limitations" below).

Where the data comes from
-------------------------
The Copernicus Land Monitoring Service download portal requires an account, so
this module instead queries the EEA's public ArcGIS REST service
(``image.discomap.eea.europa.eu``), which serves the *same* CLC2018 vector
product anonymously. Polygons arrive in the CLC native CRS, ETRS89 / LAEA
Europe (EPSG:3035).

CRS handling
------------
The labels are reprojected to the imagery's UTM CRS, never the other way round.
Reprojecting a 10 m image resamples every pixel and smears class boundaries;
reprojecting vector polygons is exact up to the datum transformation, and the
rasterisation onto the image grid then happens once, deliberately.

Known label limitations (quantified by this module)
---------------------------------------------------
*   **25 ha minimum mapping unit.** CLC cannot represent anything smaller. At
    10 m resolution one CLC polygon is at least 2500 pixels, so narrow rivers,
    hedgerows, individual buildings and field margins are absent by design.
*   **Boundary error.** CLC polygons are photo-interpreted at 1:100 000, so
    their edges are only accurate to ~100 m. Pixels near a polygon edge are
    frequently mislabelled. This module writes a distance-to-boundary raster so
    that error can be measured rather than assumed away.
*   **Mixed classes.** "Complex cultivation patterns" (242) and "agriculture
    with significant natural vegetation" (243) are explicitly *mosaics*. No
    per-pixel spectral classifier can reproduce them faithfully; they are
    label-space constructs, not surface types.

Usage
-----
    uv run python -m src.labels.corine
    uv run python -m src.labels.corine --min-class-share 0.0   # keep rare classes
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import requests
from rasterio.features import rasterize
from rasterio.warp import transform_bounds
from scipy import ndimage

from src.ingest.sentinel2 import TargetGrid, grid_from_raster

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROCESSED = PROJECT_ROOT / "data" / "processed"
DEFAULT_RAW = PROJECT_ROOT / "data" / "raw"

#: EEA public ArcGIS REST endpoint, LAEA (EPSG:3035) variant, vector layer 0.
CLC_SERVICE = (
    "https://image.discomap.eea.europa.eu/arcgis/rest/services"
    "/Corine/CLC2018_LAEA/MapServer/0/query"
)
CLC_CRS = "EPSG:3035"
CLC_CODE_FIELD = "Code_18"
#: The service caps a single response at 1000 features.
PAGE_SIZE = 1000
#: Fetch a little beyond the AOI so edge polygons arrive whole.
QUERY_BUFFER_M = 1000.0

#: CLC's minimum mapping unit. Nothing smaller than this exists in the product.
CLC_MMU_HA = 25.0


@dataclass(frozen=True)
class LandCoverClass:
    code: int
    name: str
    colour: str  # for QGIS / matplotlib legends


#: Target scheme: CLC level-3 (44 classes) aggregated to 12 classes that are
#: plausibly separable from a single-date optical image. Level 3 is far too
#: fine -- it distinguishes e.g. "port areas" from "airports", a distinction
#: that is about *function*, not reflectance, and that no spectral classifier
#: can recover. Level 1 (5 classes) throws away the forest-type and
#: arable/pasture splits, which are exactly what Sentinel-2 is good at.
CLASSES: tuple[LandCoverClass, ...] = (
    LandCoverClass(1, "Artificial surfaces", "#e6004d"),
    LandCoverClass(2, "Arable land", "#ffffa8"),
    LandCoverClass(3, "Permanent crops", "#e68000"),
    LandCoverClass(4, "Pastures", "#e6e64d"),
    LandCoverClass(5, "Heterogeneous agriculture", "#ffe6a6"),
    LandCoverClass(6, "Broad-leaved forest", "#80ff00"),
    LandCoverClass(7, "Coniferous forest", "#00a600"),
    LandCoverClass(8, "Mixed forest", "#4dff00"),
    LandCoverClass(9, "Scrub / herbaceous", "#a6ff80"),
    LandCoverClass(10, "Open spaces, little vegetation", "#ccffcc"),
    LandCoverClass(11, "Wetlands", "#a6a6ff"),
    LandCoverClass(12, "Water bodies", "#00ccf2"),
)
CLASS_BY_CODE = {c.code: c for c in CLASSES}

#: 0 means "no valid label"; it is never a class.
NODATA_LABEL = 0

#: CLC level-3 code -> target class. Complete for the 44 standard CLC codes.
CLC_TO_CLASS: dict[int, int] = {
    # 1.x Artificial surfaces. Green urban areas (141) and sport/leisure (142)
    # are lumped in with sealed surfaces even though they are vegetated: they
    # are a fraction of a percent here, and splitting them out would create a
    # class defined by land *use* rather than land cover.
    111: 1, 112: 1, 121: 1, 122: 1, 123: 1, 124: 1,
    131: 1, 132: 1, 133: 1, 141: 1, 142: 1,
    # 2.1 Arable
    211: 2, 212: 2, 213: 2,
    # 2.2 Permanent crops (vineyards matter on the Luxembourg Moselle)
    221: 3, 222: 3, 223: 3,
    # 2.3 Pastures
    231: 4,
    # 2.4 Heterogeneous agricultural areas -- mosaics, see module docstring
    241: 5, 242: 5, 243: 5, 244: 5,
    # 3.1 Forests, kept split: Sentinel-2's red-edge and SWIR bands separate
    # broadleaf from conifer well, and it is the most defensible fine
    # distinction in the scheme.
    311: 6, 312: 7, 313: 8,
    # 3.2 Scrub and herbaceous
    321: 9, 322: 9, 323: 9, 324: 9,
    # 3.3 Open spaces with little or no vegetation
    331: 10, 332: 10, 333: 10, 334: 10, 335: 10,
    # 4.x Wetlands
    411: 11, 412: 11, 421: 11, 422: 11, 423: 11,
    # 5.x Water
    511: 12, 512: 12, 521: 12, 522: 12, 523: 12,
}

#: Classes covering less than this share of the AOI are demoted to nodata.
#: A class present as one or two polygons cannot be honestly evaluated under a
#: spatially-blocked split -- it lands entirely inside one fold, so its test
#: score is either undefined or measured on zero held-out pixels.
DEFAULT_MIN_CLASS_SHARE = 0.005

#: Pixels within this distance of a class boundary are flagged as
#: label-uncertain. 100 m reflects CLC's 1:100 000 photo-interpretation scale.
DEFAULT_BOUNDARY_BUFFER_M = 100.0


# --------------------------------------------------------------------------- #
# Fetching
# --------------------------------------------------------------------------- #


def fetch_clc_polygons(
    bounds_wgs84: tuple[float, float, float, float],
    service: str = CLC_SERVICE,
    timeout: int = 180,
) -> gpd.GeoDataFrame:
    """Download CLC2018 polygons intersecting ``bounds_wgs84`` (min/max lon/lat).

    Pages through the service's 1000-feature response cap.
    """
    features: list[dict] = []
    offset = 0
    while True:
        params = {
            "where": "1=1",
            "geometry": ",".join(str(v) for v in bounds_wgs84),
            "geometryType": "esriGeometryEnvelope",
            "inSR": 4326,
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": f"{CLC_CODE_FIELD},Area_Ha,ID",
            "returnGeometry": "true",
            "outSR": 3035,
            "f": "geojson",
            "resultOffset": offset,
            "resultRecordCount": PAGE_SIZE,
        }
        resp = requests.get(service, params=params, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
        if "error" in payload:
            raise RuntimeError(f"CLC service error: {payload['error']}")
        page = payload.get("features", [])
        features.extend(page)
        log.info("  fetched %d polygons (offset %d)", len(page), offset)
        if len(page) < PAGE_SIZE:
            break
        offset += PAGE_SIZE

    if not features:
        raise RuntimeError("CLC service returned no polygons for this AOI.")

    gdf = gpd.GeoDataFrame.from_features(features, crs=CLC_CRS)
    gdf[CLC_CODE_FIELD] = gdf[CLC_CODE_FIELD].astype(int)
    log.info("CLC2018: %d polygons in %s", len(gdf), CLC_CRS)
    return gdf


def query_bounds_for_grid(grid: TargetGrid, buffer_m: float = QUERY_BUFFER_M):
    """AOI bounds in WGS84, padded so edge polygons come back whole."""
    left, bottom, right, top = grid.bounds
    padded = (left - buffer_m, bottom - buffer_m, right + buffer_m, top + buffer_m)
    return transform_bounds(grid.crs, "EPSG:4326", *padded, densify_pts=64)


# --------------------------------------------------------------------------- #
# Rasterisation
# --------------------------------------------------------------------------- #


def assign_classes(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Map CLC level-3 codes to the target scheme, dropping unmapped codes."""
    gdf = gdf.copy()
    gdf["class_id"] = gdf[CLC_CODE_FIELD].map(CLC_TO_CLASS)
    unmapped = gdf[gdf["class_id"].isna()][CLC_CODE_FIELD].unique()
    if len(unmapped):
        log.warning("Dropping unmapped CLC codes: %s", sorted(unmapped.tolist()))
        gdf = gdf[gdf["class_id"].notna()]
    gdf["class_id"] = gdf["class_id"].astype(int)
    return gdf


def rasterise(gdf: gpd.GeoDataFrame, grid: TargetGrid, field: str) -> np.ndarray:
    """Burn ``field`` onto the Sentinel-2 grid, reprojecting the vectors first.

    ``all_touched=False`` gives the pixel-centre rule: every pixel takes exactly
    one class. ``all_touched=True`` would let adjacent polygons overwrite each
    other along shared edges, systematically biasing boundary pixels toward
    whichever polygon happens to be drawn last.
    """
    projected = gdf.to_crs(grid.crs)
    # Largest polygons first, so that where CLC has (rare) overlaps the smaller,
    # more specific polygon wins by being burned last.
    order = projected.assign(_a=projected.area).sort_values("_a", ascending=False)
    shapes = ((geom, int(value)) for geom, value in zip(order.geometry, order[field], strict=True))
    return rasterize(
        shapes,
        out_shape=grid.shape,
        transform=grid.transform,
        fill=NODATA_LABEL,
        all_touched=False,
        dtype="int32",
    )


def boundary_distance_m(labels: np.ndarray, pixel_size_m: float) -> np.ndarray:
    """Distance from each pixel to the nearest class boundary, in metres."""
    interior = np.ones_like(labels, dtype=bool)
    interior[:, :-1] &= labels[:, :-1] == labels[:, 1:]
    interior[:, 1:] &= labels[:, 1:] == labels[:, :-1]
    interior[:-1, :] &= labels[:-1, :] == labels[1:, :]
    interior[1:, :] &= labels[1:, :] == labels[:-1, :]
    return ndimage.distance_transform_edt(interior).astype(np.float32) * pixel_size_m


def drop_rare_classes(
    labels: np.ndarray, min_share: float
) -> tuple[np.ndarray, list[dict]]:
    """Demote under-represented classes to nodata, returning the class table."""
    labelled = labels != NODATA_LABEL
    total = int(labelled.sum())
    table: list[dict] = []
    for cls in CLASSES:
        n = int((labels == cls.code).sum())
        if n == 0:
            continue
        share = n / total
        keep = share >= min_share
        table.append(
            {
                "class_id": cls.code,
                "name": cls.name,
                "colour": cls.colour,
                "pixels": n,
                "share": share,
                "kept": keep,
            }
        )
        if not keep:
            log.warning(
                "Dropping class %d (%s): %.3f%% of the AOI is too little to "
                "evaluate under a spatially-blocked split",
                cls.code, cls.name, 100 * share,
            )
            labels = np.where(labels == cls.code, NODATA_LABEL, labels)
    return labels, table


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #


def _write(path: Path, data: np.ndarray, grid: TargetGrid, dtype, nodata, name):
    with rasterio.open(
        path, "w", driver="GTiff", height=grid.height, width=grid.width, count=1,
        dtype=dtype, crs=grid.crs, transform=grid.transform, nodata=nodata,
        tiled=True, blockxsize=512, blockysize=512, compress="deflate",
        predictor=3 if np.dtype(dtype).kind == "f" else 2,
    ) as dst:
        dst.write(data.astype(dtype), 1)
        dst.set_band_description(1, name)
        dst.update_tags(1, **{"class_scheme": "corine-aggregated-12"})
    log.info("wrote %s", path.name)


def build_labels(
    processed_dir: Path = DEFAULT_PROCESSED,
    raw_dir: Path = DEFAULT_RAW,
    min_class_share: float = DEFAULT_MIN_CLASS_SHARE,
    boundary_buffer_m: float = DEFAULT_BOUNDARY_BUFFER_M,
) -> dict:
    manifest = json.loads((processed_dir / "latest_ingest.json").read_text())
    stack_path = processed_dir / manifest["files"]["stack"]
    grid = grid_from_raster(stack_path)
    stem = stack_path.stem.replace("_stack", "")
    log.info("Target grid from %s: %dx%d @ 10 m, %s", stack_path.name,
             grid.width, grid.height, grid.crs)

    gdf = fetch_clc_polygons(query_bounds_for_grid(grid))
    raw_dir.mkdir(parents=True, exist_ok=True)
    vector_path = raw_dir / f"clc2018_{stem}.gpkg"
    gdf.to_file(vector_path, driver="GPKG", layer="clc2018")
    log.info("wrote %s (%d polygons, %s)", vector_path.name, len(gdf), gdf.crs)

    gdf = assign_classes(gdf)

    # Rasterise both the aggregated scheme and the original CLC codes: the
    # level-3 raster is what lets us later ask *which* CLC class an error came
    # from, which the aggregated one has already thrown away.
    log.info("Reprojecting %s -> %s and rasterising", CLC_CRS, grid.crs)
    labels = rasterise(gdf, grid, "class_id").astype(np.uint8)
    clc_codes = rasterise(gdf, grid, CLC_CODE_FIELD).astype(np.uint16)

    unlabelled = int((labels == NODATA_LABEL).sum())
    if unlabelled:
        log.info(
            "%d pixels (%.3f%%) have no CLC polygon -- reprojection slivers at "
            "the AOI edge", unlabelled, 100 * unlabelled / labels.size,
        )

    labels, class_table = drop_rare_classes(labels, min_class_share)
    dist = boundary_distance_m(labels, pixel_size_m=abs(grid.transform.a))
    near_boundary = dist < boundary_buffer_m

    log.info("Class composition (of labelled pixels):")
    for row in class_table:
        flag = "" if row["kept"] else "   [dropped]"
        log.info("  %2d %-32s %7.3f%%%s", row["class_id"], row["name"],
                 100 * row["share"], flag)

    kept = [r for r in class_table if r["kept"]]
    log.info(
        "%d usable classes; %.1f%% of labelled pixels lie within %.0f m of a "
        "class boundary and are therefore label-uncertain",
        len(kept), 100 * near_boundary[labels != NODATA_LABEL].mean(), boundary_buffer_m,
    )

    _write(processed_dir / f"{stem}_labels.tif", labels, grid, "uint8",
           NODATA_LABEL, "class_id")
    _write(processed_dir / f"{stem}_labels_clc3.tif", clc_codes, grid, "uint16",
           0, "clc_code_18")
    _write(processed_dir / f"{stem}_label_boundary_dist.tif", dist, grid, "float32",
           np.nan, "distance_to_class_boundary_m")

    areas_ha = gdf.to_crs(grid.crs).area / 10_000
    meta = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {
            "product": "CORINE Land Cover 2018 (CLC2018), vector",
            "provider": "European Environment Agency",
            "service": CLC_SERVICE,
            "native_crs": CLC_CRS,
            "licence": "Copernicus data, free and open (Reg. (EU) No 1159/2013)",
            "polygons": int(len(gdf)),
        },
        "grid": {
            "crs": grid.crs.to_string(),
            "epsg": grid.crs.to_epsg(),
            "width": grid.width,
            "height": grid.height,
            "resolution_m": abs(grid.transform.a),
            "reprojection": f"{CLC_CRS} -> {grid.crs.to_string()} (vector, before rasterising)",
            "rasterisation": "pixel-centre rule (all_touched=False)",
        },
        "class_scheme": {
            "name": "corine-aggregated-12",
            "nodata": NODATA_LABEL,
            "classes": class_table,
            "min_class_share": min_class_share,
        },
        "limitations": {
            "clc_minimum_mapping_unit_ha": CLC_MMU_HA,
            "min_polygon_area_ha": float(areas_ha.min()),
            "median_polygon_area_ha": float(areas_ha.median()),
            "boundary_buffer_m": boundary_buffer_m,
            "fraction_within_boundary_buffer": float(
                near_boundary[labels != NODATA_LABEL].mean()
            ),
            "unlabelled_pixels": unlabelled,
            "note": (
                "CLC has a 25 ha minimum mapping unit and is photo-interpreted "
                "at 1:100 000, so pixels near polygon edges are frequently "
                "mislabelled. Classes 5 (heterogeneous agriculture) are mosaics "
                "by definition and cannot be reproduced per-pixel."
            ),
        },
        "files": {
            "labels": f"{stem}_labels.tif",
            "clc_level3": f"{stem}_labels_clc3.tif",
            "boundary_distance": f"{stem}_label_boundary_dist.tif",
            "vector": str(vector_path.relative_to(PROJECT_ROOT)),
        },
    }
    meta_path = processed_dir / f"{stem}_labels_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    (processed_dir / "latest_labels.json").write_text(json.dumps(meta, indent=2))
    log.info("wrote %s", meta_path.name)
    return meta


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Fetch CORINE 2018 and rasterise it onto the Sentinel-2 grid.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED)
    p.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    p.add_argument("--min-class-share", type=float, default=DEFAULT_MIN_CLASS_SHARE,
                   help="Demote classes below this area share to nodata.")
    p.add_argument("--boundary-buffer", type=float, default=DEFAULT_BOUNDARY_BUFFER_M,
                   help="Distance (m) within which a pixel is label-uncertain.")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S",
    )
    logging.getLogger("rasterio").setLevel(logging.WARNING)

    meta = build_labels(
        processed_dir=args.processed_dir,
        raw_dir=args.raw_dir,
        min_class_share=args.min_class_share,
        boundary_buffer_m=args.boundary_buffer,
    )
    kept = [c for c in meta["class_scheme"]["classes"] if c["kept"]]
    print(f"\nLabels ready: {len(kept)} classes on a "
          f"{meta['grid']['width']}x{meta['grid']['height']} grid (EPSG:{meta['grid']['epsg']})")
    for c in kept:
        print(f"  {c['class_id']:2d}  {c['name']:<32s} {c['share']:7.2%}")
    lim = meta["limitations"]
    print(f"  label-uncertain (< {lim['boundary_buffer_m']:.0f} m from a boundary): "
          f"{lim['fraction_within_boundary_buffer']:.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
