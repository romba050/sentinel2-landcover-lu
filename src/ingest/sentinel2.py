"""Sentinel-2 L2A ingestion for the Luxembourg land-cover project.

What this module does
---------------------
1. Searches the Element84 *Earth Search* STAC API for Sentinel-2 L2A scenes over
   an area of interest (AOI). The underlying data are Cloud-Optimised GeoTIFFs
   (COGs) in the public ``s3://sentinel-cogs`` AWS Open Data bucket, so no
   credentials are required and we never download a whole 1 GB scene.
2. Picks the scene whose granule footprint *fully contains* the AOI and has the
   lowest cloud cover.
3. Reads only the AOI window out of each band COG over HTTP range requests
   (GDAL ``/vsicurl``), resampling the 20 m bands onto the native 10 m grid.
4. Converts digital numbers to surface reflectance, handling the
   post-baseline-04.00 BOA offset.
5. Derives a cloud/shadow validity mask from the Scene Classification Layer
   (SCL) and four spectral indices (NDVI, NDWI, NDBI, NDMI).
6. Writes a stacked float32 GeoTIFF plus SCL/validity rasters and a JSON
   provenance sidecar.

Deliberate design decisions
---------------------------
*   **The output CRS is whatever the scene's native UTM zone is.** It is read
    from the STAC item, never hardcoded. Luxembourg straddles the UTM 31N/32N
    boundary at 6 degrees East, so the "obvious" answer (EPSG:32632) is wrong
    for AOIs that only sit inside an MGRS tile of zone 31. Reprojecting imagery
    would resample every pixel and blur class boundaries; instead we keep the
    imagery on its native grid and later reproject the *labels* onto it.
*   **The target grid is snapped to the 10 m band's own pixel grid**, with even
    offsets and even dimensions. Sentinel-2's 10 m / 20 m / 60 m rasters share
    an upper-left corner, so an even-aligned 10 m window maps to an exactly
    integral 20 m window: the 2x upsample introduces no sub-pixel shift.
*   **Masked pixels become NaN rather than 0.** Zero is a plausible reflectance
    and would be silently learned as a feature value by a classifier.

Usage
-----
    uv run python -m src.ingest.sentinel2 --list
    uv run python -m src.ingest.sentinel2
    uv run python -m src.ingest.sentinel2 --item-id S2B_31UGR_20240918_0_L2A
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from pystac_client import Client
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from rasterio.windows import Window
from rasterio.windows import from_bounds as window_from_bounds
from rasterio.windows import transform as window_transform
from shapely.geometry import box, shape

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

STAC_URL = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-l2a"

#: Default AOI: a ~20 x 20 km window around Luxembourg City, in EPSG:4326
#: (min_lon, min_lat, max_lon, max_lat). Chosen because it packs a lot of class
#: diversity into a small area -- dense urban core, the Kirchberg office
#: district, Findel airport, the Gruenewald broadleaf forest, the Alzette
#: valley, and arable/pasture land to the east around Niederanven.
DEFAULT_BBOX = (6.01, 49.54, 6.29, 49.72)
DEFAULT_AOI_NAME = "luxembourg-city"

DEFAULT_START = "2024-05-01"
DEFAULT_END = "2024-09-30"
DEFAULT_MAX_CLOUD = 10.0

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "processed"

#: Sentinel-2 L2A digital numbers are scaled reflectance.
REFLECTANCE_SCALE = 10_000.0
#: From processing baseline 04.00 (2022-01-25) onward, ESA added a +1000 offset
#: to L2A BOA reflectance so that slightly-negative retrievals survive the
#: unsigned integer encoding. It must be subtracted again -- unless the STAC
#: provider already did so (``earthsearch:boa_offset_applied``).
BOA_ADD_OFFSET = -1000.0
BOA_OFFSET_BASELINE = 4.0


@dataclass(frozen=True)
class BandSpec:
    """One Sentinel-2 band we want in the output stack."""

    name: str  # ESA band id, e.g. "B08"
    asset: str  # Earth Search STAC asset key, e.g. "nir"
    resolution_m: int  # native ground sample distance
    wavelength_nm: int
    description: str


#: Band order of the output stack. The four 10 m bands come first, then the six
#: 20 m bands. B01/B09/B10 (60 m, atmospheric) are excluded: they carry no
#: usable land-cover signal at 10 m and would just add noise features.
BANDS: tuple[BandSpec, ...] = (
    BandSpec("B02", "blue", 10, 490, "Blue"),
    BandSpec("B03", "green", 10, 560, "Green"),
    BandSpec("B04", "red", 10, 665, "Red"),
    BandSpec("B08", "nir", 10, 842, "NIR (broad)"),
    BandSpec("B05", "rededge1", 20, 705, "Red edge 1"),
    BandSpec("B06", "rededge2", 20, 740, "Red edge 2"),
    BandSpec("B07", "rededge3", 20, 783, "Red edge 3"),
    BandSpec("B8A", "nir08", 20, 865, "NIR (narrow)"),
    BandSpec("B11", "swir16", 20, 1610, "SWIR 1"),
    BandSpec("B12", "swir22", 20, 2190, "SWIR 2"),
)

#: Scene Classification Layer codes (ESA L2A Algorithm Theoretical Basis Doc).
SCL_CLASSES = {
    0: "no_data",
    1: "saturated_or_defective",
    2: "dark_area_pixels",
    3: "cloud_shadow",
    4: "vegetation",
    5: "not_vegetated",
    6: "water",
    7: "unclassified",
    8: "cloud_medium_probability",
    9: "cloud_high_probability",
    10: "thin_cirrus",
    11: "snow_or_ice",
}

#: SCL codes we refuse to train on. Note that 2 (dark area / topographic
#: shadow) and 7 (unclassified) are *kept*: dropping them would systematically
#: remove north-facing slopes and genuinely ambiguous land cover, biasing the
#: training set toward easy pixels. 11 (snow) is dropped because a summer scene
#: flagged as snow is almost always a bright-roof or cloud false positive.
INVALID_SCL = frozenset({0, 1, 3, 8, 9, 10, 11})

#: GDAL settings for efficient anonymous range-reads of remote COGs.
GDAL_ENV = dict(
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
    GDAL_HTTP_MULTIPLEX="YES",
    GDAL_HTTP_VERSION="2",
    VSI_CACHE="TRUE",
    VSI_CACHE_SIZE="536870912",  # 512 MB
    AWS_NO_SIGN_REQUEST="YES",
)


# --------------------------------------------------------------------------- #
# 1. Scene discovery
# --------------------------------------------------------------------------- #


def search_scenes(
    bbox: tuple[float, float, float, float],
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    max_cloud: float = DEFAULT_MAX_CLOUD,
    stac_url: str = STAC_URL,
) -> list:
    """Return STAC items intersecting ``bbox`` within the date/cloud filters."""
    client = Client.open(stac_url)
    search = client.search(
        collections=[COLLECTION],
        bbox=list(bbox),
        datetime=f"{start}/{end}",
        query={"eo:cloud_cover": {"lt": max_cloud}},
    )
    items = list(search.item_collection())
    log.info(
        "STAC: %d scenes intersect the AOI between %s and %s with cloud < %.0f%%",
        len(items),
        start,
        end,
        max_cloud,
    )
    return items


def tile_id(item) -> str:
    """MGRS tile id, e.g. ``31UGR``, from whichever property the API exposes."""
    code = item.properties.get("grid:code")
    if code:
        return code.replace("MGRS-", "")
    p = item.properties
    return f"{p['mgrs:utm_zone']}{p['mgrs:latitude_band']}{p['mgrs:grid_square']}"


def scene_summary(item, aoi_geom) -> dict:
    """Compact, sortable description of a candidate scene."""
    footprint = shape(item.geometry)
    covered = footprint.intersection(aoi_geom).area / aoi_geom.area
    return {
        "id": item.id,
        "datetime": item.properties["datetime"][:10],
        "tile": tile_id(item),
        "epsg": item.properties["proj:epsg"],
        "cloud_cover": float(item.properties["eo:cloud_cover"]),
        "aoi_coverage": float(covered),
        "contains_aoi": bool(footprint.contains(aoi_geom)),
    }


def select_scene(items: list, bbox: tuple[float, float, float, float]):
    """Pick the least-cloudy scene whose footprint fully contains the AOI.

    We require full containment rather than mosaicking several granules. A
    mosaic would mix acquisition dates (and therefore sun angle and phenology)
    across the AOI, which is exactly the kind of artefact a per-pixel
    classifier happily learns as if it were land cover.
    """
    aoi_geom = box(*bbox)
    summaries = [scene_summary(it, aoi_geom) for it in items]
    by_id = {it.id: it for it in items}

    complete = [s for s in summaries if s["contains_aoi"]]
    if not complete:
        best = sorted(summaries, key=lambda s: -s["aoi_coverage"])[:5]
        raise RuntimeError(
            "No single scene fully contains the AOI. Best partial coverage:\n"
            + "\n".join(f"  {s['id']}  {s['aoi_coverage']:.1%}" for s in best)
            + "\nShrink the AOI (--bbox) or widen the date range (--start/--end)."
        )

    complete.sort(key=lambda s: (s["cloud_cover"], s["datetime"]))
    chosen = complete[0]
    log.info(
        "Selected %s (%s, tile %s, EPSG:%d, %.2f%% cloud) from %d fully-covering candidates",
        chosen["id"],
        chosen["datetime"],
        chosen["tile"],
        chosen["epsg"],
        chosen["cloud_cover"],
        len(complete),
    )
    return by_id[chosen["id"]], complete


# --------------------------------------------------------------------------- #
# 2. Target grid
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TargetGrid:
    """The 10 m raster grid that every output of this project is snapped to."""

    crs: rasterio.crs.CRS
    transform: rasterio.Affine
    width: int
    height: int

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        left = self.transform.c
        top = self.transform.f
        right = left + self.width * self.transform.a
        bottom = top + self.height * self.transform.e  # transform.e is negative
        return (left, bottom, right, top)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)


def build_target_grid(item, bbox: tuple[float, float, float, float]) -> TargetGrid:
    """Snap the WGS84 AOI onto the scene's native 10 m pixel grid.

    Offsets and dimensions are forced even so that the window is also integral
    on the 20 m grid (the two share an origin), making the 2x upsample of the
    red-edge/SWIR bands exact.
    """
    ref_href = item.assets[BANDS[0].asset].href  # a 10 m band
    with rasterio.Env(**GDAL_ENV), rasterio.open(ref_href) as src:
        if abs(src.transform.a) != 10:
            raise RuntimeError(f"Expected a 10 m reference band, got {src.transform.a} m")

        # densify_pts matters: the AOI edges are straight in lon/lat but curved
        # in UTM, so a 4-corner reprojection can clip a sliver off the middle.
        utm_bounds = transform_bounds("EPSG:4326", src.crs, *bbox, densify_pts=64)

        win = window_from_bounds(*utm_bounds, transform=src.transform)
        col_off = int(np.floor(win.col_off))
        row_off = int(np.floor(win.row_off))
        col_end = int(np.ceil(win.col_off + win.width))
        row_end = int(np.ceil(win.row_off + win.height))

        # Grow (never shrink) to even offsets and even extents.
        col_off -= col_off % 2
        row_off -= row_off % 2
        width = col_end - col_off
        height = row_end - row_off
        width += width % 2
        height += height % 2

        outside = (
            col_off < 0
            or row_off < 0
            or col_off + width > src.width
            or row_off + height > src.height
        )
        if outside:
            raise RuntimeError("AOI window falls outside the scene raster extent")

        win = Window(col_off, row_off, width, height)
        grid = TargetGrid(
            crs=src.crs,
            transform=window_transform(win, src.transform),
            width=width,
            height=height,
        )

    log.info(
        "Target grid: %d x %d px @ 10 m in %s (%.1f x %.1f km)",
        grid.width,
        grid.height,
        grid.crs,
        grid.width * 10 / 1000,
        grid.height * 10 / 1000,
    )
    return grid


# --------------------------------------------------------------------------- #
# 3. Windowed reads
# --------------------------------------------------------------------------- #


def read_to_grid(
    href: str, grid: TargetGrid, resampling: Resampling
) -> tuple[np.ndarray, float | None]:
    """Read only the AOI window from a remote COG, resampled onto ``grid``.

    Returns the raw (unscaled) array and the source nodata value.
    """
    with rasterio.Env(**GDAL_ENV), rasterio.open(href) as src:
        if src.crs != grid.crs:
            raise RuntimeError(f"{href} is in {src.crs}, expected {grid.crs}")

        win = window_from_bounds(*grid.bounds, transform=src.transform)
        # For an aligned grid these are already integers; round defensively so
        # floating-point noise cannot shift the window by a pixel.
        win = Window(
            round(win.col_off), round(win.row_off), round(win.width), round(win.height)
        )
        with warnings.catch_warnings():
            # rasterio 1.5 reshapes the output buffer in-place, which NumPy 2.5
            # deprecates. Harmless here and not ours to fix.
            warnings.filterwarnings(
                "ignore", message=".*Setting the shape on a NumPy array.*",
                category=DeprecationWarning,
            )
            arr = src.read(1, window=win, out_shape=grid.shape, resampling=resampling)
        return arr, src.nodata


def to_reflectance(dn: np.ndarray, nodata: float | None, offset: float) -> np.ndarray:
    """Scale digital numbers to surface reflectance, nodata -> NaN."""
    invalid = dn == (0 if nodata is None else nodata)
    refl = (dn.astype(np.float32) + np.float32(offset)) / np.float32(REFLECTANCE_SCALE)
    refl[invalid] = np.nan
    return refl


def boa_offset_for(item) -> float:
    """How much to add to the DNs before scaling.

    Earth Search flags whether it already removed ESA's +1000 BOA offset. If it
    did, or if the scene predates baseline 04.00, no correction is needed.
    Getting this wrong shifts every band by 0.1 reflectance -- enough to wreck
    NDVI and to make two scenes silently incomparable.
    """
    if item.properties.get("earthsearch:boa_offset_applied", False):
        return 0.0
    try:
        baseline = float(item.properties.get("s2:processing_baseline", "0"))
    except ValueError:
        baseline = 0.0
    return BOA_ADD_OFFSET if baseline >= BOA_OFFSET_BASELINE else 0.0


def read_stack(
    item, grid: TargetGrid, resampling: Resampling = Resampling.bilinear
) -> tuple[np.ndarray, np.ndarray]:
    """Read all bands + SCL for the AOI. Returns (reflectance stack, SCL)."""
    offset = boa_offset_for(item)
    log.info(
        "BOA offset applied by provider: %s -> using DN offset %+.0f",
        item.properties.get("earthsearch:boa_offset_applied", False),
        offset,
    )

    stack = np.empty((len(BANDS), grid.height, grid.width), dtype=np.float32)
    for i, spec in enumerate(BANDS):
        # 10 m bands need no resampling; 20 m bands get bilinear (smooth, no
        # blocking) -- SCL is the exception and must stay categorical.
        rs = Resampling.nearest if spec.resolution_m == 10 else resampling
        dn, nodata = read_to_grid(item.assets[spec.asset].href, grid, rs)
        stack[i] = to_reflectance(dn, nodata, offset)
        log.info("  read %-4s (%s, %2d m) -> %s", spec.name, spec.asset, spec.resolution_m, rs.name)

    scl, _ = read_to_grid(item.assets["scl"].href, grid, Resampling.nearest)
    log.info("  read SCL  (scl, 20 m) -> nearest")
    return stack, scl.astype(np.uint8)


# --------------------------------------------------------------------------- #
# 4. Masking and indices
# --------------------------------------------------------------------------- #


def validity_mask(scl: np.ndarray, stack: np.ndarray) -> np.ndarray:
    """Boolean mask of pixels usable for training/prediction."""
    scl_ok = ~np.isin(scl, list(INVALID_SCL))
    finite = np.isfinite(stack).all(axis=0)
    return scl_ok & finite


def normalised_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(a - b) / (a + b), with a zero denominator yielding NaN rather than inf."""
    denom = a + b
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(denom == 0, np.nan, (a - b) / denom)
    return out.astype(np.float32)


def compute_indices(stack: np.ndarray) -> dict[str, np.ndarray]:
    """Spectral indices that give a per-pixel classifier cheap, robust contrast.

    Each index uses a distinct band pair. NDMI is deliberately *not* included:
    NDMI = (NIR - SWIR1)/(NIR + SWIR1) is the exact negative of NDBI, so adding
    both would hand a tree ensemble two perfectly anti-correlated features,
    splitting its importance scores across a duplicate for no gain.
    """
    idx = {s.name: i for i, s in enumerate(BANDS)}
    green, red, rededge1 = stack[idx["B03"]], stack[idx["B04"]], stack[idx["B05"]]
    nir, swir1, swir2 = stack[idx["B08"]], stack[idx["B11"]], stack[idx["B12"]]
    return {
        # Vegetation vigour -- separates vegetated from built/bare.
        "NDVI": normalised_difference(nir, red),
        # McFeeters water index -- open water and wet surfaces.
        "NDWI": normalised_difference(green, nir),
        # Built-up index -- impervious surfaces vs vegetation.
        "NDBI": normalised_difference(swir1, nir),
        # Red-edge chlorophyll -- saturates far later than NDVI, so it still
        # discriminates dense broadleaf forest from vigorous summer crops.
        "NDRE": normalised_difference(nir, rededge1),
        # Bare-soil / senescence contrast, useful against ploughed fields.
        "NBR2": normalised_difference(swir1, swir2),
    }


# --------------------------------------------------------------------------- #
# 5. Writing
# --------------------------------------------------------------------------- #


def _write_raster(path: Path, data: np.ndarray, grid: TargetGrid, *, dtype, nodata, names):
    profile = dict(
        driver="GTiff",
        height=grid.height,
        width=grid.width,
        count=data.shape[0],
        dtype=dtype,
        crs=grid.crs,
        transform=grid.transform,
        nodata=nodata,
        tiled=True,
        blockxsize=512,
        blockysize=512,
        compress="deflate",
        predictor=3 if np.dtype(dtype).kind == "f" else 2,
        BIGTIFF="IF_SAFER",
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype(dtype))
        for i, name in enumerate(names, start=1):
            dst.set_band_description(i, name)
        # Overviews so the rasters pan smoothly in QGIS.
        dst.build_overviews([2, 4, 8, 16], Resampling.average)
    log.info("wrote %s (%d bands, %.1f MB)", path.name, data.shape[0], path.stat().st_size / 1e6)


def write_outputs(
    out_dir: Path,
    aoi_name: str,
    item,
    grid: TargetGrid,
    stack: np.ndarray,
    indices: dict[str, np.ndarray],
    scl: np.ndarray,
    valid: np.ndarray,
    bbox: tuple[float, float, float, float],
    candidates: list[dict],
) -> dict:
    """Write the feature stack, SCL, validity mask and a provenance sidecar."""
    out_dir.mkdir(parents=True, exist_ok=True)
    date = item.properties["datetime"][:10]
    stem = f"{aoi_name}_{tile_id(item)}_{date}"

    band_names = [s.name for s in BANDS] + list(indices)
    features = np.concatenate([stack, np.stack(list(indices.values()))], axis=0)
    # One masking rule, applied once, to everything: invalid -> NaN.
    features[:, ~valid] = np.nan

    stack_path = out_dir / f"{stem}_stack.tif"
    _write_raster(
        stack_path, features, grid, dtype="float32", nodata=np.nan, names=band_names
    )
    _write_raster(
        out_dir / f"{stem}_scl.tif", scl[None], grid, dtype="uint8", nodata=0, names=["SCL"]
    )
    _write_raster(
        out_dir / f"{stem}_valid.tif",
        valid[None].astype(np.uint8),
        grid,
        dtype="uint8",
        nodata=255,
        names=["valid"],
    )

    scl_hist = {
        SCL_CLASSES.get(int(c), str(c)): int(n)
        for c, n in zip(*np.unique(scl, return_counts=True), strict=True)
    }
    meta = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "aoi_name": aoi_name,
        "aoi_bbox_wgs84": list(bbox),
        "stac": {
            "api": STAC_URL,
            "collection": COLLECTION,
            "item_id": item.id,
            "datetime": item.properties["datetime"],
            "tile": tile_id(item),
            "platform": item.properties.get("platform"),
            "scene_cloud_cover_pct": item.properties.get("eo:cloud_cover"),
            "processing_baseline": item.properties.get("s2:processing_baseline"),
            "boa_offset_applied_by_provider": item.properties.get(
                "earthsearch:boa_offset_applied"
            ),
            "s3_path": item.properties.get("earthsearch:s3_path"),
        },
        "grid": {
            "crs": grid.crs.to_string(),
            "epsg": grid.crs.to_epsg(),
            "transform": list(grid.transform)[:6],
            "width": grid.width,
            "height": grid.height,
            "resolution_m": 10,
            "bounds_utm": list(grid.bounds),
        },
        "bands": [
            {
                "index": i + 1,
                "name": s.name,
                "asset": s.asset,
                "native_resolution_m": s.resolution_m,
                "wavelength_nm": s.wavelength_nm,
                "description": s.description,
            }
            for i, s in enumerate(BANDS)
        ]
        + [
            {"index": len(BANDS) + i + 1, "name": n, "description": "spectral index"}
            for i, n in enumerate(indices)
        ],
        "masking": {
            "invalid_scl_codes": sorted(INVALID_SCL),
            "invalid_scl_names": [SCL_CLASSES[c] for c in sorted(INVALID_SCL)],
            "scl_histogram": scl_hist,
            "valid_pixels": int(valid.sum()),
            "total_pixels": int(valid.size),
            "valid_fraction": float(valid.mean()),
        },
        "files": {
            "stack": stack_path.name,
            "scl": f"{stem}_scl.tif",
            "valid": f"{stem}_valid.tif",
        },
        "alternative_scenes": candidates[:10],
    }
    meta_path = out_dir / f"{stem}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    log.info("wrote %s", meta_path.name)

    # A stable pointer so downstream modules do not have to guess the filename.
    (out_dir / "latest_ingest.json").write_text(json.dumps(meta, indent=2))
    return meta


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #


def ingest(
    bbox: tuple[float, float, float, float] = DEFAULT_BBOX,
    aoi_name: str = DEFAULT_AOI_NAME,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    max_cloud: float = DEFAULT_MAX_CLOUD,
    out_dir: Path = DEFAULT_OUT_DIR,
    item_id: str | None = None,
    resampling: Resampling = Resampling.bilinear,
) -> dict:
    """Run the full ingestion and return the provenance metadata."""
    items = search_scenes(bbox, start, end, max_cloud)
    if not items:
        raise RuntimeError("STAC search returned no scenes; relax --max-cloud or the date range.")

    if item_id:
        matches = [i for i in items if i.id == item_id]
        if not matches:
            raise RuntimeError(f"{item_id} is not among the {len(items)} search results.")
        item = matches[0]
        candidates = [scene_summary(i, box(*bbox)) for i in items]
        log.info("Using pinned scene %s", item_id)
    else:
        item, candidates = select_scene(items, bbox)

    grid = build_target_grid(item, bbox)
    stack, scl = read_stack(item, grid, resampling)
    valid = validity_mask(scl, stack)
    log.info(
        "Valid pixels: %d / %d (%.2f%%)", valid.sum(), valid.size, 100 * valid.mean()
    )
    if valid.mean() < 0.5:
        log.warning("Less than half the AOI is usable -- consider a different scene.")

    indices = compute_indices(stack)
    for name, arr in indices.items():
        v = arr[valid]
        log.info(
            "  %-5s min=%+.3f mean=%+.3f max=%+.3f",
            name, np.nanmin(v), np.nanmean(v), np.nanmax(v),
        )

    return write_outputs(
        out_dir, aoi_name, item, grid, stack, indices, scl, valid, bbox, candidates
    )


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Fetch and preprocess a Sentinel-2 L2A subset for the AOI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--bbox", nargs=4, type=float, default=list(DEFAULT_BBOX),
                   metavar=("MINLON", "MINLAT", "MAXLON", "MAXLAT"))
    p.add_argument("--aoi-name", default=DEFAULT_AOI_NAME)
    p.add_argument("--start", default=DEFAULT_START)
    p.add_argument("--end", default=DEFAULT_END)
    p.add_argument("--max-cloud", type=float, default=DEFAULT_MAX_CLOUD)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--item-id", default=None,
                   help="Pin a specific STAC item instead of auto-selecting.")
    p.add_argument("--resampling", default="bilinear", choices=["nearest", "bilinear", "cubic"],
                   help="How to upsample the 20 m bands to 10 m.")
    p.add_argument("--list", action="store_true", help="List candidate scenes and exit.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    # rasterio logs "boto3 not available, falling back to a DummySession" on
    # every remote open. We read anonymous public COGs over https, so that is
    # expected, not a problem worth 11 lines of output.
    logging.getLogger("rasterio").setLevel(logging.WARNING)
    bbox = tuple(args.bbox)

    if args.list:
        items = search_scenes(bbox, args.start, args.end, args.max_cloud)
        aoi = box(*bbox)
        rows = sorted(
            (scene_summary(i, aoi) for i in items),
            key=lambda s: (not s["contains_aoi"], s["cloud_cover"]),
        )
        print(f"\n{'item id':32s} {'date':11s} {'tile':6s} {'epsg':6s} {'cloud%':>7s} {'aoi%':>6s}")
        print("-" * 76)
        for s in rows:
            flag = "" if s["contains_aoi"] else "  (partial)"
            print(f"{s['id']:32s} {s['datetime']:11s} {s['tile']:6s} "
                  f"{s['epsg']:<6d} {s['cloud_cover']:7.2f} {s['aoi_coverage']:5.1%}{flag}")
        return 0

    try:
        meta = ingest(
            bbox=bbox,
            aoi_name=args.aoi_name,
            start=args.start,
            end=args.end,
            max_cloud=args.max_cloud,
            out_dir=args.out_dir,
            item_id=args.item_id,
            resampling=Resampling[args.resampling],
        )
    except RuntimeError as exc:
        log.error("%s", exc)
        return 1

    print(f"\nIngested {meta['stac']['item_id']} -> {args.out_dir}")
    print(f"  grid   : {meta['grid']['width']} x {meta['grid']['height']} @ 10 m, "
          f"EPSG:{meta['grid']['epsg']}")
    print(f"  bands  : {len(meta['bands'])}")
    print(f"  valid  : {meta['masking']['valid_fraction']:.2%} of pixels")
    return 0


if __name__ == "__main__":
    sys.exit(main())
