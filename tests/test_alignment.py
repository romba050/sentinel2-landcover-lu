"""Integration checks against a real ingested stack.

Skipped automatically when ``data/processed`` is empty, so a fresh clone can
still run ``pytest`` without a 260 MB download. Produce the inputs with::

    uv run python -m src.ingest.sentinel2
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"
MANIFEST = PROCESSED / "latest_ingest.json"



def _stack_available() -> bool:
    """True only when the manifest AND the raster it points to are on disk.

    The manifest is committed to git but the .tif is gitignored, so a fresh
    clone (e.g. CI) has the former without the latter.
    """
    if not MANIFEST.exists():
        return False
    stack = json.loads(MANIFEST.read_text())["files"]["stack"]
    return (PROCESSED / stack).exists()


pytestmark = pytest.mark.skipif(
    not _stack_available(), reason="no ingested stack; run `python -m src.ingest.sentinel2`"
)


@pytest.fixture(scope="module")
def meta() -> dict:
    return json.loads(MANIFEST.read_text())


@pytest.fixture(scope="module")
def stack(meta):
    with rasterio.open(PROCESSED / meta["files"]["stack"]) as src:
        yield src


def test_grid_is_10m_and_origin_snaps_to_the_20m_grid(stack):
    """Guarantees the 20 m bands land on exact 2x2 blocks of the 10 m grid."""
    assert stack.res == (10.0, 10.0)
    assert stack.bounds.left % 20 == 0
    assert stack.bounds.top % 20 == 0
    assert stack.width % 2 == 0
    assert stack.height % 2 == 0


def test_metadata_matches_the_raster_on_disk(stack, meta):
    assert stack.width == meta["grid"]["width"]
    assert stack.height == meta["grid"]["height"]
    assert stack.crs.to_epsg() == meta["grid"]["epsg"]
    assert list(stack.descriptions) == [b["name"] for b in meta["bands"]]


def test_aoi_is_fully_contained_in_the_written_grid(stack, meta):
    """Snapping must grow the window, never clip the requested AOI."""
    from rasterio.warp import transform_bounds

    got = transform_bounds(stack.crs, "EPSG:4326", *stack.bounds)
    want = meta["aoi_bbox_wgs84"]
    assert got[0] <= want[0] and got[1] <= want[1]
    assert got[2] >= want[2] and got[3] >= want[3]


def test_nodata_is_nan_not_zero(stack):
    """Zero is a plausible reflectance; using it as nodata would poison training."""
    assert np.isnan(stack.nodata)


def test_reflectance_is_physically_plausible(stack, meta):
    """Catches a mis-applied BOA offset, which shifts every band by 0.1."""
    reflectance_bands = [b for b in meta["bands"] if b.get("native_resolution_m")]
    for band in reflectance_bands:
        data = stack.read(band["index"])
        valid = data[np.isfinite(data)]
        median = float(np.median(valid))
        assert 0.0 < median < 0.6, f"{band['name']} median {median:.3f} is implausible"
        assert valid.min() >= 0.0, f"{band['name']} has negative reflectance"


def test_ndvi_separates_the_airport_runway_from_forest(stack, meta):
    """A weak end-to-end semantic check: NDVI must span built-up to dense forest."""
    names = [b["name"] for b in meta["bands"]]
    ndvi = stack.read(names.index("NDVI") + 1)
    valid = ndvi[np.isfinite(ndvi)]
    assert np.percentile(valid, 1) < 0.2, "no low-NDVI (built/bare) pixels found"
    assert np.percentile(valid, 99) > 0.8, "no high-NDVI (dense vegetation) pixels found"


def test_upsampled_20m_band_is_not_shifted_against_a_native_10m_band(stack, meta):
    """The decisive alignment test.

    B08 (10 m) and B8A (20 m, upsampled) view almost the same NIR wavelengths,
    so their correlation must peak at zero displacement. A half-pixel error in
    the window arithmetic would move the peak to a neighbouring offset.
    """
    names = [b["name"] for b in meta["bands"]]
    b08 = stack.read(names.index("B08") + 1)
    b8a = stack.read(names.index("B8A") + 1)
    core = (slice(3, -3), slice(3, -3))

    best_r, best_shift = -np.inf, None
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            shifted = np.roll(np.roll(b8a, dy, axis=0), dx, axis=1)
            m = (np.isfinite(b08) & np.isfinite(shifted))[core]
            r = np.corrcoef(b08[core][m], shifted[core][m])[0, 1]
            if r > best_r:
                best_r, best_shift = r, (dy, dx)

    assert best_shift == (0, 0), f"bands are shifted by {best_shift} (r={best_r:.4f})"
    assert best_r > 0.9


def test_masked_pixels_are_nan_across_every_band(stack, meta):
    """Masking is applied once to the whole stack; no band may leak cloud data."""
    with rasterio.open(PROCESSED / meta["files"]["valid"]) as vsrc:
        valid = vsrc.read(1).astype(bool)
    if valid.all():
        pytest.skip("scene is entirely valid; nothing to check")
    for band in meta["bands"]:
        data = stack.read(band["index"])
        assert np.isnan(data[~valid]).all(), f"{band['name']} retains data under the mask"
