"""Unit tests for the Sentinel-2 ingestion logic.

These cover the pure functions only -- no network. The end-to-end correctness of
the windowed reads is checked separately by the band-alignment test in
``tests/test_alignment.py``, which needs an ingested stack on disk.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.ingest import sentinel2 as s2


class _FakeItem:
    def __init__(self, **properties):
        self.properties = properties
        self.id = properties.get("id", "fake")
        self.geometry = properties.get("geometry")


# --------------------------------------------------------------------------- #
# BOA offset
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("applied", "baseline", "expected"),
    [
        (True, "05.11", 0.0),  # provider already removed it
        (False, "05.11", -1000.0),  # modern baseline, we must remove it
        (False, "04.00", -1000.0),  # exactly at the cutover
        (False, "03.01", 0.0),  # pre-baseline-04.00: no offset exists
        (False, "not-a-number", 0.0),  # malformed metadata must not crash
    ],
)
def test_boa_offset_for(applied, baseline, expected):
    item = _FakeItem(
        **{"earthsearch:boa_offset_applied": applied, "s2:processing_baseline": baseline}
    )
    assert s2.boa_offset_for(item) == expected


def test_boa_offset_missing_properties_defaults_to_no_offset():
    assert s2.boa_offset_for(_FakeItem()) == 0.0


# --------------------------------------------------------------------------- #
# Reflectance scaling
# --------------------------------------------------------------------------- #


def test_to_reflectance_scales_and_nans_nodata():
    dn = np.array([[0, 1000, 2000]], dtype=np.uint16)
    out = s2.to_reflectance(dn, nodata=0, offset=0.0)
    assert np.isnan(out[0, 0])
    assert out[0, 1] == pytest.approx(0.1)
    assert out[0, 2] == pytest.approx(0.2)


def test_to_reflectance_applies_boa_offset():
    """A DN of 1500 is 0.05 reflectance once the +1000 offset is removed."""
    dn = np.array([[1500]], dtype=np.uint16)
    assert s2.to_reflectance(dn, nodata=0, offset=-1000.0)[0, 0] == pytest.approx(0.05)


def test_to_reflectance_treats_missing_nodata_as_zero():
    dn = np.array([[0, 500]], dtype=np.uint16)
    out = s2.to_reflectance(dn, nodata=None, offset=0.0)
    assert np.isnan(out[0, 0]) and out[0, 1] == pytest.approx(0.05)


# --------------------------------------------------------------------------- #
# Indices
# --------------------------------------------------------------------------- #


def test_normalised_difference_zero_denominator_is_nan_not_inf():
    a = np.array([0.0, 0.5], dtype=np.float32)
    b = np.array([0.0, 0.1], dtype=np.float32)
    out = s2.normalised_difference(a, b)
    assert np.isnan(out[0])
    assert out[1] == pytest.approx((0.5 - 0.1) / 0.6)


def test_normalised_difference_is_bounded():
    rng = np.random.default_rng(0)
    a = rng.uniform(0.001, 1.0, 5000).astype(np.float32)
    b = rng.uniform(0.001, 1.0, 5000).astype(np.float32)
    out = s2.normalised_difference(a, b)
    assert np.all(np.abs(out) <= 1.0 + 1e-6)


def _synthetic_stack(**overrides) -> np.ndarray:
    """A 1-pixel stack with plausible vegetation reflectance."""
    values = {
        "B02": 0.03, "B03": 0.06, "B04": 0.04, "B08": 0.32, "B05": 0.10,
        "B06": 0.25, "B07": 0.30, "B8A": 0.34, "B11": 0.19, "B12": 0.10,
    }
    values.update(overrides)
    return np.array([[[values[s.name]]] for s in s2.BANDS], dtype=np.float32)


def test_compute_indices_known_values():
    idx = s2.compute_indices(_synthetic_stack())
    assert idx["NDVI"][0, 0] == pytest.approx((0.32 - 0.04) / 0.36, rel=1e-5)
    assert idx["NDWI"][0, 0] == pytest.approx((0.06 - 0.32) / 0.38, rel=1e-5)
    assert idx["NDBI"][0, 0] == pytest.approx((0.19 - 0.32) / 0.51, rel=1e-5)
    assert idx["NDRE"][0, 0] == pytest.approx((0.32 - 0.10) / 0.42, rel=1e-5)
    assert idx["NBR2"][0, 0] == pytest.approx((0.19 - 0.10) / 0.29, rel=1e-5)


def test_indices_are_not_linearly_redundant():
    """Guards the NDMI mistake: no index may be another's exact negative.

    NDMI = (NIR-SWIR1)/(NIR+SWIR1) is identically -NDBI. Shipping both would
    give a tree ensemble two perfectly anti-correlated features and split its
    importance scores across a duplicate.
    """
    rng = np.random.default_rng(42)
    stack = rng.uniform(0.01, 0.6, (len(s2.BANDS), 64, 64)).astype(np.float32)
    idx = s2.compute_indices(stack)
    names = list(idx)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            r = np.corrcoef(idx[a].ravel(), idx[b].ravel())[0, 1]
            assert abs(r) < 0.99, f"{a} and {b} are redundant (r={r:.4f})"


def test_water_pixel_has_positive_ndwi_and_negative_ndvi():
    """Sanity check against a physically realistic spectrum."""
    water = _synthetic_stack(B03=0.05, B04=0.03, B08=0.01, B11=0.005, B05=0.02)
    idx = s2.compute_indices(water)
    assert idx["NDWI"][0, 0] > 0
    assert idx["NDVI"][0, 0] < 0


# --------------------------------------------------------------------------- #
# Masking
# --------------------------------------------------------------------------- #


def test_validity_mask_rejects_cloud_and_keeps_shadowed_terrain():
    scl = np.array([[4, 5, 6, 7, 2, 3, 8, 9, 10, 11, 0, 1]], dtype=np.uint8)
    stack = np.ones((len(s2.BANDS), *scl.shape), dtype=np.float32) * 0.2
    mask = s2.validity_mask(scl, stack)
    # kept: vegetation, not-vegetated, water, unclassified, dark-area
    assert mask[0, :5].all()
    # dropped: shadow, clouds, cirrus, snow, no-data, defective
    assert not mask[0, 5:].any()


def test_validity_mask_rejects_nan_reflectance():
    scl = np.full((1, 3), 4, dtype=np.uint8)
    stack = np.ones((len(s2.BANDS), 1, 3), dtype=np.float32) * 0.2
    stack[3, 0, 1] = np.nan  # one band missing is enough to void the pixel
    mask = s2.validity_mask(scl, stack)
    assert mask.tolist() == [[True, False, True]]


def test_invalid_scl_codes_are_documented():
    assert set(s2.INVALID_SCL) <= set(s2.SCL_CLASSES)


# --------------------------------------------------------------------------- #
# Scene selection
# --------------------------------------------------------------------------- #


def _stac_like(item_id, cloud, geom_bbox, date="2024-08-24"):
    minx, miny, maxx, maxy = geom_bbox
    return _FakeItem(
        id=item_id,
        geometry={
            "type": "Polygon",
            "coordinates": [
                [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy), (minx, miny)]
            ],
        },
        **{
            "eo:cloud_cover": cloud,
            "datetime": f"{date}T10:37:24Z",
            "proj:epsg": 32631,
            "grid:code": "MGRS-31UGR",
        },
    )


def test_select_scene_prefers_full_coverage_over_low_cloud():
    """A pristine scene that only half-covers the AOI must lose to a cloudier
    one that covers all of it -- mosaicking dates is worse than a little cloud."""
    aoi = (6.0, 49.5, 6.3, 49.7)
    partial = _stac_like("partial-but-clear", 0.0, (6.0, 49.5, 6.15, 49.7))
    full = _stac_like("full-but-cloudy", 5.0, (5.9, 49.4, 6.4, 49.8))
    item, candidates = s2.select_scene([partial, full], aoi)
    assert item.id == "full-but-cloudy"
    assert [c["id"] for c in candidates] == ["full-but-cloudy"]


def test_select_scene_picks_least_cloudy_among_full_coverage():
    aoi = (6.0, 49.5, 6.3, 49.7)
    items = [
        _stac_like("cloudy", 8.0, (5.9, 49.4, 6.4, 49.8)),
        _stac_like("clear", 0.5, (5.9, 49.4, 6.4, 49.8)),
    ]
    item, _ = s2.select_scene(items, aoi)
    assert item.id == "clear"


def test_select_scene_raises_with_guidance_when_nothing_covers_the_aoi():
    aoi = (6.0, 49.5, 6.3, 49.7)
    items = [_stac_like("partial", 0.0, (6.0, 49.5, 6.1, 49.7))]
    with pytest.raises(RuntimeError, match="No single scene fully contains"):
        s2.select_scene(items, aoi)


def test_tile_id_from_grid_code_and_from_mgrs_parts():
    assert s2.tile_id(_FakeItem(**{"grid:code": "MGRS-31UGR"})) == "31UGR"
    parts = _FakeItem(
        **{"mgrs:utm_zone": 31, "mgrs:latitude_band": "U", "mgrs:grid_square": "GR"}
    )
    assert s2.tile_id(parts) == "31UGR"


# --------------------------------------------------------------------------- #
# Target grid geometry
# --------------------------------------------------------------------------- #


def test_target_grid_bounds_and_shape_are_consistent():
    from rasterio.crs import CRS
    from rasterio.transform import from_origin

    grid = s2.TargetGrid(
        crs=CRS.from_epsg(32631),
        transform=from_origin(716940.0, 5512700.0, 10.0, 10.0),
        width=2108,
        height=2086,
    )
    left, bottom, right, top = grid.bounds
    assert (left, top) == (716940.0, 5512700.0)
    assert right == left + 2108 * 10
    assert bottom == top - 2086 * 10
    assert grid.shape == (2086, 2108)
    # Even extents keep the 20 m bands exactly integral on this grid.
    assert grid.width % 2 == 0 and grid.height % 2 == 0
