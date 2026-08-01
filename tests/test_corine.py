"""Unit tests for the CORINE label pipeline (pure logic, no network)."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from rasterio.crs import CRS
from rasterio.transform import from_origin
from shapely.geometry import box

from src.ingest.sentinel2 import TargetGrid
from src.labels import corine

# --------------------------------------------------------------------------- #
# Class scheme integrity
# --------------------------------------------------------------------------- #


def test_every_clc_code_maps_to_a_declared_class():
    assert set(corine.CLC_TO_CLASS.values()) <= set(corine.CLASS_BY_CODE)


def test_all_44_standard_clc_level3_codes_are_covered():
    """CLC has exactly 44 level-3 classes; an unmapped one would be silently
    dropped from the labels, quietly shrinking the training set."""
    assert len(corine.CLC_TO_CLASS) == 44


def test_nodata_is_not_a_class():
    assert corine.NODATA_LABEL not in corine.CLASS_BY_CODE


def test_class_ids_are_contiguous_from_one():
    assert [c.code for c in corine.CLASSES] == list(range(1, len(corine.CLASSES) + 1))


def test_forest_types_stay_separate():
    """Broadleaf/coniferous/mixed must not collapse -- they are the distinction
    Sentinel-2's red-edge and SWIR bands are best placed to make."""
    assert len({corine.CLC_TO_CLASS[c] for c in (311, 312, 313)}) == 3


def test_clc_level1_groups_are_not_mixed_across_target_classes():
    """A target class may merge CLC codes, but never across level-1 groups
    (e.g. an artificial code must not land in an agricultural class)."""
    by_class: dict[int, set[int]] = {}
    for clc, target in corine.CLC_TO_CLASS.items():
        by_class.setdefault(target, set()).add(clc // 100)
    for target, level1 in by_class.items():
        assert len(level1) == 1, f"class {target} spans CLC level-1 groups {level1}"


# --------------------------------------------------------------------------- #
# Rasterisation onto the imagery grid
# --------------------------------------------------------------------------- #


@pytest.fixture
def grid() -> TargetGrid:
    # 100 x 100 px at 10 m, origin on a round UTM coordinate.
    return TargetGrid(
        crs=CRS.from_epsg(32631),
        transform=from_origin(700000.0, 5500000.0, 10.0, 10.0),
        width=100,
        height=100,
    )


def _gdf_in_laea(grid: TargetGrid, polys: list[tuple[object, int]]) -> gpd.GeoDataFrame:
    gdf = gpd.GeoDataFrame(
        {"class_id": [c for _, c in polys]},
        geometry=[g for g, _ in polys],
        crs=grid.crs,
    )
    # Round-trip through CLC's native CRS so the test exercises the real
    # reprojection path rather than a no-op.
    return gdf.to_crs(corine.CLC_CRS)


def test_rasterise_reprojects_from_laea_and_lands_on_the_grid(grid):
    """A polygon covering the left half of the AOI must burn the left half."""
    left, bottom, right, top = grid.bounds
    mid = (left + right) / 2
    gdf = _gdf_in_laea(grid, [(box(left, bottom, mid, top), 6)])

    out = corine.rasterise(gdf, grid, "class_id")

    assert out.shape == grid.shape
    # Allow a one-pixel margin for the datum transformation.
    assert (out[:, :49] == 6).all()
    assert (out[:, 51:] == corine.NODATA_LABEL).all()


def test_rasterise_leaves_uncovered_area_as_nodata(grid):
    left, bottom, _, _ = grid.bounds
    gdf = _gdf_in_laea(grid, [(box(left, bottom, left + 100, bottom + 100), 4)])
    out = corine.rasterise(gdf, grid, "class_id")
    assert (out == corine.NODATA_LABEL).sum() > 0.9 * out.size


def test_rasterise_smaller_polygon_wins_over_the_larger_one(grid):
    """Where CLC polygons overlap, the more specific (smaller) one should win."""
    left, bottom, right, top = grid.bounds
    big = box(left, bottom, right, top)
    small = box(left + 200, bottom + 200, left + 400, bottom + 400)
    gdf = _gdf_in_laea(grid, [(big, 2), (small, 7)])
    out = corine.rasterise(gdf, grid, "class_id")
    assert 7 in np.unique(out)
    assert (out == 7).sum() < (out == 2).sum()


# --------------------------------------------------------------------------- #
# Boundary distance
# --------------------------------------------------------------------------- #


def test_boundary_distance_is_zero_on_an_edge_and_grows_inward():
    labels = np.ones((1, 11), dtype=np.uint8)
    labels[0, 6:] = 2
    dist = corine.boundary_distance_m(labels, pixel_size_m=10.0)
    # The two pixels straddling the change are both boundary pixels.
    assert dist[0, 5] == 0 and dist[0, 6] == 0
    # Distance increases as you move away from the edge.
    assert dist[0, 0] > dist[0, 3] > dist[0, 4] > dist[0, 5]


def test_boundary_distance_scales_with_pixel_size():
    labels = np.array([[1, 1, 1, 2, 2, 2]], dtype=np.uint8)
    d10 = corine.boundary_distance_m(labels, 10.0)
    d20 = corine.boundary_distance_m(labels, 20.0)
    assert np.allclose(d20, 2 * d10)


def test_uniform_labels_have_no_boundary():
    labels = np.ones((20, 20), dtype=np.uint8)
    assert (corine.boundary_distance_m(labels, 10.0) > 0).all()


# --------------------------------------------------------------------------- #
# Rare-class handling
# --------------------------------------------------------------------------- #


def test_drop_rare_classes_demotes_only_the_rare_ones():
    labels = np.full(1000, 6, dtype=np.uint8)
    labels[:2] = 11  # 0.2% -- below the 0.5% default
    labels[2:300] = 1
    labels, table = corine.drop_rare_classes(labels, min_share=0.005)

    assert 11 not in np.unique(labels)
    assert (labels == corine.NODATA_LABEL).sum() == 2
    kept = {r["class_id"] for r in table if r["kept"]}
    assert kept == {1, 6}
    dropped = next(r for r in table if r["class_id"] == 11)
    assert dropped["kept"] is False and dropped["pixels"] == 2


def test_drop_rare_classes_with_zero_threshold_keeps_everything():
    labels = np.full(1000, 6, dtype=np.uint8)
    labels[:1] = 11
    out, table = corine.drop_rare_classes(labels, min_share=0.0)
    assert set(np.unique(out)) == {6, 11}
    assert all(r["kept"] for r in table)


def test_drop_rare_classes_ignores_nodata_when_computing_shares():
    labels = np.zeros(1000, dtype=np.uint8)
    labels[:100] = 6  # 100 labelled pixels, all one class
    _, table = corine.drop_rare_classes(labels, min_share=0.005)
    assert next(r for r in table if r["class_id"] == 6)["share"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Query geometry
# --------------------------------------------------------------------------- #


def test_query_bounds_pad_the_aoi(grid):
    padded = corine.query_bounds_for_grid(grid, buffer_m=1000.0)
    from rasterio.warp import transform_bounds

    exact = transform_bounds(grid.crs, "EPSG:4326", *grid.bounds, densify_pts=64)
    assert padded[0] < exact[0] and padded[1] < exact[1]
    assert padded[2] > exact[2] and padded[3] > exact[3]
