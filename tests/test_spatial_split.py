"""Tests for spatially-blocked splitting and the metrics that use it."""

from __future__ import annotations

import numpy as np
import pytest
from rasterio.transform import from_origin

from src.evaluation import metrics as M
from src.evaluation import spatial_split as S

TRANSFORM = from_origin(700000.0, 5500000.0, 10.0, 10.0)  # 10 m pixels


# --------------------------------------------------------------------------- #
# Block geometry
# --------------------------------------------------------------------------- #


def test_block_ids_group_pixels_into_squares_of_the_requested_size():
    blocks = S.block_ids((40, 40), TRANSFORM, block_size_m=100.0)  # 10 px blocks
    assert blocks[0, 0] == blocks[9, 9]  # same block
    assert blocks[0, 0] != blocks[0, 10]  # next block across
    assert blocks[0, 0] != blocks[10, 0]  # next block down
    assert len(np.unique(blocks)) == 16  # 4 x 4 blocks


def test_block_ids_are_stable_for_the_same_grid():
    a = S.block_ids((40, 40), TRANSFORM, 100.0)
    b = S.block_ids((40, 40), TRANSFORM, 100.0)
    assert np.array_equal(a, b)


def test_block_size_smaller_than_a_pixel_degrades_to_one_block_per_pixel():
    blocks = S.block_ids((5, 5), TRANSFORM, block_size_m=1.0)
    assert len(np.unique(blocks)) == 25


# --------------------------------------------------------------------------- #
# The core guarantee
# --------------------------------------------------------------------------- #


def test_blocked_split_never_puts_the_same_block_on_both_sides():
    """The whole point: a block is entirely train or entirely test."""
    eligible = np.ones((100, 100), dtype=bool)
    split = S.spatial_block_split(eligible, TRANSFORM, test_fraction=0.3,
                                  block_size_m=100.0, seed=0)
    blocks = S.block_ids(eligible.shape, TRANSFORM, 100.0)
    train_blocks = set(np.unique(blocks[split.train]).tolist())
    test_blocks = set(np.unique(blocks[split.test]).tolist())
    assert not (train_blocks & test_blocks)


def test_random_split_does_put_neighbours_on_both_sides():
    """The contrast case -- this is exactly the leakage the blocked split avoids."""
    eligible = np.ones((100, 100), dtype=bool)
    split = S.random_pixel_split(eligible, test_fraction=0.3, seed=0)
    # A test pixel with an immediately adjacent training pixel.
    adjacent = split.test[:, :-1] & split.train[:, 1:]
    assert adjacent.sum() > 1000


def test_split_partitions_exactly_the_eligible_pixels():
    eligible = np.zeros((60, 60), dtype=bool)
    eligible[10:50, 10:50] = True
    split = S.spatial_block_split(eligible, TRANSFORM, 0.3, 100.0, seed=1)
    assert np.array_equal(split.train | split.test, eligible)
    assert not (split.train & split.test).any()


def test_ineligible_pixels_are_never_selected():
    eligible = np.zeros((60, 60), dtype=bool)
    eligible[:30] = True
    split = S.spatial_block_split(eligible, TRANSFORM, 0.3, 100.0, seed=1)
    assert not split.train[30:].any()
    assert not split.test[30:].any()


def test_blocked_split_test_fraction_is_roughly_honoured():
    eligible = np.ones((200, 200), dtype=bool)
    split = S.spatial_block_split(eligible, TRANSFORM, test_fraction=0.25,
                                  block_size_m=200.0, seed=3)
    frac = split.test.sum() / eligible.sum()
    assert 0.15 < frac < 0.35


def test_too_few_blocks_raises_rather_than_silently_degrading():
    eligible = np.ones((20, 20), dtype=bool)
    with pytest.raises(ValueError, match="usable blocks"):
        S.spatial_block_split(eligible, TRANSFORM, 0.3, block_size_m=10_000.0)


def test_split_is_reproducible_for_a_fixed_seed():
    eligible = np.ones((100, 100), dtype=bool)
    a = S.spatial_block_split(eligible, TRANSFORM, 0.3, 100.0, seed=7)
    b = S.spatial_block_split(eligible, TRANSFORM, 0.3, 100.0, seed=7)
    c = S.spatial_block_split(eligible, TRANSFORM, 0.3, 100.0, seed=8)
    assert np.array_equal(a.test, b.test)
    assert not np.array_equal(a.test, c.test)


# --------------------------------------------------------------------------- #
# Folds, coverage, subsampling
# --------------------------------------------------------------------------- #


def test_block_folds_assign_every_eligible_pixel_to_exactly_one_fold():
    eligible = np.ones((100, 100), dtype=bool)
    folds = S.spatial_block_folds(eligible, TRANSFORM, n_folds=5, block_size_m=100.0)
    assert set(np.unique(folds).tolist()) == {0, 1, 2, 3, 4}
    blocks = S.block_ids(eligible.shape, TRANSFORM, 100.0)
    for block in np.unique(blocks):
        assert len(np.unique(folds[blocks == block])) == 1


def test_block_folds_mark_ineligible_pixels_as_minus_one():
    eligible = np.zeros((40, 40), dtype=bool)
    eligible[:20] = True
    folds = S.spatial_block_folds(eligible, TRANSFORM, 4, 100.0)
    assert (folds[20:] == -1).all()
    assert (folds[:20] >= 0).all()


def test_class_coverage_flags_a_class_missing_from_the_test_side():
    labels = np.ones((40, 40), dtype=np.uint8)
    labels[:5, :5] = 2  # a rare class confined to one corner
    split = S.Split(
        train=np.zeros((40, 40), dtype=bool), test=np.zeros((40, 40), dtype=bool),
        kind="t", detail={},
    )
    split.train[:20] = True
    split.test[20:] = True
    coverage = S.class_coverage(labels, split)
    assert coverage[2]["test"] == 0
    assert coverage[2]["train"] == 25


def test_subsample_caps_the_pixel_count_without_adding_any():
    mask = np.ones((100, 100), dtype=bool)
    out = S.subsample(mask, max_pixels=250, seed=0)
    assert out.sum() == 250
    assert (out & ~mask).sum() == 0


def test_subsample_is_a_noop_when_under_the_cap():
    mask = np.zeros((10, 10), dtype=bool)
    mask[:3] = True
    assert np.array_equal(S.subsample(mask, 1000), mask)
    assert np.array_equal(S.subsample(mask, None), mask)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def test_evaluate_keeps_a_never_predicted_class_in_the_report():
    """A class the model ignores must appear with zero recall, not disappear.

    Without an explicit label list scikit-learn infers classes from the data,
    so coniferous forest -- which this model almost never predicts -- would
    silently vanish from the confusion matrix instead of showing its failure.
    """
    y_true = np.array([1, 1, 2, 2, 3, 3])
    y_pred = np.array([1, 1, 2, 2, 1, 1])  # class 3 never predicted
    report = M.evaluate(y_true, y_pred, [1, 2, 3], ["a", "b", "c"])
    assert report.confusion.shape == (3, 3)
    assert next(r for r in report.per_class if r["class_id"] == 3)["recall"] == 0.0
    assert next(r for r in report.per_class if r["class_id"] == 3)["support"] == 2


def test_evaluate_perfect_prediction():
    y = np.array([1, 2, 3, 1, 2, 3])
    report = M.evaluate(y, y, [1, 2, 3], ["a", "b", "c"])
    assert report.overall_accuracy == 1.0
    assert report.macro_f1 == 1.0
    assert report.kappa == 1.0


def test_macro_f1_punishes_ignoring_a_rare_class_far_more_than_accuracy_does():
    """The reason per-class metrics lead the report."""
    y_true = np.array([1] * 98 + [2] * 2)
    y_pred = np.array([1] * 100)  # never predicts the rare class
    report = M.evaluate(y_true, y_pred, [1, 2], ["common", "rare"])
    assert report.overall_accuracy == pytest.approx(0.98)
    assert report.macro_f1 < 0.55


def test_accuracy_by_boundary_distance_bins_and_orders_correctly():
    y_true = np.array([1, 1, 1, 1])
    y_pred = np.array([2, 2, 1, 1])  # wrong near the edge, right far away
    dist = np.array([5.0, 10.0, 250.0, 500.0])
    rows = M.accuracy_by_boundary_distance(y_true, y_pred, dist)
    by_band = {r["min_distance_m"]: r for r in rows}
    assert by_band[0.0]["accuracy"] == 0.0
    assert by_band[200.0]["accuracy"] == 1.0
    assert by_band[400.0]["accuracy"] == 1.0
    assert sum(r["n_pixels"] for r in rows) == 4


def test_accuracy_by_boundary_distance_skips_empty_bins():
    rows = M.accuracy_by_boundary_distance(
        np.array([1]), np.array([1]), np.array([1000.0])
    )
    assert len(rows) == 1
    assert rows[0]["min_distance_m"] == 400.0


def test_compare_splits_reports_the_random_split_as_inflation():
    spatial = M.evaluate(np.array([1, 1, 2, 2]), np.array([1, 2, 2, 1]), [1, 2], ["a", "b"])
    random_ = M.evaluate(np.array([1, 1, 2, 2]), np.array([1, 1, 2, 2]), [1, 2], ["a", "b"])
    out = M.compare_splits(spatial, random_)
    assert out["inflation"]["overall_accuracy"] == pytest.approx(0.5)
    assert out["spatial_block_split"]["overall_accuracy"] == pytest.approx(0.5)


def test_report_text_renders_every_class():
    report = M.evaluate(np.array([1, 2]), np.array([1, 2]), [1, 2], ["alpha", "beta"])
    text = report.text()
    assert "alpha" in text and "beta" in text
    assert "Cohen's kappa" in text
