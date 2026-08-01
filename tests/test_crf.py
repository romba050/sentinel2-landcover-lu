"""Tests for the mean-field CRF and the OOF fold extension."""

from __future__ import annotations

import numpy as np
import pytest
from rasterio.transform import from_origin

from src.evaluation.spatial_split import spatial_block_folds
from src.postprocessing import crf

TRANSFORM = from_origin(700000.0, 5500000.0, 10.0, 10.0)


# --------------------------------------------------------------------------- #
# shift
# --------------------------------------------------------------------------- #


def test_shift_brings_the_neighbour_value_onto_each_pixel():
    a = np.arange(9, dtype=float).reshape(3, 3)
    # Neighbour at offset (0, 1) is the pixel to the RIGHT... the value that
    # lands on (y, x) must be a[y, x - 1]? No: out[y, x] = a[y - dy, x - dx].
    out = crf.shift(a, 0, 1)
    assert out[1, 1] == a[1, 0]
    out = crf.shift(a, 1, 0)
    assert out[1, 1] == a[0, 1]


def test_shift_zero_fills_the_border():
    a = np.ones((3, 3))
    assert crf.shift(a, 1, 0)[0].sum() == 0
    assert crf.shift(a, -1, -1)[-1].sum() == 0
    assert crf.shift(a, -1, -1)[:, -1].sum() == 0


def test_shift_works_on_leading_class_axis():
    a = np.random.default_rng(0).random((4, 5, 6))
    out = crf.shift(a, 0, 1)
    assert out.shape == a.shape
    assert np.allclose(out[:, :, 1:], a[:, :, :-1])


# --------------------------------------------------------------------------- #
# kernels
# --------------------------------------------------------------------------- #


def _flat_features(h=8, w=8, n=2, value=0.0):
    return np.full((n, h, w), value, dtype=np.float32)


def test_kernel_is_high_for_similar_and_low_for_different_features():
    f = _flat_features()
    f[:, :, 4:] = 5.0  # a hard vertical feature edge
    valid = np.ones((8, 8), dtype=bool)
    k_right = crf.contrast_kernels(f, valid, sigma=1.0)[crf.OFFSETS.index((0, 1))]
    assert k_right[2, 2] == pytest.approx(1.0)  # inside the flat half
    assert k_right[2, 4] < 1e-4  # across the edge


def test_kernel_diagonals_are_downweighted_by_distance():
    f = _flat_features()
    valid = np.ones((8, 8), dtype=bool)
    kernels = dict(zip(crf.OFFSETS, crf.contrast_kernels(f, valid, 1.0), strict=True))
    assert kernels[(0, 1)][4, 4] == pytest.approx(1.0)
    assert kernels[(1, 1)][4, 4] == pytest.approx(1 / np.sqrt(2))


def test_kernel_is_zero_when_either_side_is_invalid():
    f = _flat_features()
    valid = np.ones((8, 8), dtype=bool)
    valid[3, 3] = False
    kernels = dict(zip(crf.OFFSETS, crf.contrast_kernels(f, valid, 1.0), strict=True))
    # the invalid pixel neither receives nor supplies weight
    assert kernels[(0, 1)][3, 3] == 0.0  # message into (3,3) from its left
    assert kernels[(0, 1)][3, 4] == 0.0  # message into (3,4) from (3,3)


def test_uniform_kernels_ignore_features_but_respect_validity_and_distance():
    valid = np.ones((6, 6), dtype=bool)
    valid[2, 2] = False
    kernels = dict(zip(crf.OFFSETS, crf.uniform_kernels(valid), strict=True))
    assert kernels[(0, 1)][4, 4] == pytest.approx(1.0)
    assert kernels[(1, 1)][4, 4] == pytest.approx(1 / np.sqrt(2))
    assert kernels[(0, 1)][2, 2] == 0.0
    assert kernels[(0, 1)][2, 3] == 0.0  # its neighbour across the hole


def test_estimate_sigma_matches_a_known_constant_gradient():
    h, w = 6, 6
    f = np.tile(np.arange(w, dtype=np.float32) * 2.0, (h, 1))[None]
    valid = np.ones((h, w), dtype=bool)
    # horizontal steps are all 2.0, vertical steps all 0; the pooled median
    # interpolates between the two equal-sized populations -> 1.0
    assert crf.estimate_sigma(f, valid) == pytest.approx(1.0)
    # a pure gradient in both directions gives exactly that step
    f2 = (np.add.outer(np.arange(h), np.arange(w)).astype(np.float32) * 3.0)[None]
    assert crf.estimate_sigma(f2, valid) == pytest.approx(3.0)


# --------------------------------------------------------------------------- #
# mean-field behaviour
# --------------------------------------------------------------------------- #


def _uniform_setup(h=9, w=9):
    features = _flat_features(h, w)
    valid = np.ones((h, w), dtype=bool)
    kernels = crf.contrast_kernels(features, valid, sigma=1.0)
    return kernels, valid


def test_theta_zero_reproduces_the_unary_argmax():
    rng = np.random.default_rng(1)
    probs = rng.dirichlet(np.ones(3), size=(9, 9)).transpose(2, 0, 1).astype(np.float32)
    kernels, valid = _uniform_setup()
    Q, _ = crf.mean_field(probs, kernels, theta=0.0, valid=valid, n_iters=5)
    assert np.array_equal(np.argmax(Q, 0), np.argmax(probs, 0))


def test_q_stays_a_probability_distribution():
    rng = np.random.default_rng(2)
    probs = rng.dirichlet(np.ones(4), size=(9, 9)).transpose(2, 0, 1).astype(np.float32)
    kernels, valid = _uniform_setup()
    Q, _ = crf.mean_field(probs, kernels, theta=1.0, valid=valid, n_iters=6)
    assert (Q >= 0).all()
    assert np.allclose(Q[:, valid].sum(axis=0), 1.0, atol=1e-5)


def test_isolated_flipped_pixel_is_absorbed():
    """The denoising case: one weakly-confident dissenting pixel in a sea of
    agreement gets pulled to the consensus."""
    probs = np.zeros((2, 9, 9), dtype=np.float32)
    probs[0], probs[1] = 0.9, 0.1
    probs[0, 4, 4], probs[1, 4, 4] = 0.4, 0.6  # the speck
    kernels, valid = _uniform_setup()
    Q, _ = crf.mean_field(probs, kernels, theta=0.8, valid=valid, n_iters=8)
    assert np.argmax(Q[:, 4, 4]) == 0


def test_confident_dissent_with_feature_contrast_survives():
    """The anti-bulldozer case: a pixel that disagrees AND sits on a real
    feature edge (e.g. a pond in a field) must keep its label."""
    probs = np.zeros((2, 9, 9), dtype=np.float32)
    probs[0], probs[1] = 0.9, 0.1
    probs[0, 4, 4], probs[1, 4, 4] = 0.05, 0.95
    features = _flat_features(9, 9)
    features[:, 4, 4] = 8.0  # spectrally nothing like its neighbours
    valid = np.ones((9, 9), dtype=bool)
    kernels = crf.contrast_kernels(features, valid, sigma=1.0)
    Q, _ = crf.mean_field(probs, kernels, theta=2.0, valid=valid, n_iters=8)
    assert np.argmax(Q[:, 4, 4]) == 1


def test_genuine_edge_between_two_regions_survives_when_contrast_backed():
    probs = np.zeros((2, 8, 8), dtype=np.float32)
    probs[0, :, :4], probs[1, :, :4] = 0.8, 0.2
    probs[0, :, 4:], probs[1, :, 4:] = 0.2, 0.8
    features = _flat_features(8, 8)
    features[:, :, 4:] = 6.0  # the label edge coincides with a feature edge
    valid = np.ones((8, 8), dtype=bool)
    kernels = crf.contrast_kernels(features, valid, sigma=1.0)
    Q, _ = crf.mean_field(probs, kernels, theta=2.0, valid=valid, n_iters=8)
    labels = np.argmax(Q, 0)
    assert (labels[:, :4] == 0).all()
    assert (labels[:, 4:] == 1).all()


def test_convergence_delta_shrinks():
    rng = np.random.default_rng(3)
    probs = rng.dirichlet(np.ones(3), size=(12, 12)).transpose(2, 0, 1).astype(np.float32)
    kernels, valid = _uniform_setup(12, 12)
    _, deltas = crf.mean_field(probs, kernels, theta=0.5, valid=valid, n_iters=10)
    assert deltas[-1] < deltas[0]
    assert deltas[-1] < 1e-2


def test_map_labels_uses_class_ids_and_zeroes_invalid():
    Q = np.zeros((2, 3, 3), dtype=np.float32)
    Q[0], Q[1] = 0.3, 0.7
    valid = np.ones((3, 3), dtype=bool)
    valid[0, 0] = False
    out = crf.map_labels(Q, class_ids=[4, 6], valid=valid)
    assert out[1, 1] == 6
    assert out[0, 0] == 0


# --------------------------------------------------------------------------- #
# fold extension for OOF
# --------------------------------------------------------------------------- #


def test_folds_extend_to_valid_but_unlabelled_blocks():
    eligible = np.zeros((60, 60), dtype=bool)
    eligible[:30] = True  # labels only in the top half
    valid = np.ones((60, 60), dtype=bool)  # but the whole scene is cloud-free
    folds = spatial_block_folds(eligible, TRANSFORM, n_folds=3, block_size_m=100.0,
                                seed=0, assign=valid)
    assert (folds >= 0).all()  # every valid pixel got a fold


def test_extension_does_not_change_the_eligible_assignment():
    eligible = np.zeros((60, 60), dtype=bool)
    eligible[:30] = True
    valid = np.ones((60, 60), dtype=bool)
    base = spatial_block_folds(eligible, TRANSFORM, 3, 100.0, seed=0)
    ext = spatial_block_folds(eligible, TRANSFORM, 3, 100.0, seed=0, assign=valid)
    assert np.array_equal(base[eligible], ext[eligible])


def test_default_behaviour_unchanged_without_assign():
    eligible = np.ones((40, 40), dtype=bool)
    folds = spatial_block_folds(eligible, TRANSFORM, 4, 100.0, seed=1)
    assert set(np.unique(folds).tolist()) == {0, 1, 2, 3}
