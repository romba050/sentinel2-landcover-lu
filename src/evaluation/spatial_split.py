"""Spatially-blocked train/test splitting.

The problem with a random pixel split
-------------------------------------
Land cover is strongly spatially autocorrelated. Two adjacent 10 m pixels are
usually the same class, have near-identical reflectance, and -- because CORINE
polygons are at least 25 ha -- come from the *same label polygon*.

Split those pixels at random and almost every test pixel has a near-duplicate
of itself in the training set. The model does not need to learn what a forest
looks like; it only needs to memorise the training pixels and interpolate. The
reported accuracy then measures interpolation within known polygons, not
generalisation to new ground, and is optimistic by a wide margin.

Splitting by spatial blocks fixes this: whole contiguous tiles go entirely to
train or entirely to test, so no test pixel has a training neighbour except
along block edges. The resulting score answers the question a user actually
cares about -- "how well does this work over ground the model has never seen?"

This module provides both splits so the difference can be measured rather than
asserted. Quantifying that gap is the point.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from rasterio.transform import Affine

log = logging.getLogger(__name__)

#: Block edge length. Must comfortably exceed the spatial autocorrelation range
#: of the labels. CORINE's 25 ha minimum mapping unit means a polygon is at
#: least ~500 m across, so 2 km blocks put a clear margin between train and
#: test ground while still leaving ~100 blocks in a 21 km AOI.
DEFAULT_BLOCK_SIZE_M = 2000.0


@dataclass(frozen=True)
class Split:
    """A boolean train/test partition over a raster."""

    train: np.ndarray  # bool, True where the pixel is a training sample
    test: np.ndarray  # bool, True where the pixel is a test sample
    kind: str  # "spatial_blocks" or "random_pixels"
    detail: dict

    def summary(self) -> str:
        n = self.train.sum() + self.test.sum()
        return (
            f"{self.kind}: {self.train.sum():,} train / {self.test.sum():,} test "
            f"({self.test.sum() / max(n, 1):.1%} held out)"
        )


def block_ids(shape: tuple[int, int], transform: Affine, block_size_m: float) -> np.ndarray:
    """Label every pixel with the id of the square spatial block it falls in.

    Blocks are anchored to the raster origin, so the same AOI always yields the
    same blocks regardless of how the array was sliced.
    """
    height, width = shape
    px = abs(transform.a)
    block_px = max(1, int(round(block_size_m / px)))
    rows = np.arange(height) // block_px
    cols = np.arange(width) // block_px
    n_block_cols = int(cols.max()) + 1
    return (rows[:, None] * n_block_cols + cols[None, :]).astype(np.int32)


def spatial_block_split(
    eligible: np.ndarray,
    transform: Affine,
    test_fraction: float = 0.3,
    block_size_m: float = DEFAULT_BLOCK_SIZE_M,
    seed: int = 42,
) -> Split:
    """Assign whole blocks to train or test.

    ``eligible`` marks pixels that are both cloud-free and labelled. Blocks
    containing no eligible pixel are ignored so they cannot skew the ratio.
    """
    blocks = block_ids(eligible.shape, transform, block_size_m)
    present = np.unique(blocks[eligible])
    if present.size < 4:
        raise ValueError(
            f"Only {present.size} usable blocks at {block_size_m:.0f} m; "
            "reduce --block-size or enlarge the AOI."
        )

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(present)
    n_test = max(1, int(round(test_fraction * shuffled.size)))
    test_blocks = set(shuffled[:n_test].tolist())

    is_test = np.isin(blocks, list(test_blocks))
    split = Split(
        train=eligible & ~is_test,
        test=eligible & is_test,
        kind="spatial_blocks",
        detail={
            "block_size_m": block_size_m,
            "n_blocks": int(present.size),
            "n_test_blocks": int(n_test),
            "seed": seed,
        },
    )
    log.info("%s across %d blocks of %.0f m", split.summary(), present.size, block_size_m)
    return split


def random_pixel_split(
    eligible: np.ndarray, test_fraction: float = 0.3, seed: int = 42
) -> Split:
    """The naive split, provided *only* as a baseline to expose its optimism."""
    rng = np.random.default_rng(seed)
    draw = rng.random(eligible.shape) < test_fraction
    split = Split(
        train=eligible & ~draw,
        test=eligible & draw,
        kind="random_pixels",
        detail={"seed": seed, "test_fraction": test_fraction},
    )
    log.info("%s (leaky: neighbouring pixels land on both sides)", split.summary())
    return split


def spatial_block_folds(
    eligible: np.ndarray,
    transform: Affine,
    n_folds: int = 5,
    block_size_m: float = DEFAULT_BLOCK_SIZE_M,
    seed: int = 42,
    assign: np.ndarray | None = None,
) -> np.ndarray:
    """Fold index per pixel for spatially-blocked K-fold CV (-1 = unassigned).

    ``assign`` optionally widens *which pixels receive a fold index* beyond the
    eligible ones -- e.g. pass the cloud-free mask so that valid-but-unlabelled
    pixels also get an out-of-fold prediction. Fold balance is still decided by
    the blocks that contain eligible pixels; blocks holding only ``assign``
    pixels are appended round-robin afterwards, so they cannot skew the
    train/test balance of the labelled data.
    """
    blocks = block_ids(eligible.shape, transform, block_size_m)
    present = np.unique(blocks[eligible])
    rng = np.random.default_rng(seed)
    order = rng.permutation(present).tolist()
    if assign is not None:
        extra = np.setdiff1d(np.unique(blocks[assign]), present)
        order += rng.permutation(extra).tolist()

    fold_of_block = {b: i % n_folds for i, b in enumerate(order)}
    target = eligible if assign is None else (eligible | assign)
    folds = np.full(eligible.shape, -1, dtype=np.int8)
    for block, fold in fold_of_block.items():
        folds[target & (blocks == block)] = fold
    return folds


def class_coverage(labels: np.ndarray, split: Split) -> dict[int, dict[str, int]]:
    """Per-class pixel counts on each side of the split.

    A blocked split can leave a rare class entirely inside the training half,
    in which case its test metrics are undefined and must not be reported as
    zero. This is what makes the check worth running rather than assuming.
    """
    coverage: dict[int, dict[str, int]] = {}
    for cls in np.unique(labels[split.train | split.test]):
        coverage[int(cls)] = {
            "train": int((labels[split.train] == cls).sum()),
            "test": int((labels[split.test] == cls).sum()),
        }
    missing = [c for c, v in coverage.items() if v["test"] == 0 or v["train"] == 0]
    if missing:
        log.warning("Classes absent from one side of the split: %s", missing)
    return coverage


def subsample(
    mask: np.ndarray, max_pixels: int | None, seed: int = 42
) -> np.ndarray:
    """Thin a pixel mask to at most ``max_pixels``, uniformly at random.

    Uniform, not class-stratified: stratifying would silently rewrite the class
    priors, and a Random Forest trained on rebalanced priors reports accuracies
    that do not transfer back to the real landscape.
    """
    if max_pixels is None:
        return mask
    idx = np.flatnonzero(mask)
    if idx.size <= max_pixels:
        return mask
    rng = np.random.default_rng(seed)
    keep = rng.choice(idx, size=max_pixels, replace=False)
    out = np.zeros(mask.size, dtype=bool)
    out[keep] = True
    return out.reshape(mask.shape)
