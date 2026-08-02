"""Mean-field inference for a grid CRF over class probabilities.

This is my MSc. thesis method (mean-field networks for pixel-wise segmentation)
transplanted from retinal imagery to satellite imagery, with the neural unary
swapped for the Random Forest's out-of-fold ``predict_proba``.

The model
---------
Energy over the label field x, 8-neighbour grid N, Potts pairwise term:

    E(x) = sum_i  psi_u(x_i)  +  sum_(i,j) in N  psi_p(x_i, x_j)

    psi_u(x_i = l)  = -log P_RF(l | i)              # unary: the existing RF
    psi_p(x_i, x_j) =  w_ij * [x_i != x_j]          # Potts
    w_ij = (theta / d_ij) * exp(-||f_i - f_j||^2 / (2 sigma^2))

``f`` is a standardised spectral feature vector (the SWIR bands -- feature
importance shows B11/B12/NBR2 carry the signal here, NDVI does not). The
contrast term is what stops the smoothing from bulldozing real land-cover
edges: where the imagery changes sharply, w_ij collapses and the CRF leaves
the boundary alone. ``d_ij`` is the pixel distance (1 or sqrt(2)), so diagonal
neighbours count slightly less than edge neighbours.

Mean-field update
-----------------
Approximate the posterior with a fully factorised Q(x) = prod_i Q_i(x_i).
Minimising KL(Q || P) gives the coordinate update

    log Q_i(l) = -psi_u(l) - sum_j in N(i)  sum_l'  psi_p(l, l') Q_j(l')
               = -psi_u(l) - sum_j in N(i)  w_ij (1 - Q_j(l))     # Potts
               PROPORTIONAL TO  -psi_u(l) + sum_j w_ij Q_j(l)

The dropped term sum_j w_ij does not depend on l, so it cancels in the softmax
normalisation -- which is why the update below is just "log unary plus a
weighted sum of neighbour beliefs, then softmax". Iterate a handful of times;
each sweep is eight array shifts, fully vectorised, no loop over pixels.

Hand-rolled rather than pydensecrf on purpose: ~90 lines that can be derived
at a whiteboard beat an unmaintained Cython dependency.
"""

from __future__ import annotations

import numpy as np

#: 8-neighbour offsets (dy, dx) and their pixel distances.
OFFSETS: tuple[tuple[int, int], ...] = (
    (-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1),
)
_DIST = {o: float(np.hypot(*o)) for o in OFFSETS}


def shift(a: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Shift the last two axes by (dy, dx), zero-filling the vacated border.

    ``out[..., y, x] = a[..., y - dy, x - dx]`` -- i.e. the value of the
    neighbour at offset (dy, dx) is brought onto each pixel.
    """
    out = np.zeros_like(a)
    ys = slice(max(dy, 0), a.shape[-2] + min(dy, 0))
    xs = slice(max(dx, 0), a.shape[-1] + min(dx, 0))
    ys_src = slice(max(-dy, 0), a.shape[-2] + min(-dy, 0))
    xs_src = slice(max(-dx, 0), a.shape[-1] + min(-dx, 0))
    out[..., ys, xs] = a[..., ys_src, xs_src]
    return out


def standardise(features: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Zero-mean unit-variance per band over valid pixels; invalid -> 0."""
    out = np.zeros_like(features, dtype=np.float32)
    for b in range(features.shape[0]):
        v = features[b, valid]
        out[b] = (features[b] - v.mean()) / max(v.std(), 1e-12)
    out[:, ~valid] = 0.0
    return out


def estimate_sigma(features: np.ndarray, valid: np.ndarray) -> float:
    """Data-driven contrast scale: the median 4-neighbour feature distance.

    With this sigma, a typical within-parcel step keeps w near theta
    (exp(-0.5) at exactly the median) while a land-cover edge several times
    stronger than the median suppresses w toward zero. It removes one knob
    that would otherwise need hand-tuning per scene.
    """
    dists = []
    for dy, dx in ((0, 1), (1, 0)):
        both = valid & shift(valid, dy, dx).astype(bool)
        diff = features - shift(features, dy, dx)
        d = np.sqrt((diff[:, both] ** 2).sum(axis=0))
        dists.append(d)
    return float(np.median(np.concatenate(dists)))


def contrast_kernels(features: np.ndarray, valid: np.ndarray, sigma: float) -> list[np.ndarray]:
    """Per-offset weight maps for theta = 1 (scale by theta at use time).

    Pairs involving an invalid pixel get weight zero, so masked ground neither
    sends nor receives messages.
    """
    kernels = []
    for off in OFFSETS:
        both = valid & shift(valid.astype(np.uint8), *off).astype(bool)
        diff = features - shift(features, *off)
        d2 = (diff * diff).sum(axis=0)
        k = np.exp(-d2 / (2.0 * sigma * sigma)).astype(np.float32) / _DIST[off]
        k[~both] = 0.0
        kernels.append(k)
    return kernels


def uniform_kernels(valid: np.ndarray) -> list[np.ndarray]:
    """Contrast-ablated weights: every valid neighbour pair gets 1 / d_ij.

    Exists to *demonstrate* why the contrast term matters rather than assert
    it: run mean-field with these at the same theta and the smoothing has no
    way to tell a road from noise -- the bulldozer case.
    """
    kernels = []
    for off in OFFSETS:
        both = valid & shift(valid.astype(np.uint8), *off).astype(bool)
        k = np.full(valid.shape, 1.0 / _DIST[off], dtype=np.float32)
        k[~both] = 0.0
        kernels.append(k)
    return kernels


def softmax(logits: np.ndarray, axis: int = 0) -> np.ndarray:
    z = logits - logits.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def mean_field(
    probs: np.ndarray,
    kernels: list[np.ndarray],
    theta: float,
    valid: np.ndarray,
    n_iters: int = 8,
    eps: float = 1e-6,
) -> tuple[np.ndarray, list[float]]:
    """Run mean-field updates; returns (Q, per-iteration mean |dQ|).

    ``probs`` is the (C, H, W) unary probability stack, ``kernels`` the output
    of :func:`contrast_kernels`. theta = 0 reproduces the unary argmax exactly.
    """
    log_unary = np.log(np.clip(probs, eps, 1.0))
    Q = probs.astype(np.float32).copy()
    Q[:, ~valid] = 0.0
    deltas: list[float] = []

    for _ in range(n_iters):
        message = np.zeros_like(Q)
        for off, k in zip(OFFSETS, kernels, strict=True):
            message += (theta * k) * shift(Q, *off)
        Q_new = softmax(log_unary + message, axis=0)
        Q_new[:, ~valid] = 0.0
        deltas.append(float(np.abs(Q_new - Q)[:, valid].mean()))
        Q = Q_new
    return Q, deltas


def map_labels(Q: np.ndarray, class_ids: list[int], valid: np.ndarray) -> np.ndarray:
    """Argmax decoding of Q back to class-id space; invalid pixels -> 0."""
    out = np.zeros(Q.shape[1:], dtype=np.uint8)
    out[valid] = np.asarray(class_ids, dtype=np.uint8)[np.argmax(Q[:, valid], axis=0)]
    return out
