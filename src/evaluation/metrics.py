"""Classification metrics for land cover.

Why not overall accuracy
------------------------
The classes here range from 29% of the AOI (broad-leaved forest) to 2%
(coniferous forest). A model that never predicts coniferous forest at all
loses barely two points of overall accuracy while being useless for the one
question a forestry user would ask it. Every report therefore leads with
per-class recall and macro-F1, and treats overall accuracy as a footnote.

Cohen's kappa is included because it is the convention in the remote-sensing
literature (Congalton's accuracy-assessment framework), so a reviewer from that
field will look for it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


@dataclass
class ClassificationReport:
    class_ids: list[int]
    class_names: list[str]
    confusion: np.ndarray  # rows = truth, cols = prediction
    per_class: list[dict]
    overall_accuracy: float
    macro_f1: float
    weighted_f1: float
    kappa: float
    n_test: int
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "n_test_pixels": self.n_test,
            "overall_accuracy": self.overall_accuracy,
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
            "cohen_kappa": self.kappa,
            "per_class": self.per_class,
            "confusion_matrix": self.confusion.tolist(),
            "class_ids": self.class_ids,
            "class_names": self.class_names,
            **self.extra,
        }

    def text(self) -> str:
        w = max(len(n) for n in self.class_names) + 2
        lines = [
            f"{'class':<{w}}{'precision':>10}{'recall':>9}{'f1':>8}{'support':>10}",
            "-" * (w + 37),
        ]
        for row in self.per_class:
            lines.append(
                f"{row['name']:<{w}}{row['precision']:>10.3f}{row['recall']:>9.3f}"
                f"{row['f1']:>8.3f}{row['support']:>10,}"
            )
        lines += [
            "-" * (w + 37),
            f"{'macro F1':<{w}}{self.macro_f1:>27.3f}",
            f"{'weighted F1':<{w}}{self.weighted_f1:>27.3f}",
            f"{'overall accuracy':<{w}}{self.overall_accuracy:>27.3f}",
            # Nested same-type quotes inside an f-string need Python 3.12; this
            # project supports 3.10.
            "{:<{w}}{:>27.3f}".format("Cohen's kappa", self.kappa, w=w),
            f"{'test pixels':<{w}}{self.n_test:>27,}",
        ]
        return "\n".join(lines)


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_ids: list[int],
    class_names: list[str],
) -> ClassificationReport:
    """Build a full report over a fixed, explicit class list.

    Passing ``labels=class_ids`` matters: without it scikit-learn infers the
    label set from the data, so a class the model never predicts silently
    vanishes from the confusion matrix instead of showing up as a row of zeros.
    """
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=class_ids, zero_division=0
    )
    per_class = [
        {
            "class_id": int(cid),
            "name": name,
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
            "support": int(s),
        }
        for cid, name, p, r, f, s in zip(
            class_ids, class_names, precision, recall, f1, support, strict=True
        )
    ]
    return ClassificationReport(
        class_ids=list(class_ids),
        class_names=list(class_names),
        confusion=confusion_matrix(y_true, y_pred, labels=class_ids),
        per_class=per_class,
        overall_accuracy=float((y_true == y_pred).mean()),
        macro_f1=float(f1_score(y_true, y_pred, labels=class_ids, average="macro",
                                zero_division=0)),
        weighted_f1=float(f1_score(y_true, y_pred, labels=class_ids, average="weighted",
                                   zero_division=0)),
        kappa=float(cohen_kappa_score(y_true, y_pred, labels=class_ids)),
        n_test=int(y_true.size),
    )


def accuracy_by_boundary_distance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    distance_m: np.ndarray,
    edges: tuple[float, ...] = (0, 20, 50, 100, 200, 400, np.inf),
) -> list[dict]:
    """Accuracy as a function of distance from the nearest label boundary.

    This is the sharpest available evidence about *label* quality rather than
    model quality. CORINE is photo-interpreted at 1:100 000, so its polygon
    edges are only good to ~100 m. If accuracy climbs steeply with distance
    from a boundary, much of the apparent error is the label being wrong, not
    the prediction -- and the deep interior accuracy is the fairer estimate of
    what the model actually learned.
    """
    rows = []
    correct = y_true == y_pred
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (distance_m >= lo) & (distance_m < hi)
        if not m.any():
            continue
        rows.append(
            {
                "min_distance_m": float(lo),
                "max_distance_m": None if np.isinf(hi) else float(hi),
                "n_pixels": int(m.sum()),
                "accuracy": float(correct[m].mean()),
            }
        )
    return rows


def compare_splits(spatial: ClassificationReport, random_: ClassificationReport) -> dict:
    """Quantify how much a random pixel split inflates the score."""
    return {
        "random_split": {
            "overall_accuracy": random_.overall_accuracy,
            "macro_f1": random_.macro_f1,
            "kappa": random_.kappa,
        },
        "spatial_block_split": {
            "overall_accuracy": spatial.overall_accuracy,
            "macro_f1": spatial.macro_f1,
            "kappa": spatial.kappa,
        },
        "inflation": {
            "overall_accuracy": random_.overall_accuracy - spatial.overall_accuracy,
            "macro_f1": random_.macro_f1 - spatial.macro_f1,
            "kappa": random_.kappa - spatial.kappa,
        },
        "note": (
            "The random-pixel split is reported only to show how much it "
            "overstates performance. The spatially-blocked figure is the one "
            "to quote."
        ),
    }
