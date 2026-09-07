"""Shared eval output builder for 3D detection models.

Every detection model decodes its head into per sample typed predictions and pairs them with
the ground truth boxes and labels of the batch. This helper builds the flat eval output dict
that Detection3DMetricSuite reads, so each model's build_eval_output is a one line delegation.
The metric side contract stays dict based, the conversion from the typed predictions happens
here.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from autoware_ml.datamodule.samples.batch import Batch
from autoware_ml.models.detection3d.outputs import Detection3DPrediction


def detection_eval_output(
    predictions: Sequence[Detection3DPrediction], batch: Batch
) -> dict[str, Any]:
    """Pair decoded predictions with ground truth for the detection metric.

    Args:
        predictions: Per sample typed predictions as returned by the head's predict.
        batch: The typed batch holding the ground truth boxes and labels.

    Returns:
        Flat eval output dict consumed by the detection metric.
    """
    if batch.gt_boxes is None or batch.gt_labels is None:
        raise ValueError("The detection eval output requires ground truth boxes and labels.")
    eval_out: dict[str, Any] = {
        "predictions": [
            {
                "bboxes_3d": prediction.bboxes_3d,
                "scores_3d": prediction.scores_3d,
                "labels_3d": prediction.labels_3d,
            }
            for prediction in predictions
        ],
        "gt_boxes": list(batch.gt_boxes),
        "gt_labels": list(batch.gt_labels),
    }
    # Per frame evaluation metadata, copied through when the dataset supplies it. Region and
    # collision filters need the ego pose and scene token. A configured filter that needs a
    # missing key fails loud in the suite naming it, so absence is never silent.
    if batch.gt_num_points is not None:
        eval_out["gt_num_points"] = list(batch.gt_num_points)
    if batch.ego2global is not None:
        eval_out["ego2global"] = list(batch.ego2global)
    if batch.scene_token is not None:
        eval_out["scene_token"] = list(batch.scene_token)
    return eval_out
