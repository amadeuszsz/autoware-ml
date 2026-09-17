"""Shared eval output builder for 3D detection models.

Every detection model decodes its head into per sample typed predictions and pairs them with
the ground truth boxes and labels of the batch. This helper builds the flat eval output dict
that Detection3DMetricSuite reads, so each model's build_eval_output is a one line delegation.
The metric side contract stays dict based, the conversion from the typed predictions happens
here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from autoware_ml.models.detection3d.outputs import Detection3DPrediction


def detection_eval_output(
    predictions: Sequence[Detection3DPrediction], batch_inputs_dict: Mapping[str, Any]
) -> dict[str, Any]:
    """Pair decoded predictions with ground truth for the detection metric.

    Args:
        predictions: Per sample typed predictions as returned by the head's predict.
        batch_inputs_dict: Model inputs holding the ground truth boxes and labels.

    Returns:
        Flat eval output dict consumed by the detection metric.
    """
    if "gt_boxes" not in batch_inputs_dict or "gt_labels" not in batch_inputs_dict:
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
        "gt_boxes": list(batch_inputs_dict["gt_boxes"]),
        "gt_labels": list(batch_inputs_dict["gt_labels"]),
    }
    # Per frame evaluation metadata, copied through when the dataset supplies it. Region and
    # collision filters need the ego pose and scene token. A configured filter that needs a
    # missing key fails loud in the suite naming it, so absence is never silent.
    for key in ("gt_num_points", "ego2global", "scene_token"):
        if key in batch_inputs_dict:
            eval_out[key] = list(batch_inputs_dict[key])
    return eval_out
