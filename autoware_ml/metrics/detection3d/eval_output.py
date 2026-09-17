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

from autoware_ml.dataclasses.models.detection3d.predictions import Detection3DSamplePredictions
from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.dataclasses.models.model_predictions import ModelPredictions


def detection_eval_output(
    predictions: Sequence[Detection3DSamplePredictions], batch_inputs_dict: Mapping[str, Any]
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


def multi_task_eval_output(
    multi_task_predictions: ModelPredictions, multi_task_batch_inputs: ModelBatchInputs
) -> dict[str, Any]:
    """
    Pair decoded predictions with ground truth for the detection metric.
    This function is a temporary interface between ModelPredictions, MultiTaskFeatures and
    detection_eval_output, and this will be removed once the detection metric is refactored to
    accept ModelPredictions and MultiTaskFeatures directly.

    Args:
        multi_task_predictions: ModelPredictions containing the decoded predictions.
        multi_task_batch_inputs: ModelBatchInputs containing the ground-truth boxes and labels.

    Returns:
        Flat eval-output dict consumed by the detection metric.
    """
    if multi_task_predictions.detection3d_predictions is None:
        raise ValueError(
            "ModelPredictions must contain detection3d_predictions for multi_task_eval_output."
        )

    if multi_task_batch_inputs.multi_task_gt_batch.detection3d_gt_batch is None:
        raise ValueError(
            "ModelBatchInputs must contain detection3d_gt_batch for multi_task_eval_output."
        )

    gt_detections = multi_task_batch_inputs.multi_task_gt_batch.detection3d_gt_batch
    valid = gt_detections.gt_valid_bboxes
    batch: dict[str, Any] = {
        "gt_boxes": [gt_detections.gt_bboxes_3d[i, : valid[i]] for i in range(len(valid))],
        "gt_labels": [gt_detections.gt_labels_3d[i, : valid[i]] for i in range(len(valid))],
        "gt_num_points": [
            gt_detections.gt_bboxes_num_points[i, : valid[i]] for i in range(len(valid))
        ],
    }
    # Per-frame evaluation metadata for the region and collision filters, when the dataset
    # attached it to the samples.
    frame_meta_batch = multi_task_batch_inputs.multi_task_gt_batch.frame_meta_batch
    if frame_meta_batch is not None:
        batch["ego2global"] = list(frame_meta_batch.ego2globals)
        batch["scene_token"] = list(frame_meta_batch.scene_tokens)

    return detection_eval_output(
        predictions=multi_task_predictions.detection3d_predictions, batch_inputs_dict=batch
    )
