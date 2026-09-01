"""Shared eval-output builder for the 3D segmentation suites.

Both segmentation suites consume one contract: ``seg_frames``, a list with one
entry per frame carrying that frame's points (``coord``, predicted/target labels,
per-class ``scores``) plus any per-frame metadata the configured filters need
(``ego2global``, ``scene_token``) and, for the cross-task partial detection
metric, the frame's detection GT boxes.

Metadata keys are copied from the batch when the dataset supplies them. A filter
or metric that needs a missing key fails loud in the suite with a message naming
it, so absence is never silent.
"""

from __future__ import annotations

from typing import Any

import torch
from jaxtyping import Float32, Int64
from torch import Tensor

from autoware_ml.datamodule.samples.batch import Batch

# Per-frame metadata passed through to each frame entry when the batch carries it.
_FRAME_META_KEYS = ("ego2global", "scene_token", "timestamp")
# Per-frame detection ground truth for the cross-task partial-detection metric.
_FRAME_BOX_KEYS = ("gt_boxes", "gt_labels")


def segmentation_frames_eval_output(
    coord: Float32[Tensor, "num_points 3"],
    pred_labels: Int64[Tensor, " num_points"],
    target_labels: Int64[Tensor, " num_points"],
    scores: Float32[Tensor, "num_points num_classes"],
    frame_ids: Int64[Tensor, " num_points"],
    num_frames: int,
    batch: Batch,
) -> dict[str, Any]:
    """Split batch-concatenated per-point tensors into the ``seg_frames`` list.

    Args:
        coord: ``(N, 3+)`` point coordinates (base_link).
        pred_labels: ``(N,)`` predicted class per point.
        target_labels: ``(N,)`` ground-truth class per point.
        scores: ``(N, C)`` per-class scores per point.
        frame_ids: ``(N,)`` frame index of every point.
        num_frames: Number of frames in the batch.
        batch: Typed batch, per-frame metadata and detection GT are copied into each frame
            entry when present.

    Returns:
        ``{"seg_frames": [...]}`` with one entry per frame.
    """
    per_frame_values = {
        key: getattr(batch, key)
        for key in (*_FRAME_META_KEYS, *_FRAME_BOX_KEYS)
        if getattr(batch, key) is not None
    }
    for key, values in per_frame_values.items():
        if len(values) != num_frames:
            raise ValueError(
                f"batch.{key} has {len(values)} entries for {num_frames} frames, "
                "per-frame metadata must align one-to-one or points get another frame's context."
            )
    # frame_ids come from cumulative point offsets, so they are non-decreasing.
    # One split then replaces a full O(frames x N) equality scan per frame.
    if frame_ids.numel() and not bool((frame_ids[1:] >= frame_ids[:-1]).all()):
        raise ValueError("frame_ids must be non-decreasing (per-frame point blocks).")
    if frame_ids.numel() and int(frame_ids.max()) >= num_frames:
        raise ValueError(
            f"frame_ids reach {int(frame_ids.max())} for {num_frames} frames, a malformed "
            "offset would silently drop the trailing frames' points."
        )
    counts = torch.bincount(frame_ids, minlength=num_frames).tolist()
    coord_split = torch.split(coord, counts)
    pred_split = torch.split(pred_labels, counts)
    target_split = torch.split(target_labels, counts)
    scores_split = torch.split(scores, counts)

    frames: list[dict[str, Any]] = []
    for frame_index in range(num_frames):
        frame: dict[str, Any] = {
            "coord": coord_split[frame_index],
            "pred": pred_split[frame_index],
            "target": target_split[frame_index],
            "scores": scores_split[frame_index],
        }
        for key in _FRAME_META_KEYS:
            if key in per_frame_values:
                frame[key] = per_frame_values[key][frame_index]
        for key in _FRAME_BOX_KEYS:
            if key in per_frame_values:
                frame["gt_boxes" if key == "gt_boxes" else "gt_box_labels"] = per_frame_values[key][
                    frame_index
                ]
        frames.append(frame)
    return {"seg_frames": frames}


def concat_frame_ids(
    offset: Int64[Tensor, " batch_size"], point_to_batch: Int64[Tensor, " num_points"]
) -> Int64[Tensor, " num_points"]:
    """Frame index per point from the batch ``offset`` (inclusive cumulative lengths).

    ``point_to_batch`` maps each point to its position in the batch-concatenated
    primary space: the point's own index for directly-concatenated points, or
    the ``inverse`` mapping for original-resolution points scattered from the
    sampled space.

    Args:
        offset: Inclusive cumulative frame lengths ``(B,)``.
        point_to_batch: Point to batch-concatenated position mapping.

    Returns:
        The frame index per point.
    """
    return torch.searchsorted(offset.to(point_to_batch.device), point_to_batch, right=True)
