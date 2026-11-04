# Copyright 2026 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Online mapping metrics: average precision of polylines matched by chamfer distance.

Predictions of a class are matched to the ground truth of the class in score order over the
whole evaluation, a match needing a chamfer distance within the threshold. Polylines are densified
along their length before the distance (Liao et al., MapTR, 2023).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from autoware_ml.metrics.base import EvalStage, Metric, MetricFilter, MetricRange, MetricSuite
from autoware_ml.metrics.eval_output import EvalOutput, MetricRequirements
from autoware_ml.metrics.report import MetricReport


@dataclass(frozen=True)
class MapFrame:
    """Predicted and ground truth polylines of one frame, points in the frame.

    Attributes:
        pred_points: Predicted polylines ``(P, N, 2)``.
        pred_scores: Prediction scores ``(P,)``.
        pred_labels: Predicted classes ``(P,)``.
        gt_points: Ground truth polylines ``(G, N, 2)``.
        gt_labels: Ground truth classes ``(G,)``.
    """

    pred_points: np.ndarray
    pred_scores: np.ndarray
    pred_labels: np.ndarray
    gt_points: np.ndarray
    gt_labels: np.ndarray


FRAME_FIELDS = ("pred_points", "pred_scores", "pred_labels", "gt_points", "gt_labels")


@dataclass(frozen=True)
class MapState:
    """Frames of the evaluation and the settings of the metrics."""

    frames: list[MapFrame]
    class_names: tuple[str, ...]
    thresholds: tuple[float, ...]
    num_dense_points: int


def densify(polylines: np.ndarray, num_points: int) -> np.ndarray:
    """Resample polylines at equal steps along their length.

    Args:
        polylines: Polylines ``(K, N, 2)``.
        num_points: Points of every resampled polyline.

    Returns:
        Resampled polylines ``(K, num_points, 2)``.
    """
    segments = np.linalg.norm(np.diff(polylines, axis=1), axis=-1)
    along = np.concatenate([np.zeros((len(polylines), 1)), np.cumsum(segments, axis=1)], axis=1)
    targets = np.linspace(0.0, 1.0, num_points)[None] * along[:, -1:]
    return np.stack(
        [
            np.stack([np.interp(target, distance, line[:, axis]) for axis in range(2)], axis=-1)
            for target, distance, line in zip(targets, along, polylines, strict=True)
        ]
    ).reshape(len(polylines), num_points, 2)


def chamfer_distances(pred: np.ndarray, gt: np.ndarray, chunk: int = 16) -> np.ndarray:
    """Symmetric chamfer distance of every prediction to every ground truth polyline.

    Args:
        pred: Dense predicted polylines ``(P, D, 2)``.
        gt: Dense ground truth polylines ``(G, D, 2)``.
        chunk: Predictions compared at once, bounding the memory of the point distances.

    Returns:
        Mean of both directional mean nearest point distances ``(P, G)``.
    """
    result = np.zeros((len(pred), len(gt)), dtype=np.float32)
    pred, gt = pred.astype(np.float32), gt.astype(np.float32)
    for begin in range(0, len(pred), chunk):
        block = pred[begin : begin + chunk]
        distance = np.linalg.norm(block[:, None, :, None] - gt[None, :, None], axis=-1)
        result[begin : begin + chunk] = 0.5 * (
            distance.min(axis=3).mean(axis=2) + distance.min(axis=2).mean(axis=2)
        )
    return result


def average_precision(scores: np.ndarray, true_positive: np.ndarray, num_gt: int) -> float:
    """Area under the precision envelope over recall.

    Args:
        scores: Score of every prediction.
        true_positive: Whether every prediction matched.
        num_gt: Number of ground truth elements.

    Returns:
        The average precision.
    """
    order = np.argsort(-scores, kind="stable")
    hits = np.cumsum(true_positive[order])
    recall = np.concatenate([[0.0], hits / num_gt])
    precision = np.concatenate([[1.0], hits / np.arange(1, len(order) + 1)])
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    return float(np.sum(np.diff(recall) * envelope[1:]))


class MapVectorAP(Metric[MapState]):
    """Average precision per class and threshold, their means per class and over the classes."""

    def evaluate(self, state: MapState, stage: EvalStage) -> MetricReport:
        """Match the predictions of every class by chamfer distance and integrate precision.

        Args:
            state: Frames of the evaluation.
            stage: Evaluation stage, unused.

        Returns:
            The values per class and threshold, per class and their mean over the classes.
        """
        report = MetricReport()
        class_means = []
        for label, name in enumerate(state.class_names):
            scores, distances, num_gt = [], [], 0
            for frame in state.frames:
                pred = frame.pred_labels == label
                gt = frame.gt_labels == label
                num_gt += int(gt.sum())
                scores.append(frame.pred_scores[pred])
                distances.append(
                    chamfer_distances(
                        densify(frame.pred_points[pred], state.num_dense_points),
                        densify(frame.gt_points[gt], state.num_dense_points),
                    )
                    if pred.any() and gt.any()
                    else np.full((int(pred.sum()), int(gt.sum())), np.inf)
                )
            if num_gt == 0:
                continue
            values = []
            for threshold in state.thresholds:
                hits = [
                    greedy_hits(frame_scores, frame_distances, threshold)
                    for frame_scores, frame_distances in zip(scores, distances, strict=True)
                ]
                value = average_precision(np.concatenate(scores), np.concatenate(hits), num_gt)
                report.record(f"AP_{name}_{threshold}m", value)
                values.append(value)
            class_means.append(float(np.mean(values)))
            report.record(f"AP_{name}", class_means[-1])
        if class_means:
            report.record("mAP", float(np.mean(class_means)))
        return report


def greedy_hits(scores: np.ndarray, distances: np.ndarray, threshold: float) -> np.ndarray:
    """Match the predictions of one frame in score order to the closest free ground truth.

    Args:
        scores: Prediction scores ``(P,)``.
        distances: Chamfer distances ``(P, G)``.
        threshold: Largest distance of a match.

    Returns:
        Whether every prediction matched ``(P,)``.
    """
    hits = np.zeros(len(scores), dtype=bool)
    free = np.ones(distances.shape[1], dtype=bool)
    for index in np.argsort(-scores, kind="stable"):
        candidates = np.where(free, distances[index], np.inf)
        if candidates.size == 0:
            continue
        best = int(np.argmin(candidates))
        if candidates[best] <= threshold:
            free[best] = False
            hits[index] = True
    return hits


class OnlineMapMetricSuite(MetricSuite[MapState]):
    """Accumulate the predicted and ground truth polylines of every frame."""

    prefix = "online_map"

    def __init__(
        self,
        components: list[Metric[MapState]],
        class_names: tuple[str, ...],
        thresholds: tuple[float, ...] = (0.5, 1.0, 1.5),
        num_dense_points: int = 100,
        **kwargs: Any,
    ) -> None:
        """Register the per frame accumulation state.

        Args:
            components: Injected metrics, as in :class:`MetricSuite`.
            class_names: Class names in label order.
            thresholds: Chamfer distances in meters a match must be within.
            num_dense_points: Points every polyline is densified to before the distance.
            **kwargs: Forwarded to :class:`MetricSuite`.
        """
        super().__init__(components=components, ranges=(), **kwargs)
        self.class_names = tuple(class_names)
        self.thresholds = tuple(float(threshold) for threshold in thresholds)
        self.num_dense_points = num_dense_points
        for name in (*FRAME_FIELDS, "unscored"):
            self.add_state(name, default=[], dist_reduce_fx=None)

    def requirements(self) -> MetricRequirements:
        """The online map frames.

        Returns:
            The requirements at this suite instance's stage.
        """
        return MetricRequirements(online_map=True)

    def update(self, eval_output: EvalOutput) -> None:
        """Keep the frames of a batch, one tensor per field and frame.

        Args:
            eval_output: The typed eval output the model built for one batch.

        Raises:
            ValueError: If the output carries no online map frames.
        """
        if eval_output.online_map is None:
            raise ValueError("The map suite needs the online map frames, the eval output has none.")
        for frame in eval_output.online_map.frames:
            values = [getattr(frame, name).detach() for name in FRAME_FIELDS]
            for name, value in zip(FRAME_FIELDS, values, strict=True):
                # An unscored frame keeps an empty entry so every rank gathers as many
                getattr(self, name).append(value if frame.scored else value[:0])
            self.unscored.append(torch.tensor([not frame.scored], device=values[0].device))

    def state_for(self, metric_range: MetricRange | None, metric_filter: MetricFilter) -> MapState:
        """Collect the frames with the settings of the metrics, without unscored frames.

        Args:
            metric_range: Unused, the suite reports no range windows.
            metric_filter: Unused, the suite reports no filters.

        Returns:
            The frames of the evaluation.
        """
        frames = [
            MapFrame(**{name: getattr(self, name)[index].cpu().numpy() for name in FRAME_FIELDS})
            for index in range(len(self.gt_labels))
            if not bool(self.unscored[index])
        ]
        return MapState(
            frames=frames,
            class_names=self.class_names,
            thresholds=self.thresholds,
            num_dense_points=self.num_dense_points,
        )
