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

"""Open-loop motion prediction metrics.

Predictions are matched to the ground truth of their class by BEV center distance in score order,
the matched pairs with a known future report the minimum average and final displacement over the
modes, the miss rate and the end-to-end prediction accuracy (EPA).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from autoware_ml.metrics.base import EvalStage, Metric, MetricFilter, MetricRange, MetricSuite
from autoware_ml.metrics.detection3d.matching import (
    cost_center,
    gt_keep_mask,
    ignored_prediction_mask,
)
from autoware_ml.metrics.detection3d.criticality import greedy_match
from autoware_ml.metrics.report import MetricReport


@dataclass(frozen=True)
class MotionFrame:
    """Predicted and ground truth trajectories of one frame, positions in the frame.

    Attributes:
        pred_xy: BEV centers of the predicted agents ``(P, 2)``.
        pred_labels: Classes of the predicted agents ``(P,)``.
        pred_scores: Detection scores of the predicted agents ``(P,)``.
        pred_trajectories: Trajectory of every mode ``(P, M, T, 2)``.
        gt_xy: BEV centers of the ground truth agents ``(G, 2)``.
        gt_labels: Classes of the ground truth agents ``(G,)``.
        gt_futures: Future positions ``(G, T, 2)``.
        gt_valid: Whether a future position is known ``(G, T)``.
        gt_size: Length and width ``(G, 2)``.
        gt_yaw: Heading ``(G,)``.
    """

    pred_xy: np.ndarray
    pred_labels: np.ndarray
    pred_scores: np.ndarray
    pred_trajectories: np.ndarray
    gt_xy: np.ndarray
    gt_labels: np.ndarray
    gt_futures: np.ndarray
    gt_valid: np.ndarray
    gt_size: np.ndarray
    gt_yaw: np.ndarray


@dataclass(frozen=True)
class MotionState:
    """Frames of the evaluation and the settings of the metrics."""

    frames: list[MotionFrame]
    class_names: tuple[str, ...]
    match_distance: float
    miss_distance: float


class AgentMotionSummary(Metric[MotionState]):
    """minADE, minFDE, miss rate and EPA per class and their mean over the classes."""

    def evaluate(self, state: MotionState, stage: EvalStage) -> MetricReport:
        """Score the confident predictions against every box of their class, as SparseDrive's EPA.

        Args:
            state: Frames of the evaluation.
            stage: Evaluation stage, unused.

        Returns:
            The values per class and their mean over the classes.
        """
        report = MetricReport()
        means: dict[str, list[float]] = {"minADE": [], "minFDE": [], "MR": [], "EPA": []}
        for label, name in enumerate(state.class_names):
            ade, fde, hits, false_positives, num_gt = [], [], 0, 0, 0
            for frame in state.frames:
                pred = frame.pred_labels == label
                gt = frame.gt_labels == label
                num_gt += int(gt.sum())
                is_tp, matched = greedy_match(
                    frame.gt_xy[gt],
                    frame.pred_xy[pred],
                    frame.pred_scores[pred],
                    state.match_distance,
                )
                matched_pred = np.flatnonzero(is_tp)
                matched_gt = matched[matched_pred]
                false_positives += int(pred.sum()) - len(matched_pred)
                # A match without a known future is no false positive but can never be a hit
                known = frame.gt_valid[gt][matched_gt].any(axis=1)
                trajectories = frame.pred_trajectories[pred][matched_pred][known]
                futures = frame.gt_futures[gt][matched_gt][known]
                valid = frame.gt_valid[gt][matched_gt][known]
                for modes, future, mask in zip(trajectories, futures, valid, strict=True):
                    errors = np.linalg.norm(modes - future[None], axis=-1)
                    last = int(np.flatnonzero(mask)[-1])
                    ade.append(float((errors[:, mask]).mean(axis=1).min()))
                    fde.append(float(errors[:, last].min()))
                    hits += int(fde[-1] < state.miss_distance)
            if num_gt == 0:
                continue
            values = {
                "minADE": float(np.mean(ade)) if ade else float("nan"),
                "minFDE": float(np.mean(fde)) if fde else float("nan"),
                "MR": float(np.mean(np.asarray(fde) >= state.miss_distance))
                if fde
                else float("nan"),
                "EPA": (hits - 0.5 * false_positives) / num_gt,
            }
            for key, value in values.items():
                report.record(f"{key}_{name}", value)
                if np.isfinite(value):
                    means[key].append(value)
        for key, values in means.items():
            if values:
                report.record(key, float(np.mean(values)))
        return report


class MotionMetricSuite(MetricSuite[MotionState]):
    """Accumulate the predicted and ground truth trajectories of every frame."""

    prefix = "motion"
    _required_keys = ("motion_frames", "gt_observed", "scored")
    frame_fields: tuple[str, ...] = (
        "pred_xy",
        "pred_labels",
        "pred_scores",
        "pred_trajectories",
        "gt_xy",
        "gt_labels",
        "gt_futures",
        "gt_valid",
        "gt_size",
        "gt_yaw",
    )

    def __init__(
        self,
        components: list[Metric[MotionState]],
        class_names: tuple[str, ...],
        score_threshold: float,
        match_distance: float = 2.0,
        miss_distance: float = 2.0,
        max_distances: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> None:
        """Register the per frame accumulation state.

        Args:
            components: Injected metrics, as in :class:`MetricSuite`.
            class_names: Class names in label order.
            score_threshold: Detection score from which a prediction counts.
            match_distance: Largest BEV center distance of a match in meters.
            miss_distance: Final displacement from which a matched agent is a miss.
            max_distances: Class name to maximum evaluation distance in meters, as in the
                tracking suite.
            **kwargs: Forwarded to :class:`MetricSuite`.

        Raises:
            ValueError: If a component sets a filter, the suite reports no filtered slices.
        """
        super().__init__(components=components, ranges=(), **kwargs)
        self.refuse_component_filters()
        self.class_names = tuple(class_names)
        self.score_threshold = float(score_threshold)
        self.match_distance = float(match_distance)
        self.miss_distance = float(miss_distance)
        self.max_distances = max_distances
        for name in self.frame_fields:
            self.add_state(name, default=[], dist_reduce_fx=None)
        self.add_state("unscored", default=[], dist_reduce_fx=None)

    def update(self, eval_out: dict[str, Any]) -> None:
        """Keep the frames of a batch, one tensor per field and frame.

        The boxes and the predictions follow the tracking suite: a box beyond the distance of its
        class, or not observed in its stream yet, leaves with the predictions that claim it, and
        a prediction beyond the distance of its class leaves.

        Args:
            eval_out: Flat eval-output dict built by the model for one batch, ``motion_frames``
                holds one dict of tensors per frame keyed by the frame fields of the suite.
        """
        for index, frame in enumerate(eval_out["motion_frames"]):
            scored = self.frame_scored(eval_out, index)
            fields = {name: torch.as_tensor(frame[name]).detach() for name in self.frame_fields}
            self.unscored.append(torch.tensor([not scored], device=fields["gt_xy"].device))
            if not scored:
                # An unscored frame keeps an empty entry so every rank gathers as many
                for name, value in fields.items():
                    getattr(self, name).append(value[:0])
                continue
            gt_xy, gt_labels = fields["gt_xy"], fields["gt_labels"]
            pred_xy, pred_labels = fields["pred_xy"], fields["pred_labels"]
            gt_keep = gt_keep_mask(
                gt_xy, gt_labels, None, self.class_names, self.max_distances, 0
            ) & eval_out["gt_observed"][index].to(gt_xy.device)
            # The metrics read the predictions at or above the score threshold only
            confident = (fields["pred_scores"] >= self.score_threshold) & gt_keep_mask(
                pred_xy, pred_labels, None, self.class_names, self.max_distances, 0
            )
            pred_index = confident.nonzero().flatten()
            claimed = ignored_prediction_mask(
                pred_xy[pred_index],
                fields["pred_scores"][pred_index],
                pred_labels[pred_index],
                gt_xy[gt_keep],
                gt_labels[gt_keep],
                gt_xy[~gt_keep],
                gt_labels[~gt_keep],
                cost_center,
                self.match_distance,
                self.match_distance,
            )
            pred_index = pred_index[~claimed.to(pred_index.device)]
            for name, value in fields.items():
                # A prediction and a box field keep their rows, a field of the frame stays whole
                if name.startswith("pred_"):
                    value = value[pred_index]
                elif name.startswith("gt_"):
                    value = value[gt_keep]
                getattr(self, name).append(value)

    def state_for(
        self, metric_range: MetricRange | None, metric_filter: MetricFilter
    ) -> MotionState:
        """Collect the frames with the settings of the metrics, without unscored frames.

        Args:
            metric_range: Unused, the suite reports no range windows.
            metric_filter: Unused, the suite reports no filters.

        Returns:
            The frames of the evaluation.
        """
        frames = [
            self.frame(index)
            for index in range(len(self.unscored))
            if not bool(self.unscored[index])
        ]
        return MotionState(
            frames=frames,
            class_names=self.class_names,
            match_distance=self.match_distance,
            miss_distance=self.miss_distance,
        )

    def frame(self, index: int) -> MotionFrame:
        """Build the accumulated frame at an index."""
        return MotionFrame(
            **{name: getattr(self, name)[index].cpu().numpy() for name in self.frame_fields}
        )
