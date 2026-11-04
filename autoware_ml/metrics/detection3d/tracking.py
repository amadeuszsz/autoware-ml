"""Multi-object tracking metrics of a temporal 3D detector.

The suite groups the frames of every stream in time order and matches the tracked boxes to the
ground truth of the same class by BEV center distance, keeping last frame's pairs while they
stay in reach (CLEAR MOT). It reports MOTA, ID switches and fragmentations overall and per class
at the track score threshold, and the best MOTA over a sweep of score thresholds.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from autoware_ml.metrics.base import EvalStage, Metric, MetricFilter, MetricRange, MetricSuite
from autoware_ml.metrics.detection3d.matching import (
    cost_center,
    gt_keep_mask,
    ignored_prediction_mask,
)
from autoware_ml.metrics.report import MetricReport


@dataclass(frozen=True)
class TrackingFrame:
    """Ground truth and tracked boxes of one frame.

    Attributes:
        gt_xy: BEV centers of the ground truth boxes ``(G, 2)``.
        gt_labels: Classes of the ground truth boxes ``(G,)``.
        gt_ids: Instance ids of the ground truth boxes ``(G,)``.
        pred_xy: BEV centers of the tracked boxes ``(P, 2)``.
        pred_labels: Classes of the tracked boxes ``(P,)``.
        pred_ids: Track ids of the tracked boxes ``(P,)``.
        pred_scores: Scores of the tracked boxes ``(P,)``.
    """

    gt_xy: np.ndarray
    gt_labels: np.ndarray
    gt_ids: np.ndarray
    pred_xy: np.ndarray
    pred_labels: np.ndarray
    pred_ids: np.ndarray
    pred_scores: np.ndarray

    def above(self, score_threshold: float) -> TrackingFrame:
        """Return the frame without the tracked boxes scored below ``score_threshold``."""
        keep = self.pred_scores >= score_threshold
        return TrackingFrame(
            gt_xy=self.gt_xy,
            gt_labels=self.gt_labels,
            gt_ids=self.gt_ids,
            pred_xy=self.pred_xy[keep],
            pred_labels=self.pred_labels[keep],
            pred_ids=self.pred_ids[keep],
            pred_scores=self.pred_scores[keep],
        )


@dataclass(frozen=True)
class TrackingState:
    """Frames of every stream, each stream in time order."""

    streams: list[list[TrackingFrame]]
    class_names: tuple[str, ...]
    match_distance: float
    score_threshold: float
    sweep_score_thresholds: tuple[float, ...]


class TrackingSummary(Metric[TrackingState]):
    """MOTA, ID switches and fragmentations, overall and per class, NaN where a count is zero."""

    def evaluate(self, state: TrackingState, stage: EvalStage) -> MetricReport:
        """Run the CLEAR MOT matching over every stream.

        Args:
            state: Frames of every stream.
            stage: Evaluation stage being reported.

        Returns:
            The tracking report.
        """
        counts = _clear_mot_counts(state, state.score_threshold)
        report = MetricReport()
        totals = _totals(counts)
        for label, label_counts in counts.items():
            if label < len(state.class_names) and label_counts["num_gt"] > 0:
                name = state.class_names[label]
                report.record(f"mota_{name}", _mota(label_counts))
                report.record(f"id_switches_{name}", float(label_counts["id_switches"]))
        mota = _mota(totals)
        report.record("mota", mota)
        report.record("recall", _ratio(totals["tp"], totals["num_gt"]))
        report.record("precision", _ratio(totals["tp"], totals["tp"] + totals["fp"]))
        report.record("id_switches", float(totals["id_switches"]))
        report.record("fragmentations", float(totals["fragmentations"]))
        report.record(
            "id_switches_per_gt_track", _ratio(totals["id_switches"], totals["num_gt_tracks"])
        )
        report.record("num_gt", float(totals["num_gt"]))
        # Best MOTA over the sweep, independent of the score calibration
        sweep = {
            threshold: _mota(_totals(_clear_mot_counts(state, threshold)))
            for threshold in state.sweep_score_thresholds
        }
        sweep[state.score_threshold] = mota
        best_threshold = max(
            sweep, key=lambda threshold: np.nan_to_num(sweep[threshold], nan=-np.inf)
        )
        report.record("mota_best", sweep[best_threshold])
        report.record("mota_best_score_threshold", best_threshold)
        return report


def _clear_mot_counts(state: TrackingState, score_threshold: float) -> dict[int, dict[str, int]]:
    """Match the tracked boxes from ``score_threshold`` on and count the CLEAR MOT events per class.

    Args:
        state: Frames of every stream.
        score_threshold: Score from which a tracked box counts.

    Returns:
        Event counters of every class.
    """
    counts: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for stream in state.streams:
        last_match: dict[int, int] = {}
        tracked: dict[int, bool] = {}
        previous_pairs: dict[int, int] = {}
        for full_frame in stream:
            frame = full_frame.above(score_threshold)
            pairs: dict[int, int] = {}
            for label in np.union1d(frame.gt_labels, frame.pred_labels).tolist():
                pairs |= _match_class(
                    frame,
                    int(label),
                    state.match_distance,
                    previous_pairs,
                    last_match,
                    tracked,
                    counts[int(label)],
                )
            previous_pairs = pairs
    return counts


def _totals(counts: dict[int, dict[str, int]]) -> dict[str, int]:
    """Sum the event counters over the classes."""
    totals: dict[str, int] = defaultdict(int)
    for label_counts in counts.values():
        for key, value in label_counts.items():
            totals[key] += value
    return totals


def _ratio(numerator: int, denominator: int) -> float:
    """Quotient of two counts, NaN for a zero denominator."""
    return numerator / denominator if denominator else math.nan


def _mota(counts: dict[str, int]) -> float:
    """Multi-object tracking accuracy of the given counts, NaN without ground truth."""
    return 1.0 - _ratio(counts["fn"] + counts["fp"] + counts["id_switches"], counts["num_gt"])


def _match_class(
    frame: TrackingFrame,
    label: int,
    match_distance: float,
    previous_pairs: dict[int, int],
    last_match: dict[int, int],
    tracked: dict[int, bool],
    counts: dict[str, int],
) -> dict[int, int]:
    """Match the boxes of one class in one frame and update the counts.

    Args:
        frame: Boxes of the frame.
        label: Class to match.
        match_distance: Largest BEV center distance of a match in meters.
        previous_pairs: Track every ground truth instance was matched to in the previous frame
            of the stream, the only pairs that carry over.
        last_match: Track every ground truth instance was last matched to, updated in place.
        tracked: Whether every ground truth instance was matched in its previous frame, updated
            in place.
        counts: Counters of the class, updated in place.

    Returns:
        The track every matched ground truth instance of the class was paired with.
    """
    gt_mask = frame.gt_labels == label
    pred_mask = frame.pred_labels == label
    gt_xy, gt_ids = frame.gt_xy[gt_mask], frame.gt_ids[gt_mask]
    pred_xy, pred_ids = frame.pred_xy[pred_mask], frame.pred_ids[pred_mask]
    distance = np.linalg.norm(gt_xy[:, None] - pred_xy[None], axis=-1)
    reachable = distance <= match_distance

    pairs: dict[int, int] = {}
    claimed: set[int] = set()
    for gt_index, gt_id in enumerate(gt_ids.tolist()):
        previous = np.flatnonzero(pred_ids == previous_pairs.get(gt_id, -1))
        if previous.size and int(previous[0]) not in claimed and reachable[gt_index, previous[0]]:
            pairs[gt_index] = int(previous[0])
            claimed.add(int(previous[0]))
    free_gt = [index for index in range(gt_ids.shape[0]) if index not in pairs]
    free_pred = [index for index in range(pred_ids.shape[0]) if index not in claimed]
    if free_gt and free_pred:
        cost = np.where(
            reachable[np.ix_(free_gt, free_pred)], distance[np.ix_(free_gt, free_pred)], 1e6
        )
        for row, col in zip(*linear_sum_assignment(cost), strict=True):
            if cost[row, col] < 1e6:
                pairs[free_gt[row]] = free_pred[col]

    for gt_index, gt_id in enumerate(gt_ids.tolist()):
        if gt_id not in tracked:
            counts["num_gt_tracks"] += 1
        matched = gt_index in pairs
        if matched:
            track_id = int(pred_ids[pairs[gt_index]])
            if gt_id in last_match and last_match[gt_id] != track_id:
                counts["id_switches"] += 1
            if gt_id in tracked and not tracked[gt_id] and gt_id in last_match:
                counts["fragmentations"] += 1
            last_match[gt_id] = track_id
        tracked[gt_id] = matched
    counts["num_gt"] += int(gt_ids.shape[0])
    counts["tp"] += len(pairs)
    counts["fn"] += int(gt_ids.shape[0]) - len(pairs)
    counts["fp"] += int(pred_ids.shape[0]) - len(pairs)
    return {
        int(gt_ids[gt_index]): int(pred_ids[pred_index]) for gt_index, pred_index in pairs.items()
    }


class TrackingMetricSuite(MetricSuite[TrackingState]):
    """Accumulate the tracked boxes of a temporal detector per stream."""

    prefix = "track"
    _required_keys = (
        "predictions",
        "gt_boxes",
        "gt_labels",
        "gt_instance_ids",
        "gt_observed",
        "stream_ids",
        "timestamp",
        "scored",
    )

    def __init__(
        self,
        components: list[Metric[TrackingState]],
        class_names: tuple[str, ...],
        score_threshold: float,
        sweep_score_thresholds: tuple[float, ...] = (),
        match_distance: float = 2.0,
        max_distances: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> None:
        """Register the per frame accumulation states.

        Args:
            components: Injected metrics, as in :class:`MetricSuite`.
            class_names: Class names in label order.
            score_threshold: Score from which a tracked box counts.
            sweep_score_thresholds: Further score thresholds searched for the best MOTA.
            match_distance: Largest BEV center distance of a match in meters.
            max_distances: Class name to maximum evaluation distance in meters, as in the
                detection suites. A tracked box beyond it leaves, and a ground truth box beyond
                it leaves with the tracked boxes that claim it.
            **kwargs: Forwarded to :class:`MetricSuite`.

        Raises:
            ValueError: If a component sets a filter, the suite reports no filtered slices.
        """
        super().__init__(components=components, ranges=(), **kwargs)
        self.refuse_component_filters()
        self.class_names = tuple(class_names)
        self.score_threshold = float(score_threshold)
        self.sweep_score_thresholds = tuple(float(value) for value in sweep_score_thresholds)
        self.min_score_threshold = min((self.score_threshold, *self.sweep_score_thresholds))
        self.match_distance = float(match_distance)
        self.max_distances = max_distances
        for name in (
            "unscored",
            "stream_ids",
            "timestamps",
            "gt_xy",
            "gt_labels",
            "gt_ids",
            "pred_xy",
            "pred_labels",
            "pred_ids",
            "pred_scores",
        ):
            self.add_state(name, default=[], dist_reduce_fx=None)

    def update(self, eval_out: dict[str, Any]) -> None:
        """Accumulate the observed ground truth and the confident tracked boxes of a batch.

        A query decodes into one box, so a track has at most one box per frame. A tracked box that
        claims a ground truth box the stream has not observed yet, or one beyond the distance of
        its class, leaves the evaluation with it.

        Args:
            eval_out: Flat eval-output dict built by the model for one batch.
        """
        for index, prediction in enumerate(eval_out["predictions"]):
            if prediction["instance_ids"] is None:
                raise ValueError("Tracking metrics need the track id of every predicted box.")
            boxes = eval_out["gt_boxes"][index].detach().float()
            device = boxes.device
            scored = self.frame_scored(eval_out, index)
            self.unscored.append(torch.tensor([not scored], device=device))
            self.stream_ids.append(eval_out["stream_ids"][index].reshape(1).to(device))
            self.timestamps.append(
                torch.as_tensor(
                    eval_out["timestamp"][index], dtype=torch.float64, device=device
                ).reshape(1)
            )
            if not scored:
                # An unscored frame keeps empty entries so every rank gathers as many
                self._append_empty_frame(boxes, prediction["bboxes_3d"])
                continue
            labels = eval_out["gt_labels"][index].long()
            in_range = gt_keep_mask(boxes, labels, None, self.class_names, self.max_distances, 0)
            observed = eval_out["gt_observed"][index].to(device) & in_range
            tracked = prediction["instance_ids"] >= 0
            track_ids = prediction["instance_ids"][tracked]
            if torch.unique(track_ids).numel() != track_ids.numel():
                raise ValueError(
                    f"Frame {index} gives a track more than one box, a query decodes into one box."
                )
            keep = tracked & (prediction["scores_3d"] >= self.min_score_threshold)
            pred_boxes = prediction["bboxes_3d"][keep].detach().float()
            pred_labels = prediction["labels_3d"][keep].long()
            # The class distance bounds the tracked boxes as it bounds the ground truth
            near = gt_keep_mask(
                pred_boxes, pred_labels, None, self.class_names, self.max_distances, 0
            )
            pred_boxes = pred_boxes[near]
            pred_labels = pred_labels[near]
            pred_scores = prediction["scores_3d"][keep][near].detach().float()
            pred_ids = prediction["instance_ids"][keep][near].long()
            # The claims run greedily in score order, one pass holds for every higher threshold
            claimed = ignored_prediction_mask(
                pred_boxes,
                pred_scores,
                pred_labels,
                boxes[observed],
                labels[observed],
                boxes[~observed],
                labels[~observed],
                cost_center,
                self.match_distance,
                self.match_distance,
            )
            self.gt_xy.append(boxes[observed, :2])
            self.gt_labels.append(labels[observed])
            self.gt_ids.append(eval_out["gt_instance_ids"][index][observed].long())
            self.pred_xy.append(pred_boxes[~claimed, :2])
            self.pred_labels.append(pred_labels[~claimed])
            self.pred_ids.append(pred_ids[~claimed])
            self.pred_scores.append(pred_scores[~claimed])

    def _append_empty_frame(self, gt_boxes: torch.Tensor, pred_boxes: torch.Tensor) -> None:
        """Give an unscored frame empty ground truth and prediction entries.

        Args:
            gt_boxes: Ground truth boxes of the frame, for their device.
            pred_boxes: Predicted boxes of the frame, for their device.
        """
        self.gt_xy.append(gt_boxes[:0, :2])
        self.gt_labels.append(torch.zeros(0, dtype=torch.long, device=gt_boxes.device))
        self.gt_ids.append(torch.zeros(0, dtype=torch.long, device=gt_boxes.device))
        self.pred_xy.append(pred_boxes.detach().float()[:0, :2])
        self.pred_labels.append(torch.zeros(0, dtype=torch.long, device=pred_boxes.device))
        self.pred_ids.append(torch.zeros(0, dtype=torch.long, device=pred_boxes.device))
        self.pred_scores.append(torch.zeros(0, device=pred_boxes.device))

    def state_for(
        self, metric_range: MetricRange | None, metric_filter: MetricFilter
    ) -> TrackingState:
        """Group the frames by stream and order every stream in time, without unscored frames.

        Args:
            metric_range: Unused, the suite reports no range windows.
            metric_filter: Unused, the suite reports no filters.

        Returns:
            Frames of every stream.
        """
        streams: dict[int, list[tuple[float, TrackingFrame]]] = defaultdict(list)
        for index in range(len(self.stream_ids)):
            if bool(self.unscored[index]):
                continue
            frame = TrackingFrame(
                gt_xy=self.gt_xy[index].cpu().numpy(),
                gt_labels=self.gt_labels[index].cpu().numpy(),
                gt_ids=self.gt_ids[index].cpu().numpy(),
                pred_xy=self.pred_xy[index].cpu().numpy(),
                pred_labels=self.pred_labels[index].cpu().numpy(),
                pred_ids=self.pred_ids[index].cpu().numpy(),
                pred_scores=self.pred_scores[index].cpu().numpy(),
            )
            streams[int(self.stream_ids[index])].append((float(self.timestamps[index]), frame))
        return TrackingState(
            streams=[
                [frame for _, frame in sorted(frames, key=lambda item: item[0])]
                for frames in streams.values()
            ],
            class_names=self.class_names,
            match_distance=self.match_distance,
            score_threshold=self.score_threshold,
            sweep_score_thresholds=self.sweep_score_thresholds,
        )
