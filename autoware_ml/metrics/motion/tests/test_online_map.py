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

"""Tests for the chamfer distance average precision of the map suite."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from autoware_ml.metrics.base import EvalStage
from autoware_ml.metrics.eval_output import EvalOutput
from autoware_ml.metrics.motion.eval_output import OnlineMapEvalFrame, OnlineMapEvalOutput
from autoware_ml.metrics.motion.online_map import (
    OnlineMapMetricSuite,
    MapVectorAP,
    chamfer_distances,
)

LINE = np.stack([np.linspace(0.0, 10.0, 5), np.zeros(5)], axis=-1)


def shifted(offset: float) -> np.ndarray:
    """The reference line moved sideways."""
    return LINE + np.array([0.0, offset])


def frame(preds: list[tuple[np.ndarray, float, int]], gts: list[tuple[np.ndarray, int]]) -> dict:
    """One frame of predictions (points, score, label) and ground truth (points, label)."""
    return {
        "pred_points": torch.tensor(np.stack([p for p, _, _ in preds]), dtype=torch.float32),
        "pred_scores": torch.tensor([s for _, s, _ in preds]),
        "pred_labels": torch.tensor([label for _, _, label in preds]),
        "gt_points": torch.tensor(np.stack([g for g, _ in gts]), dtype=torch.float32),
        "gt_labels": torch.tensor([label for _, label in gts]),
    }


def evaluate(frames: list[dict]) -> dict[str, float]:
    """Run the AP component over the frames."""
    suite = OnlineMapMetricSuite(
        components=[MapVectorAP()], class_names=("divider", "boundary", "ped_crossing")
    )
    suite.update(
        EvalOutput(
            online_map=OnlineMapEvalOutput(
                frames=[OnlineMapEvalFrame(**fields) for fields in frames]
            )
        )
    )
    state = suite.state_for(None, suite.components[0].filter)
    report = suite.components[0].evaluate(state, EvalStage.VAL)
    return {str(key): report.metric_value(key) for key in report.metric_names()}


class TestOnlineMapMetrics(unittest.TestCase):
    """The chamfer distance average precision of the map suite."""

    def test_the_chamfer_distance_of_a_parallel_line_is_its_offset(self) -> None:
        """
        Input: the reference line against itself and against a copy 1 m to the side.
        Expected: distances 0 and 1.
        """
        distances = chamfer_distances(np.stack([LINE, shifted(1.0)]), LINE[None])

        np.testing.assert_allclose(distances[:, 0], [0.0, 1.0], atol=1e-6)

    def test_precision_counts_the_misses_ranked_above_a_hit(self) -> None:
        """
        Input: one divider, predicted exactly at score 0.8 below a far prediction at 0.9, and a
          second frame where a 0.75 m offset prediction is the only one.
        Expected: AP 0.5 at every threshold for the first frame alone. In the second frame the
          offset prediction misses at 0.5 m and hits at 1.0 and 1.5 m.
        """
        first = evaluate([frame([(shifted(20.0), 0.9, 0), (LINE, 0.8, 0)], [(LINE, 0)])])
        second = evaluate([frame([(shifted(0.75), 0.9, 0)], [(LINE, 0)])])

        self.assertAlmostEqual(first["AP_divider_0.5m"], 0.5)
        self.assertAlmostEqual(first["AP_divider_1.5m"], 0.5)
        np.testing.assert_allclose(
            [second[f"AP_divider_{t}m"] for t in (0.5, 1.0, 1.5)], [0.0, 1.0, 1.0]
        )
        self.assertAlmostEqual(second["AP_divider"], 2.0 / 3.0)

    def test_classes_without_ground_truth_stay_out_of_the_mean(self) -> None:
        """
        Input: an exact divider and a boundary predicted on the divider of the same frame.
        Expected: divider AP 1, no boundary key, mAP equal to the divider AP.
        """
        report = evaluate([frame([(LINE, 0.9, 0), (LINE, 0.8, 1)], [(LINE, 0)])])

        self.assertAlmostEqual(report["AP_divider"], 1.0)
        self.assertNotIn("AP_boundary", report)
        self.assertAlmostEqual(report["mAP"], 1.0)
