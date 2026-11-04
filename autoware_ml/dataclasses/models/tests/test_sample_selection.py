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

"""Unit tests for the sample selection of model outputs."""

from __future__ import annotations

import unittest
from typing import NamedTuple

import torch
from pydantic import BaseModel, ConfigDict

from autoware_ml.dataclasses.models.sample_selection import select_output_samples


class _Scores(NamedTuple):
    """Scores of every sample with their own selection."""

    logits: torch.Tensor

    def select_samples(self, mask: torch.Tensor) -> _Scores:
        return _Scores(self.logits[mask])


class _Outputs(BaseModel):
    """Outputs with a tensor, a missing field, a nested output and a tuple of them."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    boxes: torch.Tensor
    track_ids: torch.Tensor | None
    scores: _Scores
    layers: tuple[_Scores, ...]


class TestSampleSelection(unittest.TestCase):
    def test_every_field_keeps_the_selected_samples(self) -> None:
        """
        Input: outputs of three samples, the second one selected.
        Expected: every field narrowed to that sample, the missing field still missing.
        """
        outputs = _Outputs(
            boxes=torch.arange(3.0),
            track_ids=None,
            scores=_Scores(torch.arange(3.0) + 10),
            layers=(_Scores(torch.arange(3.0) + 20), _Scores(torch.arange(3.0) + 30)),
        )

        selected = select_output_samples(outputs, torch.tensor([False, True, False]))

        self.assertEqual(selected["boxes"].tolist(), [1.0])
        self.assertIsNone(selected["track_ids"])
        self.assertEqual(selected["scores"].logits.tolist(), [11.0])
        self.assertEqual([layer.logits.tolist() for layer in selected["layers"]], [[21.0], [31.0]])
