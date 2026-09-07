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

"""Axis permutation transforms."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from autoware_ml.datamodule.samples.sample import Sample
from autoware_ml.transforms.base import BaseTransform


class PermuteAxes(BaseTransform):
    """Permute the axes of the fused calibration image.

    The calibration pipeline uses this to turn the fused image from height, width, channels
    into the channel first layout the model consumes.
    """

    _required_fields = ["calibration"]

    def __init__(self, *, axes: Sequence[int]) -> None:
        """Initialize the PermuteAxes transform.

        Args:
            axes: Axis order of the permuted fused image.
        """
        self.axes = tuple(axes)

    def transform(self, sample: Sample) -> Sample:
        """Permute the axes of the fused calibration image.

        Args:
            sample: Sample with a fused calibration image.

        Returns:
            Sample with the permuted fused image.
        """
        fused_image = sample.calibration.fused_image
        if fused_image is None:
            raise ValueError("PermuteAxes requires a fused calibration image.")
        if fused_image.ndim != len(self.axes):
            raise ValueError(
                f"PermuteAxes: the fused image has {fused_image.ndim} dimensions but the "
                f"configured axes are {self.axes}."
            )
        calibration = sample.calibration.model_copy(
            update={"fused_image": np.transpose(fused_image, self.axes)}
        )
        return sample.replace(calibration=calibration)
