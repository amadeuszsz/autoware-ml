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

"""Point cloud perturbation transforms."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from autoware_ml.datamodule.samples.sample import Sample
from autoware_ml.transforms.base import BaseTransform
from autoware_ml.types.geometry import PointFeatureName


class RandomJitter(BaseTransform):
    """Perturb the point coordinates with clipped Gaussian noise."""

    _required_fields = ["points"]

    def __init__(self, *, p: float | None = None, sigma: float, clip: float) -> None:
        """Initialize the RandomJitter transform.

        Args:
            p: Probability of applying the transform. None means always apply.
            sigma: Standard deviation of the Gaussian noise.
            clip: Maximum absolute jitter applied per coordinate.
        """
        self.p = p
        self.sigma = sigma
        self.clip = clip

    def transform(self, sample: Sample) -> Sample:
        """Perturb the point coordinates with Gaussian noise.

        Args:
            sample: Sample with a loaded point cloud.

        Returns:
            Sample with the jittered point coordinates.
        """
        noise = np.clip(
            self.sigma * np.random.randn(len(sample.points), 3),
            -self.clip,
            self.clip,
        ).astype(np.float32)
        return sample.replace(points=sample.points.with_coord(sample.points.coord + noise))


class RandomStrengthJitter(BaseTransform):
    """Perturb the normalized intensity with a random gamma, scale, and shift.

    Applies clip(intensity ** gamma * scale + shift, 0, 1) with parameters drawn uniformly per
    sample, emulating reflectivity calibration differences between sensors. The intensity
    feature must already be normalized to [0, 1] by PreparePointCloudInput.
    """

    _required_fields = ["points"]

    def __init__(
        self,
        *,
        p: float | None = None,
        gamma_range: Sequence[float],
        scale_range: Sequence[float],
        shift_range: Sequence[float],
    ) -> None:
        """Initialize the RandomStrengthJitter transform.

        Args:
            p: Probability of applying the transform. None means always apply.
            gamma_range: Min and max exponent applied to the normalized intensity.
            scale_range: Min and max multiplicative factor.
            shift_range: Min and max additive offset.
        """
        for name, bounds in (
            ("gamma_range", gamma_range),
            ("scale_range", scale_range),
            ("shift_range", shift_range),
        ):
            if len(bounds) != 2 or bounds[0] > bounds[1]:
                raise ValueError(f"{name} must be an ascending [min, max] pair, got {bounds}.")
        if gamma_range[0] <= 0.0:
            raise ValueError(f"gamma_range values must be positive, got {gamma_range}.")
        self.p = p
        self.gamma_range = tuple(gamma_range)
        self.scale_range = tuple(scale_range)
        self.shift_range = tuple(shift_range)

    def transform(self, sample: Sample) -> Sample:
        """Apply the sampled gamma, scale, and shift to the intensity feature.

        Args:
            sample: Sample with a loaded point cloud carrying the intensity feature.

        Returns:
            Sample with the jittered intensity feature.
        """
        gamma = np.random.uniform(*self.gamma_range)
        scale = np.random.uniform(*self.scale_range)
        shift = np.random.uniform(*self.shift_range)
        intensity = sample.points.feature(PointFeatureName.INTENSITY)
        jittered = np.clip(np.power(intensity, gamma) * scale + shift, 0.0, 1.0).astype(np.float32)
        return sample.replace(
            points=sample.points.with_feature(PointFeatureName.INTENSITY, jittered)
        )
