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

"""Tests for the axis permutation transforms."""

import numpy as np
import pytest

from autoware_ml.datamodule.samples.calibration import CalibrationSample
from autoware_ml.datamodule.samples.sample import Sample
from autoware_ml.testing.factories import make_sample
from autoware_ml.transforms.common.tensor import PermuteAxes
from autoware_ml.utils.calibration import CalibrationData


def _make_sample(fused_image: np.ndarray | None) -> Sample:
    data = CalibrationData(
        camera_matrix=np.eye(3, dtype=np.float32),
        distortion_coefficients=np.zeros(5, dtype=np.float32),
        lidar_to_camera_transformation=np.eye(4, dtype=np.float32),
    )
    calibration = CalibrationSample(data=data, camera_name="CAM_FRONT", fused_image=fused_image)
    return make_sample().replace(calibration=calibration)


def test_permute_axes_moves_channels_first() -> None:
    rng = np.random.default_rng(0)
    fused_image = rng.uniform(0.0, 1.0, size=(32, 64, 5)).astype(np.float32)
    sample = _make_sample(fused_image)

    output = PermuteAxes(axes=(2, 0, 1))(sample)

    permuted = output.calibration.fused_image
    assert permuted.shape == (5, 32, 64)
    assert np.array_equal(permuted[3], fused_image[..., 3])


def test_permute_axes_identity_keeps_values() -> None:
    rng = np.random.default_rng(1)
    fused_image = rng.uniform(0.0, 1.0, size=(8, 16, 5)).astype(np.float32)
    sample = _make_sample(fused_image)

    output = PermuteAxes(axes=(0, 1, 2))(sample)

    assert np.array_equal(output.calibration.fused_image, fused_image)


def test_permute_axes_dimension_mismatch_raises() -> None:
    fused_image = np.zeros((8, 16, 5), dtype=np.float32)
    sample = _make_sample(fused_image)

    with pytest.raises(ValueError, match="dimensions"):
        PermuteAxes(axes=(1, 0))(sample)


def test_permute_axes_requires_a_fused_image() -> None:
    sample = _make_sample(None)

    with pytest.raises(ValueError, match="fused calibration image"):
        PermuteAxes(axes=(2, 0, 1))(sample)
