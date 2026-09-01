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

"""Tests for the camera distortion correction transforms."""

import numpy as np
import pytest

from autoware_ml.datamodule.samples.calibration import CalibrationSample
from autoware_ml.datamodule.samples.sample import Sample
from autoware_ml.testing.factories import make_sample
from autoware_ml.transforms.camera.distortion import UndistortImage
from autoware_ml.utils.calibration import CalibrationData


def _make_sample(distortion_coefficients: np.ndarray) -> Sample:
    data = CalibrationData(
        camera_matrix=np.array(
            [[100.0, 0.0, 32.0], [0.0, 100.0, 24.0], [0.0, 0.0, 1.0]], dtype=np.float32
        ),
        distortion_coefficients=distortion_coefficients,
        lidar_to_camera_transformation=np.eye(4, dtype=np.float32),
    )
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, size=(48, 64, 3)).astype(np.float32)
    calibration = CalibrationSample(data=data, camera_name="CAM_FRONT", image=image)
    return make_sample().model_copy(update={"calibration": calibration})


def test_zero_distortion_passes_through() -> None:
    sample = _make_sample(np.zeros(5, dtype=np.float32))

    output = UndistortImage()(sample)

    assert output is sample


def test_undistortion_updates_image_and_calibration() -> None:
    coefficients = np.array([0.1, -0.2, 0.001, 0.001, 0.05], dtype=np.float32)
    sample = _make_sample(coefficients)

    output = UndistortImage()(sample)

    assert output.calibration.image.shape == sample.calibration.image.shape
    assert output.calibration.image.dtype == np.float32
    assert np.allclose(output.calibration.data.distortion_coefficients, 0.0)
    assert not np.allclose(
        output.calibration.data.new_camera_matrix, sample.calibration.data.camera_matrix
    )


def test_input_calibration_data_is_not_mutated() -> None:
    coefficients = np.array([0.1, -0.2, 0.001, 0.001, 0.05], dtype=np.float32)
    sample = _make_sample(coefficients)

    UndistortImage()(sample)

    assert np.allclose(sample.calibration.data.distortion_coefficients, coefficients)
    assert np.allclose(
        sample.calibration.data.new_camera_matrix, sample.calibration.data.camera_matrix
    )


def test_requires_loaded_image() -> None:
    sample = _make_sample(np.zeros(5, dtype=np.float32))
    calibration = sample.calibration.model_copy(update={"image": None})
    sample = sample.model_copy(update={"calibration": calibration})

    with pytest.raises(ValueError, match="loaded calibration image"):
        UndistortImage()(sample)
