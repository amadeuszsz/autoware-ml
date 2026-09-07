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

"""Tests for the image space photometric transforms."""

import numpy as np
import pytest

from autoware_ml.datamodule.samples.calibration import CalibrationSample
from autoware_ml.datamodule.samples.sample import Sample
from autoware_ml.testing.factories import make_sample
from autoware_ml.transforms.image.image import PhotometricDistortion
from autoware_ml.utils.calibration import CalibrationData


def _make_sample(*, with_image: bool = True) -> Sample:
    data = CalibrationData(
        camera_matrix=np.eye(3, dtype=np.float32),
        distortion_coefficients=np.zeros(5, dtype=np.float32),
        lidar_to_camera_transformation=np.eye(4, dtype=np.float32),
    )
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, size=(64, 64, 3)).astype(np.float32) if with_image else None
    calibration = CalibrationSample(data=data, camera_name="CAM_FRONT", image=image)
    return make_sample().replace(calibration=calibration)


def test_distortion_changes_the_image_and_keeps_the_layout() -> None:
    np.random.seed(0)
    sample = _make_sample()
    transform = PhotometricDistortion(p=1.0, brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)

    output = transform(sample)

    image = output.calibration.image
    assert image.shape == sample.calibration.image.shape
    assert image.dtype == np.float32
    assert image.min() >= 0.0
    assert image.max() <= 255.0
    assert not np.array_equal(image, sample.calibration.image)


def test_zero_probability_keeps_the_sample() -> None:
    sample = _make_sample()

    output = PhotometricDistortion(p=0.0)(sample)

    assert output is sample


def test_requires_loaded_image() -> None:
    sample = _make_sample(with_image=False)

    with pytest.raises(ValueError, match="loaded calibration image"):
        PhotometricDistortion(p=1.0)(sample)
