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

"""Tests for the camera image masking transforms."""

import numpy as np

from autoware_ml.datamodule.samples.camera import ImageSet
from autoware_ml.datamodule.samples.sample import Sample
from autoware_ml.testing.factories import make_sample
from autoware_ml.transforms.camera.masking import GridMask


def _make_sample_with_images(num_cameras: int = 2, size: int = 64) -> Sample:
    eye = np.tile(np.eye(4, dtype=np.float32), (num_cameras, 1, 1))
    image_set = ImageSet(
        images=np.ones((num_cameras, 3, size, size), dtype=np.float32),
        camera_names=tuple(f"CAM_{index}" for index in range(num_cameras)),
        camera_intrinsics=eye.copy(),
        lidar2cam=eye.copy(),
        lidar2img=eye.copy(),
    )
    return make_sample().model_copy(update={"images": image_set})


def test_grid_mask_zeroes_grid_cells() -> None:
    np.random.seed(0)
    sample = _make_sample_with_images()

    output = GridMask(p=1.0, ratio=0.5, rotate=0)(sample)

    assert output.images.images.shape == sample.images.images.shape
    assert (output.images.images == 0).any()
    assert (output.images.images == 1).any()


def test_grid_mask_keeps_projection_matrices() -> None:
    np.random.seed(0)
    sample = _make_sample_with_images()

    output = GridMask(p=1.0, ratio=0.5, rotate=0)(sample)

    assert np.array_equal(output.images.lidar2img, sample.images.lidar2img)
    assert np.array_equal(output.images.camera_intrinsics, sample.images.camera_intrinsics)


def test_grid_mask_skips_with_zero_probability() -> None:
    sample = _make_sample_with_images()

    output = GridMask(p=0.0)(sample)

    assert output is sample
