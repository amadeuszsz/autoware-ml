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

"""Tests for the camera and lidar fusion geometric augmentations."""

import numpy as np
import pytest

from autoware_ml.datamodule.samples.camera import ImageSet
from autoware_ml.datamodule.samples.sample import Sample
from autoware_ml.testing.factories import make_boxes3d, make_point_cloud, make_sample
from autoware_ml.transforms.camera_lidar.geometry import GlobalRotScaleTrans, RandomFlip3D


def _make_fusion_sample() -> Sample:
    eye = np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))
    lidar2cam = eye.copy()
    lidar2cam[:, :3, 3] = [[0.5, -0.2, 1.0], [-1.0, 0.3, 1.2]]
    intrinsics = eye.copy()
    intrinsics[:, 0, 0] = 100.0
    intrinsics[:, 1, 1] = 100.0
    image_set = ImageSet(
        images=np.ones((2, 3, 8, 8), dtype=np.float32),
        camera_names=("CAM_FRONT", "CAM_LEFT"),
        camera_intrinsics=intrinsics,
        lidar2cam=lidar2cam,
        lidar2img=intrinsics @ lidar2cam,
    )
    sample = make_sample(
        points=make_point_cloud(num_points=20, with_time_lag=False), boxes=make_boxes3d()
    )
    return sample.model_copy(update={"images": image_set})


def _project(matrix: np.ndarray, point: np.ndarray) -> np.ndarray:
    return matrix @ np.append(point, 1.0)


def test_global_rot_scale_trans_keeps_the_projection_consistent() -> None:
    sample = _make_fusion_sample()
    transform = GlobalRotScaleTrans(rot_range=[0.1, 0.1], scale_ratio_range=[1.5, 1.5])

    output = transform(sample)

    assert np.allclose(output.boxes.params[:, 3:6], sample.boxes.params[:, 3:6] * 1.5)
    original_point = sample.points.coord[0]
    transformed_point = output.points.coord[0]
    for camera_index in range(2):
        before = _project(sample.images.lidar2img[camera_index], original_point)
        after = _project(output.images.lidar2img[camera_index], transformed_point)
        assert np.allclose(before, after, atol=1e-3)


def test_random_flip3d_flips_scene_and_camera_matrices() -> None:
    sample = _make_fusion_sample()
    transform = RandomFlip3D(flip_ratio_bev_horizontal=1.0, flip_ratio_bev_vertical=0.0)

    output = transform(sample)

    assert np.allclose(output.points.coord[:, 1], -sample.points.coord[:, 1])
    assert np.allclose(output.boxes.params[:, 1], -sample.boxes.params[:, 1])
    assert np.allclose(output.boxes.params[:, 6], -sample.boxes.params[:, 6])
    original_point = sample.points.coord[0]
    flipped_point = output.points.coord[0]
    for camera_index in range(2):
        before = _project(sample.images.lidar2img[camera_index], original_point)
        after = _project(output.images.lidar2img[camera_index], flipped_point)
        assert np.allclose(before, after, atol=1e-4)


def test_missing_image_set_raises() -> None:
    sample = make_sample(points=make_point_cloud(num_points=10, with_time_lag=False))

    with pytest.raises(ValueError, match="images"):
        GlobalRotScaleTrans(rot_range=[0.0, 0.0], scale_ratio_range=[1.0, 1.0])(sample)
