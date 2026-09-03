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

"""Tests for the camera image loading transforms."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from autoware_ml.datamodule.samples.calibration import CalibrationSample
from autoware_ml.datamodule.samples.sample import Sample
from autoware_ml.testing.factories import (
    make_camera_frame,
    make_lidar_frame,
    make_record,
    make_sample,
)
from autoware_ml.transforms.camera.loading import LoadImageFromFile, LoadMultiViewImagesFromFiles
from autoware_ml.utils.calibration import CalibrationData


def _rotation_z(angle: float) -> np.ndarray:
    cos, sin = np.cos(angle), np.sin(angle)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:2, :2] = [[cos, -sin], [sin, cos]]
    return matrix


def _pose(angle: float, translation: tuple[float, float, float]) -> np.ndarray:
    matrix = _rotation_z(angle)
    matrix[:3, 3] = translation
    return matrix


def _write_image(
    data_root: Path, relative_path: str, height: int, width: int, bgr: tuple[int, int, int]
) -> None:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[..., 0] = bgr[0]
    image[..., 1] = bgr[1]
    image[..., 2] = bgr[2]
    path = data_root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def _make_multiview_sample(tmp_path: Path) -> Sample:
    lidar_frame = make_lidar_frame(
        sensor_to_ego=_pose(0.1, (1.0, 0.0, 1.5)),
        ego_to_global=_pose(0.3, (100.0, 50.0, 0.0)),
    )
    front = make_camera_frame(
        channel_name="CAM_FRONT",
        image_path="cams/front.png",
        width=16,
        height=8,
        intrinsic=np.array([[100.0, 0.0, 8.0], [0.0, 100.0, 4.0], [0.0, 0.0, 1.0]]),
        camera_to_ego=_pose(-0.2, (2.0, 0.5, 1.0)),
        ego_to_global=_pose(0.35, (101.0, 50.5, 0.0)),
    )
    left = make_camera_frame(
        channel_name="CAM_LEFT",
        image_path="cams/left.png",
        width=16,
        height=8,
        intrinsic=np.array([[80.0, 0.0, 8.0], [0.0, 80.0, 4.0], [0.0, 0.0, 1.0]]),
        camera_to_ego=_pose(1.2, (0.0, 1.0, 1.0)),
        ego_to_global=_pose(0.32, (100.4, 50.2, 0.0)),
    )
    _write_image(tmp_path, "cams/front.png", 8, 16, (10, 20, 30))
    _write_image(tmp_path, "cams/left.png", 8, 16, (40, 50, 60))
    record = make_record(lidar_frames=[lidar_frame], camera_frames=[front, left])
    return make_sample(record=record, data_root=str(tmp_path))


def _make_calibration_sample(tmp_path: Path, camera_name: str) -> Sample:
    sample = _make_multiview_sample(tmp_path)
    data = CalibrationData(
        camera_matrix=np.eye(3, dtype=np.float32),
        distortion_coefficients=np.zeros(5, dtype=np.float32),
        lidar_to_camera_transformation=np.eye(4, dtype=np.float32),
    )
    calibration = CalibrationSample(data=data, camera_name=camera_name)
    return sample.replace(calibration=calibration)


class TestLoadMultiViewImagesFromFiles:
    def test_matrix_composition_with_distinct_ego_poses(self, tmp_path: Path) -> None:
        sample = _make_multiview_sample(tmp_path)
        transform = LoadMultiViewImagesFromFiles(camera_order=["CAM_FRONT", "CAM_LEFT"])

        output = transform(sample)

        images = output.images
        assert images.camera_names == ("CAM_FRONT", "CAM_LEFT")
        keyframe = sample.record.lidar_frames[0]
        for index, channel in enumerate(images.camera_names):
            frame = next(
                f for f in sample.record.camera_frames if f.camera_sensor_channel_name == channel
            )
            expected = (
                np.linalg.inv(frame.camera_sensor_to_ego_pose_matrix)
                @ np.linalg.inv(frame.camera_frame_ego_pose_to_global_matrix)
                @ keyframe.lidar_frame_ego_pose_to_global_matrix
                @ keyframe.lidar_sensor_to_ego_pose_matrix
            )
            assert np.allclose(images.lidar2cam[index], expected, atol=1e-5)
            intrinsics = np.eye(4, dtype=np.float32)
            intrinsics[:3, :3] = frame.camera_intrinsic_matrix
            assert np.allclose(images.camera_intrinsics[index], intrinsics)
            assert np.allclose(
                images.lidar2img[index], intrinsics @ images.lidar2cam[index], atol=1e-4
            )
        # The two ego poses differ, so the naive shared ego composition must not match
        front = sample.record.camera_frames[0]
        naive = np.linalg.inv(front.camera_sensor_to_ego_pose_matrix) @ (
            keyframe.lidar_sensor_to_ego_pose_matrix
        )
        assert not np.allclose(images.lidar2cam[0], naive, atol=1e-3)

    def test_images_are_normalized_rgb_channel_first(self, tmp_path: Path) -> None:
        sample = _make_multiview_sample(tmp_path)
        transform = LoadMultiViewImagesFromFiles(camera_order=["CAM_FRONT", "CAM_LEFT"])

        output = transform(sample)

        images = output.images.images
        assert images.shape == (2, 3, 8, 16)
        assert images.dtype == np.float32
        # The front image was written as BGR (10, 20, 30), loaded as RGB and divided by 255
        assert np.allclose(images[0, 0], 30 / 255.0)
        assert np.allclose(images[0, 1], 20 / 255.0)
        assert np.allclose(images[0, 2], 10 / 255.0)

    def test_missing_channel_raises(self, tmp_path: Path) -> None:
        sample = _make_multiview_sample(tmp_path)
        transform = LoadMultiViewImagesFromFiles(camera_order=["CAM_FRONT", "CAM_BACK"])

        with pytest.raises(ValueError, match="CAM_BACK"):
            transform(sample)


class TestLoadImageFromFile:
    def test_loads_the_calibration_camera_image(self, tmp_path: Path) -> None:
        sample = _make_calibration_sample(tmp_path, camera_name="CAM_LEFT")

        output = LoadImageFromFile()(sample)

        image = output.calibration.image
        assert image is not None
        assert image.shape == (8, 16, 3)
        assert image.dtype == np.float32
        # The left image was written as BGR (40, 50, 60) and is loaded as RGB by default
        assert np.allclose(image[0, 0], [60.0, 50.0, 40.0])

    def test_bgr_color_type_keeps_the_stored_order(self, tmp_path: Path) -> None:
        sample = _make_calibration_sample(tmp_path, camera_name="CAM_LEFT")

        output = LoadImageFromFile(color_type="bgr")(sample)

        assert np.allclose(output.calibration.image[0, 0], [40.0, 50.0, 60.0])

    def test_missing_channel_raises(self, tmp_path: Path) -> None:
        sample = _make_calibration_sample(tmp_path, camera_name="CAM_BACK")

        with pytest.raises(ValueError, match="CAM_BACK"):
            LoadImageFromFile()(sample)

    def test_requires_seeded_calibration(self, tmp_path: Path) -> None:
        sample = _make_multiview_sample(tmp_path)

        with pytest.raises(ValueError, match="calibration"):
            LoadImageFromFile()(sample)
