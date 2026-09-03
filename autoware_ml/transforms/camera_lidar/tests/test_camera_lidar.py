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

"""Tests for the camera and lidar fusion transforms."""

from pathlib import Path

import numpy as np
import pytest

from autoware_ml.datamodule.samples.calibration import CalibrationSample
from autoware_ml.datamodule.samples.camera import ImageSet
from autoware_ml.datamodule.samples.point_cloud import PointCloud
from autoware_ml.datamodule.samples.sample import Sample
from autoware_ml.testing.factories import make_camera_frame, make_record, make_sample
from autoware_ml.transforms.camera_lidar.camera_lidar import (
    Affine,
    CalibrationMisalignment,
    ImageAug3D,
    LidarCameraFusion,
    SaveFusionPreview,
)
from autoware_ml.types.geometry import PointFeatureName
from autoware_ml.utils.calibration import CalibrationData, CalibrationStatus

POINT_FEATURES = (
    PointFeatureName.X,
    PointFeatureName.Y,
    PointFeatureName.Z,
    PointFeatureName.INTENSITY,
)


def _make_calibration_data() -> CalibrationData:
    return CalibrationData(
        camera_matrix=np.array(
            [[100.0, 0.0, 32.0], [0.0, 100.0, 24.0], [0.0, 0.0, 1.0]], dtype=np.float32
        ),
        distortion_coefficients=np.zeros(5, dtype=np.float32),
        lidar_to_camera_transformation=np.eye(4, dtype=np.float32),
    )


def _make_calibration_sample(
    *, with_image: bool = True, points: np.ndarray | None = None
) -> Sample:
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, size=(48, 64, 3)).astype(np.float32) if with_image else None
    calibration = CalibrationSample(
        data=_make_calibration_data(), camera_name="CAM_FRONT", image=image
    )
    point_cloud = None
    if points is not None:
        point_cloud = PointCloud(
            features=points.astype(np.float32),
            feature_names=POINT_FEATURES,
            num_current_points=points.shape[0],
        )
    record = make_record(camera_frames=[make_camera_frame(image_path="db/cam/42.jpg")])
    sample = make_sample(record=record, points=point_cloud)
    return sample.replace(calibration=calibration)


class TestCalibrationMisalignment:
    def test_skip_marks_the_sample_calibrated(self) -> None:
        sample = _make_calibration_sample()
        transform = CalibrationMisalignment(p=0.0, activate_roll=True)

        output = transform(sample)

        assert output.calibration.status == CalibrationStatus.CALIBRATED
        assert np.allclose(
            output.calibration.data.lidar_to_camera_transformation,
            sample.calibration.data.lidar_to_camera_transformation,
        )

    def test_applied_misalignment_perturbs_the_calibration(self) -> None:
        sample = _make_calibration_sample()
        transform = CalibrationMisalignment(
            p=1.0,
            activate_roll=True,
            min_roll_neg=5.0,
            max_roll_neg=10.0,
            min_roll_pos=5.0,
            max_roll_pos=10.0,
        )

        output = transform(sample)

        assert output.calibration.status == CalibrationStatus.MISCALIBRATED
        assert output.calibration.data.noise is not None
        assert output.calibration.data.noise.shape == (4, 4)
        assert not np.allclose(
            output.calibration.data.lidar_to_camera_transformation,
            sample.calibration.data.lidar_to_camera_transformation,
        )

    def test_input_calibration_data_is_not_mutated(self) -> None:
        sample = _make_calibration_sample()
        transform = CalibrationMisalignment(
            p=1.0, activate_x=True, min_x_pos=0.5, max_x_pos=1.0, min_x_neg=0.5, max_x_neg=1.0
        )

        transform(sample)

        assert sample.calibration.data.noise is None
        assert np.allclose(
            sample.calibration.data.lidar_to_camera_transformation, np.eye(4, dtype=np.float32)
        )

    def test_negative_magnitude_raises(self) -> None:
        with pytest.raises(ValueError, match="must be >= 0"):
            CalibrationMisalignment(p=0.5, min_roll_neg=-1.0, max_roll_neg=5.0)

    def test_min_greater_than_max_raises(self) -> None:
        with pytest.raises(ValueError, match="must be <="):
            CalibrationMisalignment(p=0.5, min_roll_neg=5.0, max_roll_neg=1.0)


class TestLidarCameraFusion:
    def test_fused_image_shape_dtype_and_range(self) -> None:
        rng = np.random.default_rng(1)
        points = np.zeros((100, 4), dtype=np.float32)
        points[:, :3] = rng.uniform(-5.0, 5.0, size=(100, 3))
        points[:, 2] += 10.0
        points[:, 3] = rng.uniform(0.0, 255.0, size=100)
        sample = _make_calibration_sample(points=points)

        output = LidarCameraFusion()(sample)

        fused = output.calibration.fused_image
        assert fused.shape == (48, 64, 5)
        assert fused.dtype == np.float32
        assert fused.min() >= 0.0
        assert fused.max() <= 1.0

    def test_depth_and_intensity_channels_carry_the_projected_point(self) -> None:
        # One point at (0.5, 0.3, 10) projects to pixel (u, v) = (37, 27) with focal 100
        points = np.array([[0.5, 0.3, 10.0, 128.0]], dtype=np.float32)
        sample = _make_calibration_sample(points=points)

        output = LidarCameraFusion(max_depth=128.0, dilation_size=1)(sample)

        fused = output.calibration.fused_image
        assert np.isclose(fused[27, 37, 3], (255.0 * 10.0 / 128.0) / 255.0)
        assert np.isclose(fused[27, 37, 4], 128.0 / 255.0)
        # The dilation paints the 3x3 patch around the projection
        assert np.isclose(fused[26, 36, 3], (255.0 * 10.0 / 128.0) / 255.0)
        assert np.count_nonzero(fused[..., 3]) == 9

    def test_point_cloud_of_the_sample_stays_untouched(self) -> None:
        points = np.array([[0.5, 0.3, 10.0, 128.0]], dtype=np.float32)
        sample = _make_calibration_sample(points=points)

        output = LidarCameraFusion(ego_box=[-1.5, -1.0, -0.1, 5.7, 1.0, 3.1])(sample)

        assert np.array_equal(output.points.features, sample.points.features)

    def test_requires_a_point_cloud(self) -> None:
        sample = _make_calibration_sample()

        with pytest.raises(ValueError, match="points"):
            LidarCameraFusion()(sample)


class TestAffine:
    def test_skip_stores_the_identity_transform(self) -> None:
        sample = _make_calibration_sample()

        output = Affine(p=0.0)(sample)

        assert np.allclose(output.calibration.affine_transform, np.eye(3, dtype=np.float32))
        assert np.array_equal(output.calibration.image, sample.calibration.image)

    def test_applied_affine_stores_a_valid_transform(self) -> None:
        np.random.seed(0)
        sample = _make_calibration_sample()

        output = Affine(p=1.0, max_distortion=0.1)(sample)

        affine = output.calibration.affine_transform
        assert affine.shape == (3, 3)
        assert affine.dtype == np.float32
        assert np.allclose(affine[2, :], [0.0, 0.0, 1.0])
        assert output.calibration.image.shape == sample.calibration.image.shape


class TestSaveFusionPreview:
    def test_writes_status_suffixed_previews(self, tmp_path: Path) -> None:
        points = np.array([[0.5, 0.3, 10.0, 128.0]], dtype=np.float32)
        sample = _make_calibration_sample(points=points)
        sample = LidarCameraFusion()(sample)
        calibration = sample.calibration.model_copy(
            update={"status": CalibrationStatus.MISCALIBRATED}
        )
        sample = sample.replace(calibration=calibration)
        transform = SaveFusionPreview(p=1.0, out_dir=str(tmp_path / "previews"))

        output = transform(sample)

        assert output is sample
        assert (tmp_path / "previews" / "42_miscalibrated_depth.png").exists()
        assert (tmp_path / "previews" / "42_miscalibrated_intensity.png").exists()


class TestImageAug3D:
    def test_matrix_bookkeeping_in_eval_mode(self) -> None:
        eye = np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))
        lidar2cam = eye.copy()
        lidar2cam[:, 0, 3] = 0.5
        image_set = ImageSet(
            images=np.ones((2, 3, 8, 8), dtype=np.float32),
            camera_names=("CAM_FRONT", "CAM_LEFT"),
            camera_intrinsics=eye.copy(),
            lidar2cam=lidar2cam,
            lidar2img=lidar2cam.copy(),
        )
        sample = make_sample().replace(images=image_set)
        transform = ImageAug3D(
            final_dim=[6, 6], resize_lim=[1.0, 1.0], bot_pct_lim=[0.0, 0.0], training=False
        )

        output = transform(sample)

        images = output.images
        assert images.images.shape == (2, 3, 6, 6)
        assert images.img_aug_matrix.shape == (2, 4, 4)
        expected_aug = np.eye(4, dtype=np.float32)
        expected_aug[0, 2] = -1.0
        expected_aug[1, 2] = -2.0
        assert np.allclose(images.img_aug_matrix[0], expected_aug)
        assert np.allclose(images.camera_intrinsics, images.img_aug_matrix @ eye)
        assert np.allclose(images.ori_camera_intrinsics, eye)
        assert np.allclose(images.lidar2img, images.camera_intrinsics @ lidar2cam)
        assert np.array_equal(images.lidar2cam, lidar2cam)
