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

"""Camera distortion correction transforms."""

from __future__ import annotations

import cv2
import numpy as np

from autoware_ml.datamodule.samples.sample import Sample
from autoware_ml.transforms.base import BaseTransform
from autoware_ml.transforms.camera.utils import copy_calibration_data


class UndistortImage(BaseTransform):
    """Undistort the calibration image using the camera calibration parameters.

    Images stored without distortion carry all zero coefficients, and the transform passes them
    through unchanged. After undistortion the coefficients are zeroed and the optimal new camera
    matrix replaces the projection intrinsics.
    """

    _required_fields = ["calibration"]

    def __init__(self, *, alpha: float = 0.0) -> None:
        """Initialize the UndistortImage transform.

        Args:
            alpha: Free scaling parameter passed to OpenCV undistortion. 0.0 crops invalid
                pixels, while 1.0 retains the full field of view.
        """
        self.alpha = alpha

    def transform(self, sample: Sample) -> Sample:
        """Undistort the calibration image and update the calibration data.

        Args:
            sample: Sample with a loaded calibration image.

        Returns:
            Sample with the undistorted image and the updated calibration data.
        """
        calibration = sample.calibration
        if calibration.image is None:
            raise ValueError("UndistortImage requires a loaded calibration image.")
        if not np.any(calibration.data.distortion_coefficients):
            return sample

        image = calibration.image.astype(np.uint8)
        height, width = image.shape[:2]
        data = copy_calibration_data(calibration.data)
        data.new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
            data.camera_matrix,
            data.distortion_coefficients,
            (width, height),
            self.alpha,
            (width, height),
        )
        image = cv2.undistort(
            image,
            data.camera_matrix,
            data.distortion_coefficients,
            newCameraMatrix=data.new_camera_matrix,
        )
        data.distortion_coefficients = np.zeros_like(data.distortion_coefficients)
        calibration = calibration.model_copy(
            update={"image": image.astype(np.float32), "data": data}
        )
        return sample.model_copy(update={"calibration": calibration})
