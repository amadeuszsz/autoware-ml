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

"""Camera image resizing and cropping transforms."""

from __future__ import annotations

import cv2
import numpy as np

from autoware_ml.datamodule.samples.sample import Sample
from autoware_ml.transforms.base import BaseTransform
from autoware_ml.transforms.camera.utils import copy_calibration_data
from autoware_ml.utils.calibration import CalibrationData


class CropAndScale(BaseTransform):
    """Crop and scale augmentation for the calibration image.

    A random region of the image is cropped and resized back to the original resolution, and
    the projection intrinsics are updated so the lidar projection stays consistent.
    """

    _required_fields = ["calibration"]

    def __init__(self, *, p: float = 0.5, crop_ratio: float = 0.8) -> None:
        """Initialize the CropAndScale transform.

        Args:
            p: Probability of applying the transform.
            crop_ratio: Minimum fraction of the image kept when cropping.
        """
        self.p = p
        self.crop_ratio = crop_ratio

    def transform(self, sample: Sample) -> Sample:
        """Apply a random crop and scale to the calibration image.

        Args:
            sample: Sample with a loaded calibration image.

        Returns:
            Sample with the cropped image and the updated calibration data.
        """
        calibration = sample.calibration
        if calibration.image is None:
            raise ValueError("CropAndScale requires a loaded calibration image.")
        image = calibration.image.astype(np.uint8)

        height, width = image.shape[:2]
        max_center_noise = (1.0 - self.crop_ratio) / 2.0
        crop_center_noise_h = self._signed_random(0, max_center_noise)
        crop_center_noise_w = self._signed_random(0, max_center_noise)
        crop_center = np.array(
            [height * (1 + crop_center_noise_h) / 2, width * (1 + crop_center_noise_w) / 2]
        )

        max_noise = max(abs(crop_center_noise_h), abs(crop_center_noise_w))
        scale_noise = np.random.uniform(self.crop_ratio, 1.0 - max_noise)
        scaled_h, scaled_w = height * scale_noise, width * scale_noise

        start_h = int(crop_center[0] - scaled_h / 2)
        end_h = int(crop_center[0] + scaled_h / 2)
        start_w = int(crop_center[1] - scaled_w / 2)
        end_w = int(crop_center[1] + scaled_w / 2)

        start_h, end_h = max(0, start_h), min(height, end_h)
        start_w, end_w = max(0, start_w), min(width, end_w)

        cropped_image = image[start_h:end_h, start_w:end_w]
        resized_image = cv2.resize(cropped_image, (width, height))

        data = copy_calibration_data(calibration.data)
        self._update_camera_matrix(data, start_w, start_h, end_w, width)

        calibration = calibration.model_copy(
            update={"image": resized_image.astype(np.float32), "data": data}
        )
        return sample.replace(calibration=calibration)

    def _update_camera_matrix(
        self,
        calibration_data: CalibrationData,
        start_w: int,
        start_h: int,
        end_w: int,
        width: int,
    ) -> None:
        """Update the projection intrinsics of a copied calibration data instance.

        Args:
            calibration_data: Calibration data copy whose new camera matrix is updated.
            start_w: Left edge of the crop in pixels.
            start_h: Top edge of the crop in pixels.
            end_w: Right edge of the crop in pixels.
            width: Width of the original image in pixels.
        """
        scale_factor = width / (end_w - start_w)
        calibration_data.new_camera_matrix[0, 0] *= scale_factor
        calibration_data.new_camera_matrix[1, 1] *= scale_factor
        calibration_data.new_camera_matrix[0, 2] = (
            calibration_data.new_camera_matrix[0, 2] - start_w
        ) * scale_factor
        calibration_data.new_camera_matrix[1, 2] = (
            calibration_data.new_camera_matrix[1, 2] - start_h
        ) * scale_factor

    def _signed_random(self, min_value: float, max_value: float) -> float:
        """Sample a value from [min_value, max_value] with a random sign.

        Args:
            min_value: Lower bound of the magnitude.
            max_value: Upper bound of the magnitude.

        Returns:
            The signed sample.
        """
        sign = 1 if np.random.random() < 0.5 else -1
        return sign * np.random.uniform(min_value, max_value)
