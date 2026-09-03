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

"""Image space photometric transforms."""

from __future__ import annotations

import cv2
import numpy as np

from autoware_ml.datamodule.samples.sample import Sample
from autoware_ml.transforms.base import BaseTransform


class PhotometricDistortion(BaseTransform):
    """Apply random brightness, contrast, saturation, and hue to the calibration image.

    The distortions run in HSV space on the color channels of the calibration image, before the
    lidar fusion appends the depth and intensity channels.
    """

    _required_fields = ["calibration"]

    def __init__(
        self,
        *,
        p: float = 0.5,
        brightness: float = 0.0,
        contrast: float = 0.0,
        saturation: float = 0.0,
        hue: float = 0.0,
    ) -> None:
        """Initialize the PhotometricDistortion transform.

        Args:
            p: Probability of applying the augmentation.
            brightness: Max brightness deviation in [0, 1].
            contrast: Max contrast deviation in [0, 1].
            saturation: Max saturation deviation in [0, 1].
            hue: Max hue deviation in [0, 0.5].
        """
        self.p = p
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def transform(self, sample: Sample) -> Sample:
        """Apply the photometric distortion to the calibration image.

        Args:
            sample: Sample with a loaded calibration image.

        Returns:
            Sample with the distorted image.
        """
        calibration = sample.calibration
        if calibration.image is None:
            raise ValueError("PhotometricDistortion requires a loaded calibration image.")
        image = calibration.image.astype(np.uint8)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)

        if self.brightness > 0:
            hsv[..., 2] *= np.random.uniform(1 - self.brightness, 1 + self.brightness)

        if self.saturation > 0:
            hsv[..., 1] *= np.random.uniform(1 - self.saturation, 1 + self.saturation)

        if self.contrast > 0:
            factor = np.random.uniform(1 - self.contrast, 1 + self.contrast)
            hsv[..., 2] = (hsv[..., 2] - 127.5) * factor + 127.5

        if self.hue > 0:
            hsv[..., 0] += np.random.uniform(-self.hue, self.hue) * 179.0
            hsv[..., 0] = np.mod(hsv[..., 0], 180.0)

        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        distorted = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        calibration = calibration.model_copy(update={"image": distorted.astype(np.float32)})
        return sample.replace(calibration=calibration)
