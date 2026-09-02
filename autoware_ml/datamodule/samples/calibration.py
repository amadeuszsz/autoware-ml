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

from __future__ import annotations

import numpy as np
from jaxtyping import Float32
from pydantic import BaseModel, ConfigDict, model_validator

from autoware_ml.utils.calibration import CalibrationData, CalibrationStatus


class CalibrationSample(BaseModel):
    """
    Working state of the calibration status task for one sample. The calibration pipeline loads
    a single camera image, perturbs the calibration, and fuses the lidar points into the image.

    Attributes:
      data: Camera intrinsics and lidar to camera calibration of the sample. The calibration
        transforms update this object as they run.
      camera_name: Camera channel name the calibration belongs to.
      image: Loaded camera image in height, width, channels layout. None until the image is
        loaded.
      fused_image: Camera image fused with projected lidar depth and intensity channels. None
        until the fusion runs.
      status: Ground truth calibration status. None until the misalignment transform decides it.
      affine_transform: Affine image space transform applied after the fusion, when one ran.
    """

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    data: CalibrationData
    camera_name: str
    image: Float32[np.ndarray, "height width channels"] | None = None
    fused_image: Float32[np.ndarray, "height width fused_channels"] | None = None
    status: CalibrationStatus | None = None
    affine_transform: Float32[np.ndarray, "3 3"] | None = None

    @model_validator(mode="after")
    def validate_calibration(self) -> CalibrationSample:
        """
        Validate the image arrays.

        Returns:
          CalibrationSample: The validated calibration sample.
        """

        if self.image is not None and self.image.ndim != 3:
            raise ValueError(f"The camera image must be 3D, got shape {self.image.shape}.")
        if self.fused_image is not None and self.fused_image.ndim != 3:
            raise ValueError(f"The fused image must be 3D, got shape {self.fused_image.shape}.")
        if self.affine_transform is not None and self.affine_transform.shape != (3, 3):
            raise ValueError(
                f"The affine transform must have shape (3, 3), got {self.affine_transform.shape}."
            )
        return self
