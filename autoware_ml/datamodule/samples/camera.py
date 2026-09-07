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


class ImageNormalization(BaseModel):
    """
    Normalization applied to the images of a sample.

    Attributes:
      mean: Per channel mean.
      std: Per channel standard deviation.
      to_rgb: Whether the channels were converted from BGR to RGB.
    """

    model_config = ConfigDict(frozen=True, strict=True)

    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    to_rgb: bool


class ImageSet(BaseModel):
    """
    Multiview camera images of one sample together with their projection matrices. All camera
    indexed arrays share the camera ordering of camera_names.

    Attributes:
      images: Images of every camera in channel first layout.
      camera_names: Camera channel name of every image.
      camera_intrinsics: Intrinsic matrix of every camera, padded to 4x4.
      lidar2cam: Transformation matrix from the lidar frame to every camera frame.
      lidar2img: Projection matrix from the lidar frame to every image plane.
      ori_camera_intrinsics: Intrinsic matrices before any image space transform, when an image
        space transform changed them.
      img_aug_matrix: Image space augmentation matrix of every camera, when an image space
        augmentation ran.
      pad_shape: Height and width of the images after padding, when padding ran.
      normalization: Normalization applied to the images, when normalization ran.
    """

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    images: Float32[np.ndarray, "num_cameras channels height width"]
    camera_names: tuple[str, ...]
    camera_intrinsics: Float32[np.ndarray, "num_cameras 4 4"]
    lidar2cam: Float32[np.ndarray, "num_cameras 4 4"]
    lidar2img: Float32[np.ndarray, "num_cameras 4 4"]
    ori_camera_intrinsics: Float32[np.ndarray, "num_cameras 4 4"] | None = None
    img_aug_matrix: Float32[np.ndarray, "num_cameras 4 4"] | None = None
    pad_shape: tuple[int, int] | None = None
    normalization: ImageNormalization | None = None

    @model_validator(mode="after")
    def validate_image_set(self) -> ImageSet:
        """
        Validate the camera indexed arrays against the camera names.

        Returns:
          ImageSet: The validated image set.
        """

        num_cameras = len(self.camera_names)
        if self.images.ndim != 4 or self.images.shape[0] != num_cameras:
            raise ValueError(
                f"Images must have shape ({num_cameras}, channels, height, width), "
                f"got {self.images.shape}."
            )
        for field_name in ("camera_intrinsics", "lidar2cam", "lidar2img"):
            field_value = getattr(self, field_name)
            if field_value.shape[0] != num_cameras:
                raise ValueError(
                    f"{field_name} covers {field_value.shape[0]} cameras but the sample "
                    f"declares {num_cameras} camera names."
                )
        return self

    def __len__(self) -> int:
        """
        Get the number of cameras.

        Returns:
          int: Number of cameras.
        """

        return len(self.camera_names)
