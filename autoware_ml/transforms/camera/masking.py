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

"""Image masking transforms to support ModelGTSample."""

from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt
import torch

from autoware_ml.dataclasses.batch.sample_batch import ModelGTSample
from autoware_ml.geometry.cameras.base_images import BaseImages
from autoware_ml.transforms.base import BaseTransform


class GridMask(BaseTransform):
    """Blank out a rotated regular grid of stripes in every image."""

    _required_keys = ["camera_image_data"]

    def __init__(self, probability: float = 0.7, ratio: float = 0.5, rotate: int = 1) -> None:
        """Initialize the GridMask transform.

        Args:
            probability: Probability of applying the transform.
            ratio: Fraction of each grid period that is masked out.
            rotate: Maximum absolute rotation in degrees applied to the mask.
        """
        super().__init__(probability=probability)
        self.ratio = ratio
        self.rotate = rotate

    def transform(self, model_gt_sample: ModelGTSample) -> ModelGTSample:
        """Mask every image of the sample with a grid pattern.

        Args:
            model_gt_sample: ModelGTSample instance holding loaded images.

        Returns:
            Updated ModelGTSample instance with masked images.
        """
        # This is checked in the _validate_required_keys()
        camera_image_data: BaseImages = model_gt_sample.camera_image_data  # type: ignore[reportOptionalMemberAccess]

        masked = torch.stack(
            [
                image * torch.from_numpy(self.build_mask(image.shape[1], image.shape[2]))
                for image in camera_image_data.images
            ]
        )
        return model_gt_sample._replace(
            camera_image_data=BaseImages.model_validate(
                camera_image_data.model_copy(update={"images": masked})
            )
        )

    def build_mask(self, height: int, width: int) -> npt.NDArray[np.float32]:
        """Build one grid mask covering an image of the given size.

        Args:
            height: Height of the image.
            width: Width of the image.

        Returns:
            npt.NDArray[np.float32]: The mask, zero on the masked stripes.
        """
        period = np.random.randint(32, max(33, min(height, width)))
        cut = max(1, int(period * self.ratio))
        mask = np.ones((height, width), dtype=np.float32)
        for x in range(np.random.randint(period), width, period):
            mask[:, x : x + cut] = 0
        for y in range(np.random.randint(period), height, period):
            mask[y : y + cut, :] = 0
        if self.rotate > 0:
            angle = np.random.uniform(-self.rotate, self.rotate)
            rotation = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
            mask = cv2.warpAffine(mask, rotation, (width, height))
        return mask[None]
