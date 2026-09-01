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

"""Camera image masking transforms."""

from __future__ import annotations

import cv2
import numpy as np
from jaxtyping import Num

from autoware_ml.datamodule.samples.sample import Sample
from autoware_ml.transforms.base import BaseTransform


class GridMask(BaseTransform):
    """Apply grid masking augmentation to every image of the image set."""

    _required_fields = ["images"]

    def __init__(self, *, p: float = 0.7, ratio: float = 0.5, rotate: int = 1) -> None:
        """Initialize the GridMask transform.

        Args:
            p: Probability of applying the transform.
            ratio: Fraction of each grid period that is masked out.
            rotate: Maximum absolute rotation in degrees applied to the mask.
        """
        self.p = p
        self.ratio = ratio
        self.rotate = rotate

    def transform(self, sample: Sample) -> Sample:
        """Mask every camera image with a regular grid pattern.

        Args:
            sample: Sample with a loaded image set.

        Returns:
            Sample with the masked images.
        """
        masked = []
        for image in sample.images.images:
            image_hwc = np.transpose(image, (1, 2, 0))
            masked.append(np.transpose(self._grid_mask(image_hwc), (2, 0, 1)))
        images = sample.images.model_copy(update={"images": np.stack(masked, axis=0)})
        return sample.model_copy(update={"images": images})

    def _grid_mask(
        self, image: Num[np.ndarray, "height width channels"]
    ) -> Num[np.ndarray, "height width channels"]:
        """Apply the grid mask to a single image.

        Args:
            image: Image in height, width, channels layout.

        Returns:
            Num[np.ndarray, "height width channels"]: The masked image.
        """
        height, width = image.shape[:2]
        period = np.random.randint(32, max(33, min(height, width)))
        cut = max(1, int(period * self.ratio))
        mask = np.ones((height, width), dtype=np.float32)
        offset_x = np.random.randint(period)
        offset_y = np.random.randint(period)
        for x in range(offset_x, width, period):
            mask[:, x : x + cut] = 0
        for y in range(offset_y, height, period):
            mask[y : y + cut, :] = 0
        if self.rotate > 0:
            angle = np.random.uniform(-self.rotate, self.rotate)
            rotation = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
            mask = cv2.warpAffine(mask, rotation, (width, height))
        if image.ndim == 3:
            mask = mask[..., None]
        return (image.astype(np.float32) * mask).astype(image.dtype)
