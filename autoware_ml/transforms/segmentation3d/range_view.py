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

"""Range view transforms for point cloud segmentation."""

from __future__ import annotations

import numpy as np

from autoware_ml.datamodule.samples.sample import Sample
from autoware_ml.transforms.base import BaseTransform
from autoware_ml.transforms.segmentation3d.utils import project_range


class RangeInterpolation(BaseTransform):
    """Fill empty range image pixels with horizontal interpolation.

    Interpolated points are appended after the stored points, so the leading current frame
    block of the cloud stays intact and num_current_points keeps marking the real points.
    Every feature column of an interpolated point is the mean of its two horizontal
    neighbors. Its label is the shared neighbor label, or ignore_index when the neighbors
    disagree. When the sample carries no segmentation labels only the points are extended.
    """

    _required_fields = ["points"]

    def __init__(
        self,
        *,
        height: int,
        width: int,
        fov_up: float,
        fov_down: float,
        ignore_index: int,
    ) -> None:
        """Initialize the RangeInterpolation transform.

        Args:
            height: Range image height in pixels.
            width: Range image width in pixels.
            fov_up: Upper vertical field of view in degrees.
            fov_down: Lower vertical field of view in degrees.
            ignore_index: Label used when interpolated neighbors disagree.
        """
        self.height = height
        self.width = width
        self.fov_up_rad = np.deg2rad(fov_up)
        self.fov_down_rad = np.deg2rad(fov_down)
        self.ignore_index = ignore_index

    def transform(self, sample: Sample) -> Sample:
        """Append horizontally interpolated range view points.

        Args:
            sample: Sample with a loaded point cloud and optional segmentation labels.

        Returns:
            Sample with the interpolated points and labels appended.
        """
        points = sample.points
        features = points.features
        proj_y, proj_x = project_range(
            points.coord, self.height, self.width, self.fov_up_rad, self.fov_down_rad
        )

        depth = np.linalg.norm(features[:, :3], ord=2, axis=1)
        order = np.argsort(depth)[::-1]

        proj_image = np.full(
            (self.height, self.width, features.shape[1]), fill_value=-1.0, dtype=np.float32
        )
        proj_mask = np.zeros((self.height, self.width), dtype=bool)
        proj_image[proj_y[order], proj_x[order]] = features[order]
        proj_mask[proj_y[order], proj_x[order]] = True

        proj_labels = None
        if sample.segment is not None:
            labels = sample.segment.labels
            proj_labels = np.full(
                (self.height, self.width), fill_value=self.ignore_index, dtype=np.int64
            )
            proj_labels[proj_y[order], proj_x[order]] = labels[order]

        # Vectorized interpolation over all empty pixels that have both a filled left and
        # right neighbor in the same row. For a 128 by 4096 grid the pixel count is around
        # 500 K, making an element wise Python loop significantly slower than this numpy
        # approach.
        inner = proj_mask[:, 1:-1]  # shape (H, W-2)
        can_interp = ~inner & proj_mask[:, :-2] & proj_mask[:, 2:]
        interp_rows, interp_cols_inner = np.where(can_interp)
        interp_cols = interp_cols_inner + 1  # shift back to full-width indices

        if interp_rows.size == 0:
            return sample

        new_features = 0.5 * (
            proj_image[interp_rows, interp_cols - 1] + proj_image[interp_rows, interp_cols + 1]
        )
        extended_points = points.model_copy(
            update={"features": np.concatenate([features, new_features.astype(np.float32)], axis=0)}
        )
        update = {"points": extended_points}

        if proj_labels is not None:
            left_labels = proj_labels[interp_rows, interp_cols - 1]
            right_labels = proj_labels[interp_rows, interp_cols + 1]
            same = left_labels == right_labels
            new_labels = np.where(same, left_labels, self.ignore_index).astype(np.int64)
            update["segment"] = sample.segment.model_copy(
                update={"labels": np.concatenate([sample.segment.labels, new_labels], axis=0)}
            )

        return sample.model_copy(update=update)
