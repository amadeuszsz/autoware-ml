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

"""Vectorized map elements around every frame, the supervision of online mapping.

A sample holds the map elements near its frame, in the frame of its raw point cloud. Collation
moves them into the frame of the augmented cloud with the lidar-space augmentation of the sample,
cuts them to the mapped region and resamples every piece to a fixed number of points.
"""

from __future__ import annotations

from typing import NamedTuple
from collections.abc import Sequence

from jaxtyping import Bool, Float32, Int64
import numpy as np
import shapely
import torch
from torch import Tensor

from autoware_ml.dataclasses.geometry.transformation import LiDARTransformationSample
from autoware_ml.geometry.vector_map import FrameMapElements


class VectorMapSpec(NamedTuple):
    """Layout of the map supervision.

    Attributes:
        roi: Mapped region of the augmented frame ``(x_min, y_min, x_max, y_max)``.
        num_points: Points every element is resampled to.
        min_length: Shortest piece the region cut keeps, in meters.
        crop_half_size: Half the side of the square of the raw frame the samples hold. The
            augmentation must keep the region inside it.
    """

    roi: tuple[float, float, float, float]
    num_points: int
    min_length: float
    crop_half_size: float


class OnlineMapGTSample(NamedTuple):
    """Map elements near one frame, in the frame of its raw point cloud.

    Attributes:
        elements: The elements, cut to the crop square of the spec.
        spec: Layout of the supervision.
    """

    elements: FrameMapElements
    spec: VectorMapSpec


class OnlineMapGTBatch(NamedTuple):
    """Map elements of a batch, in the frames of the augmented point clouds.

    Attributes:
        points: Resampled points of every padded element slot.
        labels: Class of every slot, -1 for padding.
        closed: Whether an element is a closed ring, its first and last point equal.
    """

    points: Float32[Tensor, "batch_size max_num_elements num_points 2"]
    labels: Int64[Tensor, "batch_size max_num_elements"]
    closed: Bool[Tensor, "batch_size max_num_elements"]

    @staticmethod
    def collate_gt_samples(
        online_map_gt_samples: Sequence[OnlineMapGTSample],
        lidar_transformation_samples: Sequence[LiDARTransformationSample | None],
    ) -> OnlineMapGTBatch:
        """Move the elements into the augmented frames, cut, resample and pad them.

        Args:
            online_map_gt_samples: Elements of every sample, in batch order.
            lidar_transformation_samples: Lidar-space augmentation of every sample, None for
                samples that were not augmented.

        Returns:
            OnlineMapGTBatch: The collated elements.

        Raises:
            ValueError: If the sequences disagree on the batch size or the samples on the spec.
        """
        if len(online_map_gt_samples) != len(lidar_transformation_samples):
            raise ValueError(
                f"Got {len(online_map_gt_samples)} map samples and "
                f"{len(lidar_transformation_samples)} transformations."
            )
        specs = {sample.spec for sample in online_map_gt_samples}
        if len(specs) != 1:
            raise ValueError(f"The samples of a batch need one map spec, got {specs}.")
        spec = specs.pop()
        frames = [
            vectorize(
                sample.elements,
                np.eye(4)
                if transformation is None
                else transformation.transformation_matrix.double().numpy(),
                spec,
            )
            for sample, transformation in zip(
                online_map_gt_samples, lidar_transformation_samples, strict=True
            )
        ]
        num_slots = max(len(labels) for _, labels, _ in frames)
        points = torch.zeros(len(frames), num_slots, spec.num_points, 2)
        labels = torch.full((len(frames), num_slots), -1, dtype=torch.long)
        closed = torch.zeros(len(frames), num_slots, dtype=torch.bool)
        for index, (frame_points, frame_labels, frame_closed) in enumerate(frames):
            count = len(frame_labels)
            points[index, :count] = torch.from_numpy(frame_points).float()
            labels[index, :count] = torch.from_numpy(frame_labels)
            closed[index, :count] = torch.from_numpy(frame_closed)
        return OnlineMapGTBatch(points=points, labels=labels, closed=closed)

    def select_samples(self, mask: Bool[Tensor, " batch_size"]) -> OnlineMapGTBatch:
        """Keep the elements of the samples a mask selects.

        Args:
            mask: True for every sample to keep.

        Returns:
            OnlineMapGTBatch: The elements of the selected samples.
        """
        return OnlineMapGTBatch(*(field[mask] for field in self))

    def to_device(self, device: torch.device) -> OnlineMapGTBatch:
        """Move every tensor to ``device``.

        Args:
            device: Target device.

        Returns:
            OnlineMapGTBatch: The batch on ``device``.
        """
        return OnlineMapGTBatch(*(field.to(device) for field in self))


def vectorize(
    elements: FrameMapElements, matrix: np.ndarray, spec: VectorMapSpec
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Move the elements of one frame by its augmentation, cut them to the region, resample.

    An area is cut as an area, so its outline closes along the region border. A line the
    region cuts in several pieces yields one element per piece.

    Args:
        elements: Elements in the raw frame.
        matrix: Augmentation from the raw to the augmented frame ``(4, 4)``.
        spec: Layout of the supervision.

    Returns:
        Points ``(M, num_points, 2)``, labels ``(M,)`` and whether every element is closed.

    Raises:
        ValueError: If the augmentation moves part of the region outside the crop square.
    """
    x_min, y_min, x_max, y_max = spec.roi
    corners = np.array([[x_min, y_min], [x_min, y_max], [x_max, y_min], [x_max, y_max]])
    inverse = np.linalg.inv(matrix)
    raw_corners = corners @ inverse[:2, :2].T + inverse[:2, 3]
    if np.abs(raw_corners).max() > spec.crop_half_size:
        raise ValueError(
            f"The augmentation moves the map region to {raw_corners.tolist()}, outside the "
            f"crop square of half size {spec.crop_half_size}. Raise the crop margin."
        )
    rotation, translation = matrix[:2, :2], matrix[:2, 3]
    moved = shapely.transform(elements.geometries(), lambda xy: xy @ rotation.T + translation)
    parts, index = shapely.get_parts(
        shapely.clip_by_rect(moved, x_min, y_min, x_max, y_max), return_index=True
    )
    type_ids = shapely.get_type_id(parts)
    outlines = np.where(
        type_ids == shapely.GeometryType.POLYGON, shapely.get_exterior_ring(parts), parts
    )
    keep = (
        np.isin(type_ids, (shapely.GeometryType.LINESTRING, shapely.GeometryType.POLYGON))
        & ~shapely.is_empty(parts)
        & (shapely.length(outlines) >= spec.min_length)
    )
    outlines = outlines[keep]
    distances = np.linspace(0.0, 1.0, spec.num_points)[None]
    resampled = shapely.line_interpolate_point(outlines[:, None], distances, normalized=True)
    points = shapely.get_coordinates(resampled.ravel()).reshape(len(outlines), spec.num_points, 2)
    return points, elements.labels[index[keep]], shapely.is_closed(outlines)
