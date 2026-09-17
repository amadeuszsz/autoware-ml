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

"""Range view preprocessing of a batched point cloud.

These layers run on the collated batch rather than on a single sample, so the mixing ones
draw their second cloud from the batch instead of from a second read of the dataset.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from jaxtyping import Int64
from torch import Tensor


def project_range(
    points: Tensor, height: int, width: int, fov_up_rad: float, fov_down_rad: float
) -> tuple[Int64[Tensor, " num_points"], Int64[Tensor, " num_points"]]:
    """Project 3D points into range view row and column indices.

    Args:
        points: Points whose leading three features are the coordinates.
        height: Range image height in pixels.
        width: Range image width in pixels.
        fov_up_rad: Upper vertical field of view in radians.
        fov_down_rad: Lower vertical field of view in radians.

    Returns:
        tuple: Row and column index of every point.
    """
    depth = torch.linalg.vector_norm(points[:, :3], ord=2, dim=1).clamp(min=1e-6)
    yaw = -torch.atan2(points[:, 1], points[:, 0])
    pitch = torch.asin((points[:, 2] / depth).clamp(-1.0, 1.0))
    fov = abs(fov_down_rad) + abs(fov_up_rad)

    proj_x = 0.5 * (yaw / torch.pi + 1.0) * width
    proj_y = (1.0 - (pitch + abs(fov_down_rad)) / fov) * height
    return (
        proj_y.floor().clamp(0, height - 1).to(torch.int64),
        proj_x.floor().clamp(0, width - 1).to(torch.int64),
    )


def split_batch(batch_inputs_dict: dict[str, Any]) -> tuple[list[Tensor], list[Tensor] | None]:
    """Split the concatenated points and labels of the batch per sample.

    Args:
        batch_inputs_dict: Model inputs holding ``feat``, ``offset`` and optionally ``segment``.

    Returns:
        tuple: Points of every sample, and their labels when the batch carries them.
    """
    offset = batch_inputs_dict["offset"].tolist()
    bounds = list(zip([0] + offset[:-1], offset))
    points = [batch_inputs_dict["feat"][start:end] for start, end in bounds]
    if "segment" not in batch_inputs_dict:
        return points, None
    return points, [batch_inputs_dict["segment"][start:end] for start, end in bounds]


def join_batch(
    points: Sequence[Tensor], labels: Sequence[Tensor] | None
) -> dict[str, Any]:
    """Concatenate the points and labels of every sample back into the batch.

    Args:
        points: Points of every sample.
        labels: Labels of every sample, None when the batch carries none.

    Returns:
        The model inputs the point count of the batch feeds.
    """
    counts = torch.tensor([sample.shape[0] for sample in points], dtype=torch.int64)
    concatenated = torch.cat(list(points), dim=0)
    batch_indices = torch.repeat_interleave(
        torch.arange(len(points), dtype=torch.int32, device=concatenated.device), counts
    )
    outputs: dict[str, Any] = {
        "feat": concatenated,
        "coord": concatenated[:, :3],
        "points": list(points),
        "offset": torch.cumsum(counts, dim=0),
        "batch_indices": batch_indices,
        "sample_count": len(points),
    }
    if labels is not None:
        outputs["segment"] = torch.cat(list(labels), dim=0)
    return outputs


class RangeInterpolation:
    """Fill the empty range image pixels of every sample with horizontal interpolation."""

    def __init__(
        self, height: int, width: int, fov_up: float, fov_down: float, ignore_index: int
    ) -> None:
        """Initialize the RangeInterpolation layer.

        Args:
            height: Range image height in pixels.
            width: Range image width in pixels.
            fov_up: Upper vertical field of view in degrees.
            fov_down: Lower vertical field of view in degrees.
            ignore_index: Label used when the interpolated neighbours disagree.
        """
        self.height = height
        self.width = width
        self.fov_up_rad = torch.deg2rad(torch.tensor(fov_up)).item()
        self.fov_down_rad = torch.deg2rad(torch.tensor(fov_down)).item()
        self.ignore_index = ignore_index

    def __call__(self, batch_inputs_dict: dict[str, Any], *, is_training: bool) -> dict[str, Any]:
        """Append the interpolated points of every sample.

        Args:
            batch_inputs_dict: Model inputs holding the points of the batch.
            is_training: Whether the owning model is in training mode. The interpolation is
                deterministic, so it runs in both modes.

        Returns:
            The model inputs with the interpolated points appended, together with
            ``num_points``, the point count before the interpolation.
        """
        del is_training
        points, labels = split_batch(batch_inputs_dict)
        num_points = [sample.shape[0] for sample in points]

        interpolated_points = []
        interpolated_labels = [] if labels is not None else None
        for index, sample in enumerate(points):
            new_points, new_labels = self.interpolate(
                sample, None if labels is None else labels[index]
            )
            interpolated_points.append(torch.cat([sample, new_points], dim=0))
            if interpolated_labels is not None:
                interpolated_labels.append(torch.cat([labels[index], new_labels], dim=0))

        outputs = join_batch(interpolated_points, interpolated_labels)
        outputs["num_points"] = torch.tensor(num_points, dtype=torch.int64)
        return outputs

    def interpolate(
        self, points: Tensor, labels: Tensor | None
    ) -> tuple[Tensor, Tensor | None]:
        """Build the points the empty range image pixels of one sample interpolate to.

        Args:
            points: Points of one sample.
            labels: Labels of that sample, None when the batch carries none.

        Returns:
            tuple: The interpolated points and their labels.
        """
        proj_y, proj_x = project_range(
            points, self.height, self.width, self.fov_up_rad, self.fov_down_rad
        )
        # The farthest point of a pixel is written first, so the nearest one wins the pixel
        order = torch.argsort(
            torch.linalg.vector_norm(points[:, :3], ord=2, dim=1), descending=True
        )

        proj_image = points.new_full((self.height, self.width, points.shape[1]), -1.0)
        proj_mask = torch.zeros((self.height, self.width), dtype=torch.bool)
        proj_image[proj_y[order], proj_x[order]] = points[order]
        proj_mask[proj_y[order], proj_x[order]] = True

        proj_labels = None
        if labels is not None:
            proj_labels = torch.full(
                (self.height, self.width), self.ignore_index, dtype=torch.int64
            )
            proj_labels[proj_y[order], proj_x[order]] = labels[order]

        # An empty pixel with a filled left and right neighbour takes their midpoint
        inner = proj_mask[:, 1:-1]
        can_interpolate = ~inner & proj_mask[:, :-2] & proj_mask[:, 2:]
        interpolated_rows, inner_columns = torch.where(can_interpolate)
        interpolated_columns = inner_columns + 1

        new_points = 0.5 * (
            proj_image[interpolated_rows, interpolated_columns - 1]
            + proj_image[interpolated_rows, interpolated_columns + 1]
        )
        if proj_labels is None:
            return new_points, None

        left_labels = proj_labels[interpolated_rows, interpolated_columns - 1]
        right_labels = proj_labels[interpolated_rows, interpolated_columns + 1]
        new_labels = torch.where(
            left_labels == right_labels,
            left_labels,
            torch.full_like(left_labels, self.ignore_index),
        )
        return new_points, new_labels


class FrustumMix:
    """Mix every sample of the batch with the next one along frustum aligned stripes."""

    def __init__(
        self,
        height: int,
        width: int,
        fov_up: float,
        fov_down: float,
        num_areas: Sequence[int],
        probability: float = 1.0,
    ) -> None:
        """Initialize the FrustumMix layer.

        Args:
            height: Range image height in pixels.
            width: Range image width in pixels.
            fov_up: Upper vertical field of view in degrees.
            fov_down: Lower vertical field of view in degrees.
            num_areas: Candidate stripe counts sampled per call.
            probability: Probability of mixing a sample.
        """
        self.height = height
        self.width = width
        self.fov_up_rad = torch.deg2rad(torch.tensor(fov_up)).item()
        self.fov_down_rad = torch.deg2rad(torch.tensor(fov_down)).item()
        self.num_areas = list(num_areas)
        self.probability = probability

    def __call__(self, batch_inputs_dict: dict[str, Any], *, is_training: bool) -> dict[str, Any]:
        """Mix the samples of the batch, leaving them untouched outside training.

        Args:
            batch_inputs_dict: Model inputs holding the points and the labels of the batch.
            is_training: Whether the owning model is in training mode.

        Returns:
            The model inputs of the mixed batch.
        """
        points, labels = split_batch(batch_inputs_dict)
        if not is_training or labels is None or len(points) < 2:
            return {}

        mixed_points = []
        mixed_labels = []
        for index, sample in enumerate(points):
            partner = (index + 1) % len(points)
            if torch.rand(()).item() >= self.probability:
                mixed_points.append(sample)
                mixed_labels.append(labels[index])
                continue
            mixer = self.mix_vertical if torch.rand(()).item() < 0.5 else self.mix_horizontal
            sample_points, sample_labels = mixer(
                sample, labels[index], points[partner], labels[partner]
            )
            mixed_points.append(sample_points)
            mixed_labels.append(sample_labels)

        return join_batch(mixed_points, mixed_labels)

    def mix_vertical(
        self, points: Tensor, labels: Tensor, mix_points: Tensor, mix_labels: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Take alternating horizontal stripes of the range image from the two clouds.

        Args:
            points: Points of the sample.
            labels: Labels of the sample.
            mix_points: Points of the second cloud.
            mix_labels: Labels of the second cloud.

        Returns:
            tuple: The mixed points and labels.
        """
        proj_y, _ = project_range(
            points, self.height, self.width, self.fov_up_rad, self.fov_down_rad
        )
        mix_proj_y, _ = project_range(
            mix_points, self.height, self.width, self.fov_up_rad, self.fov_down_rad
        )

        num_areas = int(self.num_areas[torch.randint(len(self.num_areas), ()).item()])
        row_bins = torch.linspace(0, self.height, num_areas + 1).to(torch.int64)
        mixed_points = []
        mixed_labels = []
        for area_index in range(num_areas):
            start_row = row_bins[area_index]
            end_row = row_bins[area_index + 1]
            if area_index % 2 == 0:
                mask = (proj_y >= start_row) & (proj_y < end_row)
                mixed_points.append(points[mask])
                mixed_labels.append(labels[mask])
            else:
                mask = (mix_proj_y >= start_row) & (mix_proj_y < end_row)
                mixed_points.append(mix_points[mask])
                mixed_labels.append(mix_labels[mask])

        return torch.cat(mixed_points, dim=0), torch.cat(mixed_labels, dim=0)

    def mix_horizontal(
        self, points: Tensor, labels: Tensor, mix_points: Tensor, mix_labels: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Replace one half of the range image columns with the second cloud.

        Args:
            points: Points of the sample.
            labels: Labels of the sample.
            mix_points: Points of the second cloud.
            mix_labels: Labels of the second cloud.

        Returns:
            tuple: The mixed points and labels.
        """
        _, proj_x = project_range(
            points, self.height, self.width, self.fov_up_rad, self.fov_down_rad
        )
        _, mix_proj_x = project_range(
            mix_points, self.height, self.width, self.fov_up_rad, self.fov_down_rad
        )

        start_column = int(torch.randint(self.width // 2, ()).item())
        end_column = start_column + self.width // 2
        keep_mask = (proj_x < start_column) | (proj_x >= end_column)
        mix_mask = (mix_proj_x >= start_column) & (mix_proj_x < end_column)

        return (
            torch.cat([points[keep_mask], mix_points[mix_mask]], dim=0),
            torch.cat([labels[keep_mask], mix_labels[mix_mask]], dim=0),
        )


class InstanceCopy:
    """Copy the points of selected classes from the next sample of the batch."""

    def __init__(self, instance_classes: Sequence[int], probability: float = 1.0) -> None:
        """Initialize the InstanceCopy layer.

        Args:
            instance_classes: Semantic class indices copied from the second cloud.
            probability: Probability of copying into a sample.
        """
        self.instance_classes = list(instance_classes)
        self.probability = probability

    def __call__(self, batch_inputs_dict: dict[str, Any], *, is_training: bool) -> dict[str, Any]:
        """Copy the selected classes between the samples of the batch.

        Args:
            batch_inputs_dict: Model inputs holding the points and the labels of the batch.
            is_training: Whether the owning model is in training mode.

        Returns:
            The model inputs of the enriched batch.
        """
        points, labels = split_batch(batch_inputs_dict)
        if not is_training or labels is None or len(points) < 2:
            return {}

        copied_points = []
        copied_labels = []
        for index, sample in enumerate(points):
            partner = (index + 1) % len(points)
            if torch.rand(()).item() >= self.probability:
                copied_points.append(sample)
                copied_labels.append(labels[index])
                continue
            instance_mask = torch.isin(
                labels[partner],
                torch.tensor(self.instance_classes, device=labels[partner].device),
            )
            copied_points.append(torch.cat([sample, points[partner][instance_mask]], dim=0))
            copied_labels.append(torch.cat([labels[index], labels[partner][instance_mask]], dim=0))

        return join_batch(copied_points, copied_labels)
