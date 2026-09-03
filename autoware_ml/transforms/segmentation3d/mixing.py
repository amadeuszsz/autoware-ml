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

"""Sample mixing transforms for point cloud segmentation.

The mixing transforms combine the current sample with a secondary sample drawn through the
pipeline context. They operate on single frame clouds, a densified cloud would lose its
leading current frame block under mixing, and they rebuild the point cloud and the
segmentation labels together so both stay aligned by construction.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from jaxtyping import Float32, Int64

from autoware_ml.datamodule.samples.point_cloud import PointCloud
from autoware_ml.datamodule.samples.sample import Sample
from autoware_ml.transforms.base import BaseTransform, TransformsCompose
from autoware_ml.transforms.segmentation3d.utils import project_range
from autoware_ml.types.geometry import PointFeatureName


def _require_single_frame(point_cloud: PointCloud, transform_name: str) -> None:
    """Reject a point cloud whose rows are not all current frame points.

    Args:
        point_cloud: Point cloud validated before mixing.
        transform_name: Name of the mixing transform for the error message.
    """
    if point_cloud.has_feature(PointFeatureName.TIMESTAMP_DIFFERENCE):
        raise ValueError(
            f"{transform_name} mixes single frame clouds only, the point cloud carries the "
            "timestamp_difference feature and its current frame block would not survive "
            "mixing."
        )


def _draw_secondary(context: Any, pre_transform: Any, primary: PointCloud, name: str) -> Sample:
    """Draw the secondary sample and validate it against the primary point cloud.

    Args:
        context: Pipeline context of the current transform call.
        pre_transform: Optional transform applied to the secondary sample.
        primary: Point cloud of the current sample.
        name: Name of the mixing transform for the error messages.

    Returns:
        Sample: Secondary sample with points and segmentation labels.
    """
    if context is None:
        raise RuntimeError(f"{name} requires pipeline context to sample secondary inputs")
    mix_sample = context.sample_secondary(pre_transform=pre_transform)
    if mix_sample.points is None or mix_sample.segment is None:
        raise ValueError(
            f"{name} requires the secondary sample to carry points and segmentation labels."
        )
    if mix_sample.points.feature_names != primary.feature_names:
        raise ValueError(
            f"{name} requires matching point feature layouts, got "
            f"{mix_sample.points.feature_names} for the secondary sample and "
            f"{primary.feature_names} for the current sample."
        )
    return mix_sample


def _rebuild_mixed_sample(
    sample: Sample,
    features: Float32[np.ndarray, "num_points num_features"],
    labels: Int64[np.ndarray, " num_points"],
) -> Sample:
    """Build the mixed sample from the combined features and labels.

    Every row of a mixed single frame cloud is a current frame point, so the current frame
    block covers the whole cloud.

    Args:
        sample: Sample the mixed data replaces the task fields of.
        features: Combined point features.
        labels: Combined segmentation labels, aligned with the features.

    Returns:
        Sample: Sample with the mixed point cloud and labels.
    """
    point_cloud = sample.points.model_copy(
        update={
            "features": features,
            "num_current_points": features.shape[0],
        }
    )
    segment = sample.segment.model_copy(update={"labels": labels})
    return sample.model_copy(update={"points": point_cloud, "segment": segment})


class FrustumMix(BaseTransform):
    """Mix two point clouds along frustum aligned stripes."""

    _required_fields = ["points", "segment"]

    def __init__(
        self,
        *,
        p: float = 1.0,
        height: int,
        width: int,
        fov_up: float,
        fov_down: float,
        num_areas: list[int],
        pre_transform: TransformsCompose | None = None,
    ) -> None:
        """Initialize the FrustumMix transform.

        Args:
            p: Probability of applying the transform.
            height: Range image height in pixels.
            width: Range image width in pixels.
            fov_up: Upper vertical field of view in degrees.
            fov_down: Lower vertical field of view in degrees.
            num_areas: Candidate stripe counts sampled per call.
            pre_transform: Optional transform applied to the secondary sample.
        """
        self.height = height
        self.width = width
        self.fov_up_rad = np.deg2rad(fov_up)
        self.fov_down_rad = np.deg2rad(fov_down)
        self.num_areas = num_areas
        self.pre_transform = pre_transform
        self.p = p

    def transform(self, sample: Sample) -> Sample:
        """Mix the sample with a secondary sample along frustum stripes.

        Draws a secondary sample and randomly applies either a vertical or a horizontal
        frustum mix, replacing the point cloud and the segmentation labels with the mixed
        result.

        Args:
            sample: Sample with a loaded point cloud and segmentation labels.

        Returns:
            Sample with the mixed point cloud and labels.
        """
        _require_single_frame(sample.points, "FrustumMix")
        mix_sample = _draw_secondary(self.context, self.pre_transform, sample.points, "FrustumMix")
        features = sample.points.features
        labels = sample.segment.labels
        mix_features = mix_sample.points.features
        mix_labels = mix_sample.segment.labels

        if np.random.rand() < 0.5:
            mixed_features, mixed_labels = self._mix_vertical(
                features, labels, mix_features, mix_labels
            )
        else:
            mixed_features, mixed_labels = self._mix_horizontal(
                features, labels, mix_features, mix_labels
            )
        return _rebuild_mixed_sample(sample, mixed_features, mixed_labels)

    def _mix_vertical(
        self,
        features: Float32[np.ndarray, "num_points num_features"],
        labels: Int64[np.ndarray, " num_points"],
        mix_features: Float32[np.ndarray, "num_mix_points num_features"],
        mix_labels: Int64[np.ndarray, " num_mix_points"],
    ) -> tuple[
        Float32[np.ndarray, "num_mixed_points num_features"],
        Int64[np.ndarray, " num_mixed_points"],
    ]:
        """Alternate horizontal stripes of the range view between the two samples.

        Args:
            features: Point features of the current sample.
            labels: Segmentation labels of the current sample.
            mix_features: Point features of the secondary sample.
            mix_labels: Segmentation labels of the secondary sample.

        Returns:
            tuple: Mixed point features and segmentation labels.
        """
        proj_y, _ = project_range(
            features[:, :3], self.height, self.width, self.fov_up_rad, self.fov_down_rad
        )
        mix_proj_y, _ = project_range(
            mix_features[:, :3], self.height, self.width, self.fov_up_rad, self.fov_down_rad
        )

        num_areas = int(np.random.choice(self.num_areas))
        row_bins = np.linspace(0, self.height, num_areas + 1, dtype=np.int64)
        mixed_features = []
        mixed_labels = []

        for area_index in range(num_areas):
            start_row = row_bins[area_index]
            end_row = row_bins[area_index + 1]
            if area_index % 2 == 0:
                mask = (proj_y >= start_row) & (proj_y < end_row)
                mixed_features.append(features[mask])
                mixed_labels.append(labels[mask])
            else:
                mask = (mix_proj_y >= start_row) & (mix_proj_y < end_row)
                mixed_features.append(mix_features[mask])
                mixed_labels.append(mix_labels[mask])

        return np.concatenate(mixed_features, axis=0), np.concatenate(mixed_labels, axis=0)

    def _mix_horizontal(
        self,
        features: Float32[np.ndarray, "num_points num_features"],
        labels: Int64[np.ndarray, " num_points"],
        mix_features: Float32[np.ndarray, "num_mix_points num_features"],
        mix_labels: Int64[np.ndarray, " num_mix_points"],
    ) -> tuple[
        Float32[np.ndarray, "num_mixed_points num_features"],
        Int64[np.ndarray, " num_mixed_points"],
    ]:
        """Swap one vertical stripe of the range view with the secondary sample.

        Args:
            features: Point features of the current sample.
            labels: Segmentation labels of the current sample.
            mix_features: Point features of the secondary sample.
            mix_labels: Segmentation labels of the secondary sample.

        Returns:
            tuple: Mixed point features and segmentation labels.
        """
        _, proj_x = project_range(
            features[:, :3], self.height, self.width, self.fov_up_rad, self.fov_down_rad
        )
        _, mix_proj_x = project_range(
            mix_features[:, :3], self.height, self.width, self.fov_up_rad, self.fov_down_rad
        )

        start_col = np.random.randint(0, self.width // 2)
        end_col = start_col + self.width // 2
        keep_mask = (proj_x < start_col) | (proj_x >= end_col)
        mix_mask = (mix_proj_x >= start_col) & (mix_proj_x < end_col)

        out_features = np.concatenate([features[keep_mask], mix_features[mix_mask]], axis=0)
        out_labels = np.concatenate([labels[keep_mask], mix_labels[mix_mask]], axis=0)
        return out_features, out_labels


class InstanceCopy(BaseTransform):
    """Copy selected semantic instances from a secondary point cloud."""

    _required_fields = ["points", "segment"]

    def __init__(
        self,
        *,
        p: float = 1.0,
        instance_classes: list[int],
        pre_transform: TransformsCompose | None = None,
    ) -> None:
        """Initialize the InstanceCopy transform.

        Args:
            p: Probability of applying the transform.
            instance_classes: Semantic class ids copied from the secondary sample.
            pre_transform: Optional transform applied to the secondary sample.
        """
        self.p = p
        self.instance_classes = instance_classes
        self.pre_transform = pre_transform

    def transform(self, sample: Sample) -> Sample:
        """Append selected class instances from a secondary sample.

        Args:
            sample: Sample with a loaded point cloud and segmentation labels.

        Returns:
            Sample with the copied instance points and labels appended.
        """
        _require_single_frame(sample.points, "InstanceCopy")
        mix_sample = _draw_secondary(
            self.context, self.pre_transform, sample.points, "InstanceCopy"
        )
        mix_features = mix_sample.points.features
        mix_labels = mix_sample.segment.labels

        feature_parts = [sample.points.features]
        label_parts = [sample.segment.labels]
        for class_id in self.instance_classes:
            class_mask = mix_labels == class_id
            feature_parts.append(mix_features[class_mask])
            label_parts.append(mix_labels[class_mask])

        return _rebuild_mixed_sample(
            sample,
            np.concatenate(feature_parts, axis=0),
            np.concatenate(label_parts, axis=0),
        )
