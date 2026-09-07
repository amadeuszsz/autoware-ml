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

from typing import Sequence

import numpy as np
import torch
from jaxtyping import Float32, Float64, Int64
from pydantic import BaseModel, ConfigDict, model_validator
from torch import Tensor

from autoware_ml.datamodule.samples.sample import Sample
from autoware_ml.types.geometry import PointFeatureName


def _uniform_presence(samples: Sequence[Sample], field_name: str) -> bool:
    """
    Check that a task field is either present in every sample or absent in every sample.

    Args:
      samples: Samples of the batch.
      field_name: Name of the task field.

    Returns:
      bool: True when the field is present in every sample.
    """

    presence = [getattr(sample, field_name) is not None for sample in samples]
    if any(presence) and not all(presence):
        raise ValueError(
            f"The task field {field_name} must be present in every sample of a batch or in "
            f"none, got mixed presence."
        )
    return all(presence)


class PointCloudBatch(BaseModel):
    """
    Point clouds of one batch, one packed feature tensor per sample.

    Attributes:
      features: Packed point features of every sample.
      feature_names: Name of every feature column, shared across the batch.
      num_current_points: Number of current frame points of every sample. None when a
        transform reordered the rows and the leading block is not tracked anymore.
    """

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    features: tuple[Float32[Tensor, "num_points num_features"], ...]
    feature_names: tuple[PointFeatureName, ...]
    num_current_points: tuple[int, ...] | None

    @property
    def lengths(self) -> tuple[int, ...]:
        """
        Get the number of points of every sample.

        Returns:
          tuple[int, ...]: Number of points of every sample.
        """

        return tuple(features.shape[0] for features in self.features)

    @property
    def offset(self) -> Int64[Tensor, " batch_size"]:
        """
        Get the inclusive cumulative point counts marking the end of every sample inside the
        concatenated point space.

        Returns:
          Int64[Tensor, " batch_size"]: Inclusive cumulative point counts.
        """

        return torch.cumsum(torch.tensor(self.lengths, dtype=torch.int64), dim=0)

    @property
    def concatenated(self) -> Float32[Tensor, "total_points num_features"]:
        """
        Get the point features of the whole batch concatenated along the point dimension.

        Returns:
          Float32[Tensor, "total_points num_features"]: Concatenated point features.
        """

        return torch.cat(self.features, dim=0)

    def to(self, device: torch.device) -> PointCloudBatch:
        """
        Move the tensors to a device.

        Args:
          device: Target device.

        Returns:
          PointCloudBatch: Batch on the target device.
        """

        return self.model_copy(
            update={"features": tuple(features.to(device) for features in self.features)}
        )

    def pin_memory(self) -> PointCloudBatch:
        """
        Pin the tensors in host memory.

        Returns:
          PointCloudBatch: Batch with pinned tensors.
        """

        return self.model_copy(
            update={"features": tuple(features.pin_memory() for features in self.features)}
        )


class Boxes3DBatch(BaseModel):
    """
    Detection ground truth boxes of one batch, one variable length set per sample.

    Attributes:
      params: Box parameters of every sample.
      labels: Box labels of every sample.
      names: Box label names of every sample.
      num_lidar_points: Number of lidar points inside the boxes of every sample.
    """

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    params: tuple[Float32[Tensor, "num_boxes num_box_params"], ...]
    labels: tuple[Int64[Tensor, " num_boxes"], ...]
    names: tuple[tuple[str, ...], ...]
    num_lidar_points: tuple[Int64[Tensor, " num_boxes"], ...]

    def to(self, device: torch.device) -> Boxes3DBatch:
        """
        Move the tensors to a device.

        Args:
          device: Target device.

        Returns:
          Boxes3DBatch: Batch on the target device.
        """

        return self.model_copy(
            update={
                "params": tuple(params.to(device) for params in self.params),
                "labels": tuple(labels.to(device) for labels in self.labels),
                "num_lidar_points": tuple(counts.to(device) for counts in self.num_lidar_points),
            }
        )

    def pin_memory(self) -> Boxes3DBatch:
        """
        Pin the tensors in host memory.

        Returns:
          Boxes3DBatch: Batch with pinned tensors.
        """

        return self.model_copy(
            update={
                "params": tuple(params.pin_memory() for params in self.params),
                "labels": tuple(labels.pin_memory() for labels in self.labels),
                "num_lidar_points": tuple(counts.pin_memory() for counts in self.num_lidar_points),
            }
        )


class SegmentationBatch(BaseModel):
    """
    Semantic segmentation labels of one batch, aligned with the point cloud batch.

    Attributes:
      labels: Semantic labels of every sample.
    """

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    labels: tuple[Int64[Tensor, " num_points"], ...]

    @property
    def concatenated(self) -> Int64[Tensor, " total_points"]:
        """
        Get the labels of the whole batch concatenated along the point dimension.

        Returns:
          Int64[Tensor, " total_points"]: Concatenated labels.
        """

        return torch.cat(self.labels, dim=0)

    def to(self, device: torch.device) -> SegmentationBatch:
        """
        Move the tensors to a device.

        Args:
          device: Target device.

        Returns:
          SegmentationBatch: Batch on the target device.
        """

        return self.model_copy(
            update={"labels": tuple(labels.to(device) for labels in self.labels)}
        )

    def pin_memory(self) -> SegmentationBatch:
        """
        Pin the tensors in host memory.

        Returns:
          SegmentationBatch: Batch with pinned tensors.
        """

        return self.model_copy(
            update={"labels": tuple(labels.pin_memory() for labels in self.labels)}
        )


class ImageBatch(BaseModel):
    """
    Multiview camera images of one batch, one image set per sample.

    Attributes:
      images: Images of every sample in channel first layout.
      camera_names: Camera channel names of every sample.
      camera_intrinsics: Intrinsic matrices of every sample.
      lidar2cam: Lidar to camera matrices of every sample.
      lidar2img: Lidar to image plane matrices of every sample.
      img_aug_matrix: Image space augmentation matrices of every sample, when augmentation ran.
    """

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    images: tuple[Float32[Tensor, "num_cameras channels height width"], ...]
    camera_names: tuple[tuple[str, ...], ...]
    camera_intrinsics: tuple[Float32[Tensor, "num_cameras 4 4"], ...]
    lidar2cam: tuple[Float32[Tensor, "num_cameras 4 4"], ...]
    lidar2img: tuple[Float32[Tensor, "num_cameras 4 4"], ...]
    img_aug_matrix: tuple[Float32[Tensor, "num_cameras 4 4"], ...] | None

    def to(self, device: torch.device) -> ImageBatch:
        """
        Move the tensors to a device.

        Args:
          device: Target device.

        Returns:
          ImageBatch: Batch on the target device.
        """

        update = {
            "images": tuple(images.to(device) for images in self.images),
            "camera_intrinsics": tuple(m.to(device) for m in self.camera_intrinsics),
            "lidar2cam": tuple(m.to(device) for m in self.lidar2cam),
            "lidar2img": tuple(m.to(device) for m in self.lidar2img),
        }
        if self.img_aug_matrix is not None:
            update["img_aug_matrix"] = tuple(m.to(device) for m in self.img_aug_matrix)
        return self.model_copy(update=update)

    def pin_memory(self) -> ImageBatch:
        """
        Pin the tensors in host memory.

        Returns:
          ImageBatch: Batch with pinned tensors.
        """

        update = {
            "images": tuple(images.pin_memory() for images in self.images),
            "camera_intrinsics": tuple(m.pin_memory() for m in self.camera_intrinsics),
            "lidar2cam": tuple(m.pin_memory() for m in self.lidar2cam),
            "lidar2img": tuple(m.pin_memory() for m in self.lidar2img),
        }
        if self.img_aug_matrix is not None:
            update["img_aug_matrix"] = tuple(m.pin_memory() for m in self.img_aug_matrix)
        return self.model_copy(update=update)


class CalibrationBatch(BaseModel):
    """
    Calibration status task data of one batch.

    Attributes:
      fused_images: Fused camera and lidar images of the batch, stacked along the batch
        dimension.
      statuses: Ground truth calibration status of every sample. None when the pipeline runs
        without ground truth.
    """

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    fused_images: Float32[Tensor, "batch_size height width fused_channels"]
    statuses: Int64[Tensor, " batch_size"] | None

    def to(self, device: torch.device) -> CalibrationBatch:
        """
        Move the tensors to a device.

        Args:
          device: Target device.

        Returns:
          CalibrationBatch: Batch on the target device.
        """

        update = {"fused_images": self.fused_images.to(device)}
        if self.statuses is not None:
            update["statuses"] = self.statuses.to(device)
        return self.model_copy(update=update)

    def pin_memory(self) -> CalibrationBatch:
        """
        Pin the tensors in host memory.

        Returns:
          CalibrationBatch: Batch with pinned tensors.
        """

        update = {"fused_images": self.fused_images.pin_memory()}
        if self.statuses is not None:
            update["statuses"] = self.statuses.pin_memory()
        return self.model_copy(update=update)


class FrameMetaBatch(BaseModel):
    """
    Frame metadata of one batch.

    Attributes:
      sample_ids: Sample ID of every sample.
      scene_tokens: Scene token of every sample. None when the dataset has no scene resources.
      timestamps: Timestamp in seconds of every sample.
      ego2globals: Ego to global matrices of every sample. None when the dataset has no ego
        poses.
      prev_exists: Whether the previous sample exists, for every sample. None when the dataset
        does not track sample continuity.
    """

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    sample_ids: tuple[str, ...]
    scene_tokens: tuple[str, ...] | None
    timestamps: tuple[float, ...]
    ego2globals: tuple[Float64[Tensor, "4 4"], ...] | None
    prev_exists: tuple[bool, ...] | None

    def to(self, device: torch.device) -> FrameMetaBatch:
        """
        Move the tensors to a device.

        Args:
          device: Target device.

        Returns:
          FrameMetaBatch: Batch on the target device.
        """

        if self.ego2globals is None:
            return self
        return self.model_copy(
            update={"ego2globals": tuple(m.to(device) for m in self.ego2globals)}
        )

    def pin_memory(self) -> FrameMetaBatch:
        """
        Pin the tensors in host memory.

        Returns:
          FrameMetaBatch: Batch with pinned tensors.
        """

        if self.ego2globals is None:
            return self
        return self.model_copy(
            update={"ego2globals": tuple(m.pin_memory() for m in self.ego2globals)}
        )


class Batch(BaseModel):
    """
    Typed batch of samples, the single interface between the datamodule and the models. Every
    task field is optional and mirrors the sample structure. The flat properties expose the
    tensors under the parameter names the models bind against.

    Attributes:
      meta: Frame metadata of the batch.
      point_cloud: Point clouds of the batch. None when the batch has no point clouds.
      boxes: Detection ground truth of the batch. None when detection is inactive.
      segmentation: Segmentation ground truth of the batch. None when segmentation is
        inactive.
      images: Multiview camera images of the batch. None when the camera modality is inactive.
      calibration: Calibration task data of the batch. None when the calibration task is
        inactive.
    """

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    meta: FrameMetaBatch
    point_cloud: PointCloudBatch | None = None
    boxes: Boxes3DBatch | None = None
    segmentation: SegmentationBatch | None = None
    images: ImageBatch | None = None
    calibration: CalibrationBatch | None = None

    @model_validator(mode="after")
    def validate_batch(self) -> Batch:
        """
        Validate the alignment between the task batches.

        Returns:
          Batch: The validated batch.
        """

        batch_size = len(self.meta.sample_ids)
        if self.point_cloud is not None and len(self.point_cloud.features) != batch_size:
            raise ValueError(
                f"Point cloud batch covers {len(self.point_cloud.features)} samples but the batch "
                f"holds {batch_size} samples."
            )
        if self.boxes is not None and len(self.boxes.params) != batch_size:
            raise ValueError(
                f"Box batch covers {len(self.boxes.params)} samples but the batch holds "
                f"{batch_size} samples."
            )
        if self.segmentation is not None:
            if self.point_cloud is None:
                raise ValueError("A segmentation batch requires a point cloud batch.")
            if len(self.segmentation.labels) != batch_size:
                raise ValueError(
                    f"Segmentation batch covers {len(self.segmentation.labels)} samples but "
                    f"the batch holds {batch_size} samples."
                )
            for labels, features in zip(
                self.segmentation.labels, self.point_cloud.features, strict=True
            ):
                if labels.shape[0] != features.shape[0]:
                    raise ValueError(
                        f"Segmentation labels cover {labels.shape[0]} points but the point "
                        f"cloud holds {features.shape[0]} points."
                    )
        if self.images is not None and len(self.images.images) != batch_size:
            raise ValueError(
                f"Image batch covers {len(self.images.images)} samples but the batch holds "
                f"{batch_size} samples."
            )
        if self.calibration is not None and self.calibration.fused_images.shape[0] != batch_size:
            raise ValueError(
                f"Calibration batch covers {self.calibration.fused_images.shape[0]} samples "
                f"but the batch holds {batch_size} samples."
            )
        return self

    @property
    def batch_size(self) -> int:
        """
        Get the number of samples in the batch.

        Returns:
          int: Number of samples.
        """

        return len(self.meta.sample_ids)

    @property
    def sample_token(self) -> tuple[str, ...]:
        """
        Get the sample ID of every sample.

        Returns:
          tuple[str, ...]: Sample IDs.
        """

        return self.meta.sample_ids

    @property
    def scene_token(self) -> tuple[str, ...] | None:
        """
        Get the scene token of every sample.

        Returns:
          tuple[str, ...] | None: Scene tokens, or None when unavailable.
        """

        return self.meta.scene_tokens

    @property
    def timestamp(self) -> tuple[float, ...]:
        """
        Get the timestamp in seconds of every sample.

        Returns:
          tuple[float, ...]: Timestamps.
        """

        return self.meta.timestamps

    @property
    def ego2global(self) -> tuple[Float64[Tensor, "4 4"], ...] | None:
        """
        Get the ego to global matrix of every sample.

        Returns:
          tuple[Float64[Tensor, "4 4"], ...] | None: Ego to global matrices, or None when
            unavailable.
        """

        return self.meta.ego2globals

    @property
    def prev_exists(self) -> tuple[bool, ...] | None:
        """
        Get whether the previous sample exists, for every sample.

        Returns:
          tuple[bool, ...] | None: Previous sample flags, or None when untracked.
        """

        return self.meta.prev_exists

    @property
    def points(self) -> tuple[Float32[Tensor, "num_points num_features"], ...] | None:
        """
        Get the packed point features of every sample.

        Returns:
          tuple[Float32[Tensor, "num_points num_features"], ...] | None: Point features per
            sample, or None when the batch has no point clouds.
        """

        if self.point_cloud is None:
            return None
        return self.point_cloud.features

    @property
    def offset(self) -> Int64[Tensor, " batch_size"] | None:
        """
        Get the inclusive cumulative point counts of the batch.

        Returns:
          Int64[Tensor, " batch_size"] | None: Inclusive cumulative point counts, or None when
            the batch has no point clouds.
        """

        if self.point_cloud is None:
            return None
        return self.point_cloud.offset

    @property
    def num_current_points(self) -> tuple[int, ...] | None:
        """
        Get the number of current frame points of every sample.

        Returns:
          tuple[int, ...] | None: Current frame point counts, or None when untracked.
        """

        if self.point_cloud is None:
            return None
        return self.point_cloud.num_current_points

    @property
    def gt_boxes(
        self,
    ) -> tuple[Float32[Tensor, "num_boxes num_box_params"], ...] | None:
        """
        Get the ground truth box parameters of every sample.

        Returns:
          tuple[Float32[Tensor, "num_boxes num_box_params"], ...] | None: Box parameters per
            sample, or None when detection is inactive.
        """

        if self.boxes is None:
            return None
        return self.boxes.params

    @property
    def gt_labels(self) -> tuple[Int64[Tensor, " num_boxes"], ...] | None:
        """
        Get the ground truth box labels of every sample.

        Returns:
          tuple[Int64[Tensor, " num_boxes"], ...] | None: Box labels per sample, or None when
            detection is inactive.
        """

        if self.boxes is None:
            return None
        return self.boxes.labels

    @property
    def gt_num_points(self) -> tuple[Int64[Tensor, " num_boxes"], ...] | None:
        """
        Get the lidar point counts of the ground truth boxes of every sample.

        Returns:
          tuple[Int64[Tensor, " num_boxes"], ...] | None: Box lidar point counts per sample,
            or None when detection is inactive.
        """

        if self.boxes is None:
            return None
        return self.boxes.num_lidar_points

    @property
    def segment(self) -> Int64[Tensor, " total_points"] | None:
        """
        Get the segmentation labels of the whole batch concatenated along the point dimension.

        Returns:
          Int64[Tensor, " total_points"] | None: Concatenated labels, or None when
            segmentation is inactive.
        """

        if self.segmentation is None:
            return None
        return self.segmentation.concatenated

    @property
    def img(
        self,
    ) -> tuple[Float32[Tensor, "num_cameras channels height width"], ...] | None:
        """
        Get the multiview images of every sample.

        Returns:
          tuple[Float32[Tensor, "num_cameras channels height width"], ...] | None: Images per
            sample, or None when the camera modality is inactive.
        """

        if self.images is None:
            return None
        return self.images.images

    @property
    def camera_intrinsics(
        self,
    ) -> tuple[Float32[Tensor, "num_cameras 4 4"], ...] | None:
        """
        Get the camera intrinsic matrices of every sample.

        Returns:
          tuple[Float32[Tensor, "num_cameras 4 4"], ...] | None: Intrinsic matrices per
            sample, or None when the camera modality is inactive.
        """

        if self.images is None:
            return None
        return self.images.camera_intrinsics

    @property
    def lidar2cam(self) -> tuple[Float32[Tensor, "num_cameras 4 4"], ...] | None:
        """
        Get the lidar to camera matrices of every sample.

        Returns:
          tuple[Float32[Tensor, "num_cameras 4 4"], ...] | None: Lidar to camera matrices per
            sample, or None when the camera modality is inactive.
        """

        if self.images is None:
            return None
        return self.images.lidar2cam

    @property
    def lidar2img(self) -> tuple[Float32[Tensor, "num_cameras 4 4"], ...] | None:
        """
        Get the lidar to image plane matrices of every sample.

        Returns:
          tuple[Float32[Tensor, "num_cameras 4 4"], ...] | None: Lidar to image matrices per
            sample, or None when the camera modality is inactive.
        """

        if self.images is None:
            return None
        return self.images.lidar2img

    @property
    def img_aug_matrix(self) -> tuple[Float32[Tensor, "num_cameras 4 4"], ...] | None:
        """
        Get the image space augmentation matrices of every sample.

        Returns:
          tuple[Float32[Tensor, "num_cameras 4 4"], ...] | None: Augmentation matrices per
            sample, or None when no image space augmentation ran.
        """

        if self.images is None:
            return None
        return self.images.img_aug_matrix

    @property
    def fused_img(
        self,
    ) -> Float32[Tensor, "batch_size height width fused_channels"] | None:
        """
        Get the fused camera and lidar images of the batch.

        Returns:
          Float32[Tensor, "batch_size height width fused_channels"] | None: Fused images, or
            None when the calibration task is inactive.
        """

        if self.calibration is None:
            return None
        return self.calibration.fused_images

    @property
    def gt_calibration_status(self) -> Int64[Tensor, " batch_size"] | None:
        """
        Get the ground truth calibration status of the batch.

        Returns:
          Int64[Tensor, " batch_size"] | None: Calibration statuses, or None when unavailable.
        """

        if self.calibration is None:
            return None
        return self.calibration.statuses

    def to(self, device: torch.device) -> Batch:
        """
        Move the batch tensors to a device.

        Args:
          device: Target device.

        Returns:
          Batch: Batch on the target device.
        """

        update = {"meta": self.meta.to(device)}
        for field_name in (
            "point_cloud",
            "boxes",
            "segmentation",
            "images",
            "calibration",
        ):
            field_value = getattr(self, field_name)
            if field_value is not None:
                update[field_name] = field_value.to(device)
        return self.model_copy(update=update)

    def pin_memory(self) -> Batch:
        """
        Pin the batch tensors in host memory.

        Returns:
          Batch: Batch with pinned tensors.
        """

        update = {"meta": self.meta.pin_memory()}
        for field_name in (
            "point_cloud",
            "boxes",
            "segmentation",
            "images",
            "calibration",
        ):
            field_value = getattr(self, field_name)
            if field_value is not None:
                update[field_name] = field_value.pin_memory()
        return self.model_copy(update=update)

    @classmethod
    def collate(cls, samples: Sequence[Sample]) -> Batch:
        """
        Collate samples into a batch. Every task field must be present in every sample of the
        batch or in none.

        Args:
          samples: Samples produced by the transform pipelines.

        Returns:
          Batch: Collated batch.
        """

        if not len(samples):
            raise ValueError("Cannot collate an empty sequence of samples.")

        meta = cls._collate_meta(samples)
        update: dict = {}
        if _uniform_presence(samples, "points"):
            update["point_cloud"] = cls._collate_points(samples)
        if _uniform_presence(samples, "boxes"):
            update["boxes"] = cls._collate_boxes(samples)
        if _uniform_presence(samples, "segment"):
            update["segmentation"] = SegmentationBatch(
                labels=tuple(torch.from_numpy(sample.segment.labels) for sample in samples)
            )
        if _uniform_presence(samples, "images"):
            update["images"] = cls._collate_images(samples)
        if _uniform_presence(samples, "calibration"):
            update["calibration"] = cls._collate_calibration(samples)
        return cls(meta=meta, **update)

    @staticmethod
    def _collate_meta(samples: Sequence[Sample]) -> FrameMetaBatch:
        """
        Collate the frame metadata of the samples.

        Args:
          samples: Samples of the batch.

        Returns:
          FrameMetaBatch: Collated frame metadata.
        """

        scene_presence = [sample.meta.scene_token is not None for sample in samples]
        if any(scene_presence) and not all(scene_presence):
            raise ValueError("Scene tokens must be present in every sample of a batch or none.")
        ego_presence = [sample.meta.ego2global is not None for sample in samples]
        if any(ego_presence) and not all(ego_presence):
            raise ValueError("Ego poses must be present in every sample of a batch or none.")
        prev_presence = [sample.meta.prev_exists is not None for sample in samples]
        if any(prev_presence) and not all(prev_presence):
            raise ValueError(
                "Previous sample flags must be present in every sample of a batch or none."
            )

        return FrameMetaBatch(
            sample_ids=tuple(sample.meta.sample_id for sample in samples),
            scene_tokens=tuple(sample.meta.scene_token for sample in samples)
            if all(scene_presence)
            else None,
            timestamps=tuple(sample.meta.timestamp_seconds for sample in samples),
            ego2globals=tuple(torch.from_numpy(sample.meta.ego2global) for sample in samples)
            if all(ego_presence)
            else None,
            prev_exists=tuple(sample.meta.prev_exists for sample in samples)
            if all(prev_presence)
            else None,
        )

    @staticmethod
    def _collate_points(samples: Sequence[Sample]) -> PointCloudBatch:
        """
        Collate the point clouds of the samples.

        Args:
          samples: Samples of the batch.

        Returns:
          PointCloudBatch: Collated point clouds.
        """

        feature_names = samples[0].points.feature_names
        for sample in samples:
            if sample.points.feature_names != feature_names:
                raise ValueError(
                    f"Point feature names must match across the batch, got "
                    f"{sample.points.feature_names} and {feature_names}."
                )
        current_counts = tuple(sample.points.num_current_points for sample in samples)
        num_current_points = current_counts if None not in current_counts else None
        return PointCloudBatch(
            features=tuple(torch.from_numpy(sample.points.features) for sample in samples),
            feature_names=feature_names,
            num_current_points=num_current_points,
        )

    @staticmethod
    def _collate_boxes(samples: Sequence[Sample]) -> Boxes3DBatch:
        """
        Collate the ground truth boxes of the samples.

        Args:
          samples: Samples of the batch.

        Returns:
          Boxes3DBatch: Collated boxes.
        """

        return Boxes3DBatch(
            params=tuple(torch.from_numpy(sample.boxes.params) for sample in samples),
            labels=tuple(torch.from_numpy(sample.boxes.labels) for sample in samples),
            names=tuple(sample.boxes.names for sample in samples),
            num_lidar_points=tuple(
                torch.from_numpy(sample.boxes.num_lidar_points) for sample in samples
            ),
        )

    @staticmethod
    def _collate_images(samples: Sequence[Sample]) -> ImageBatch:
        """
        Collate the multiview images of the samples.

        Args:
          samples: Samples of the batch.

        Returns:
          ImageBatch: Collated images.
        """

        aug_presence = [sample.images.img_aug_matrix is not None for sample in samples]
        if any(aug_presence) and not all(aug_presence):
            raise ValueError(
                "Image augmentation matrices must be present in every sample of a batch or none."
            )
        return ImageBatch(
            images=tuple(torch.from_numpy(sample.images.images) for sample in samples),
            camera_names=tuple(sample.images.camera_names for sample in samples),
            camera_intrinsics=tuple(
                torch.from_numpy(sample.images.camera_intrinsics) for sample in samples
            ),
            lidar2cam=tuple(torch.from_numpy(sample.images.lidar2cam) for sample in samples),
            lidar2img=tuple(torch.from_numpy(sample.images.lidar2img) for sample in samples),
            img_aug_matrix=tuple(
                torch.from_numpy(sample.images.img_aug_matrix) for sample in samples
            )
            if all(aug_presence)
            else None,
        )

    @staticmethod
    def _collate_calibration(samples: Sequence[Sample]) -> CalibrationBatch:
        """
        Collate the calibration task data of the samples.

        Args:
          samples: Samples of the batch.

        Returns:
          CalibrationBatch: Collated calibration data.
        """

        for sample in samples:
            if sample.calibration.fused_image is None:
                raise ValueError("Every calibration sample of a batch must carry a fused image.")
        status_presence = [sample.calibration.status is not None for sample in samples]
        if any(status_presence) and not all(status_presence):
            raise ValueError(
                "Calibration statuses must be present in every sample of a batch or none."
            )
        statuses = None
        if all(status_presence):
            statuses = torch.tensor(
                [sample.calibration.status.value for sample in samples],
                dtype=torch.int64,
            )
        return CalibrationBatch(
            fused_images=torch.from_numpy(
                np.stack([sample.calibration.fused_image for sample in samples], axis=0)
            ),
            statuses=statuses,
        )
