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

"""Factories building synthetic records and samples for unit tests."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from jaxtyping import Float64

from autoware_ml.databases.schemas.box3d_schemas import Box3DDataModel
from autoware_ml.databases.schemas.camera_frames import CameraFrameDataModel
from autoware_ml.databases.schemas.category_mapping import CategoryMappingDataModel
from autoware_ml.databases.schemas.dataset_schemas import DatasetRecord
from autoware_ml.databases.schemas.lidar_frames import LidarFrameDataModel
from autoware_ml.databases.taxonomy import (
    DatabaseTaxonomy,
    DetectionTaxonomy,
    LabelVocabulary,
    SegmentationTaxonomy,
)
from autoware_ml.datamodule.samples.boxes3d import Boxes3D
from autoware_ml.datamodule.samples.meta import FrameMeta
from autoware_ml.datamodule.samples.point_cloud import PointCloud
from autoware_ml.datamodule.samples.sample import Sample
from autoware_ml.datamodule.samples.segmentation3d import SegmentationLabels
from autoware_ml.types.geometry import PointFeatureName
from autoware_ml.types.spatial import CoordinateSystem


def make_lidar_frame(
    *,
    frame_id: str = "frame-0",
    keyframe: bool = True,
    timestamp_seconds: float = 100.0,
    pointcloud_path: str = "db/scene/0/data/lidar/0.bin",
    num_features: int = 5,
    sensor_to_ego: Float64[np.ndarray, "4 4"] | None = None,
    ego_to_global: Float64[np.ndarray, "4 4"] | None = None,
    sensor_to_sweep: Float64[np.ndarray, "4 4"] | None = None,
    semantic_mask_path: str | None = None,
) -> LidarFrameDataModel:
    """
    Build a lidar frame data model with identity poses by default.

    Args:
      frame_id: Lidar frame ID.
      keyframe: Whether the frame is a keyframe.
      timestamp_seconds: Timestamp in seconds.
      pointcloud_path: Point cloud path relative to the data root.
      num_features: Number of stored point features.
      sensor_to_ego: Sensor to ego pose matrix, identity when omitted.
      ego_to_global: Ego pose to global matrix, identity when omitted.
      sensor_to_sweep: Main sensor to sweep matrix, identity when omitted.
      semantic_mask_path: Semantic mask path relative to the data root, or None.

    Returns:
      LidarFrameDataModel: The lidar frame data model.
    """

    return LidarFrameDataModel(
        lidar_frame_id=frame_id,
        lidar_keyframe=keyframe,
        lidar_sensor_id="sensor-0",
        lidar_sensor_channel_name="LIDAR_CONCAT",
        lidar_timestamp_seconds=timestamp_seconds,
        lidar_pointcloud_path=pointcloud_path,
        lidar_pointcloud_source_path=None,
        lidar_pointcloud_num_features=num_features,
        lidar_sensor_to_ego_pose_matrix=sensor_to_ego if sensor_to_ego is not None else np.eye(4),
        lidar_frame_ego_pose_to_global_matrix=ego_to_global
        if ego_to_global is not None
        else np.eye(4),
        lidar_sensor_to_lidar_sweep_matrix=sensor_to_sweep
        if sensor_to_sweep is not None
        else np.eye(4),
        lidar_pointcloud_semantic_mask_path=semantic_mask_path,
    )


def make_camera_frame(
    *,
    channel_name: str = "CAM_FRONT",
    timestamp_seconds: float = 100.0,
    image_path: str = "db/scene/0/data/cam/0.jpg",
    width: int = 1920,
    height: int = 1080,
    intrinsic: Float64[np.ndarray, "3 3"] | None = None,
    camera_to_ego: Float64[np.ndarray, "4 4"] | None = None,
    ego_to_global: Float64[np.ndarray, "4 4"] | None = None,
    distortion_coefficients: Sequence[float] = (),
) -> CameraFrameDataModel:
    """
    Build a camera frame data model with identity poses by default.

    Args:
      channel_name: Camera channel name.
      timestamp_seconds: Timestamp in seconds.
      image_path: Image path relative to the data root.
      width: Image width in pixels.
      height: Image height in pixels.
      intrinsic: Camera intrinsic matrix, a focal 1000 pinhole when omitted.
      camera_to_ego: Camera to ego pose matrix, identity when omitted.
      ego_to_global: Ego pose to global matrix, identity when omitted.
      distortion_coefficients: Camera distortion coefficients.

    Returns:
      CameraFrameDataModel: The camera frame data model.
    """

    if intrinsic is None:
        intrinsic = np.array([[1000.0, 0.0, width / 2], [0.0, 1000.0, height / 2], [0.0, 0.0, 1.0]])
    return CameraFrameDataModel(
        camera_frame_id=f"{channel_name}-frame",
        camera_keyframe=True,
        camera_sensor_id=f"{channel_name}-sensor",
        camera_sensor_channel_name=channel_name,
        camera_timestamp_seconds=timestamp_seconds,
        camera_image_path=image_path,
        camera_image_width=width,
        camera_image_height=height,
        camera_intrinsic_matrix=intrinsic,
        camera_distortion_coefficients=list(distortion_coefficients),
        camera_distortion_model="",
        camera_sensor_to_ego_pose_matrix=camera_to_ego if camera_to_ego is not None else np.eye(4),
        camera_frame_ego_pose_to_global_matrix=ego_to_global
        if ego_to_global is not None
        else np.eye(4),
    )


def make_box3d_data_model(
    *,
    params: Sequence[float] = (1.0, 2.0, 0.5, 4.0, 2.0, 1.5, 0.0, 0.0, 0.0, 0.0),
    label_name: str = "car",
    label_index: int = 0,
    num_lidar_points: int = 10,
    valid: bool = True,
    instance_id: str = "instance-0",
    attributes: Sequence[str] = (),
) -> Box3DDataModel:
    """
    Build a box data model whose label is already resolved, a trained car by default.

    Args:
      params: Box parameters following Box3DFieldIndex.
      label_name: Resolved label name of the box.
      label_index: Resolved label index of the box, the ignore index for an untrained class.
      num_lidar_points: Number of lidar points inside the box.
      valid: Whether the box is valid.
      instance_id: Instance ID of the box.
      attributes: Attribute names of the box.

    Returns:
      Box3DDataModel: The box data model.
    """

    return Box3DDataModel(
        box3d_params=np.asarray(params, dtype=np.float64),
        box3d_instance_id=instance_id,
        box3d_dataset_label_name=label_name,
        box3d_label_name=label_name,
        box3d_label_index=label_index,
        box3d_num_lidar_points=num_lidar_points,
        box3d_num_radar_points=0,
        box3d_valid=valid,
        box3d_attributes=set(attributes),
        box3d_coordinate=CoordinateSystem.LIDAR_COMMON.name,
    )


def make_record(
    *,
    lidar_frames: Sequence[LidarFrameDataModel] | None = None,
    camera_frames: Sequence[CameraFrameDataModel] | None = None,
    boxes_3d: Sequence[Box3DDataModel] | None = None,
    category_names: Sequence[str] = (),
    category_indices: Sequence[int] = (),
    scenario_id: str = "scene-0",
    sample_id: str = "sample-0",
    timestamp_seconds: float = 100.0,
) -> DatasetRecord:
    """
    Build a dataset record with one keyframe lidar frame by default.

    Args:
      lidar_frames: Lidar frames of the record, one identity keyframe when omitted.
      camera_frames: Camera frames of the record.
      boxes_3d: Box annotations of the record.
      category_names: Category names of the category mapping.
      category_indices: Category indices of the category mapping.
      scenario_id: Scenario ID of the record.
      sample_id: Sample ID of the record.
      timestamp_seconds: Timestamp in seconds.

    Returns:
      DatasetRecord: The dataset record.
    """

    if lidar_frames is None:
        lidar_frames = [make_lidar_frame(timestamp_seconds=timestamp_seconds)]
    category_mapping = None
    if len(category_names):
        category_mapping = CategoryMappingDataModel(
            category_names=list(category_names), category_indices=list(category_indices)
        )
    return DatasetRecord(
        scenario_id=scenario_id,
        sample_id=sample_id,
        sample_index=0,
        timestamp_seconds=timestamp_seconds,
        location=None,
        vehicle_type=None,
        scenario_name=scenario_id,
        lidar_frames=list(lidar_frames),
        camera_frames=list(camera_frames) if camera_frames is not None else None,
        lidar_sources=None,
        category_mapping=category_mapping,
        boxes_3d=list(boxes_3d) if boxes_3d is not None else None,
    )


def make_point_cloud(
    *,
    num_points: int = 100,
    with_time_lag: bool = True,
    num_current_points: int | None = None,
    seed: int = 0,
) -> PointCloud:
    """
    Build a random point cloud.

    Args:
      num_points: Number of points.
      with_time_lag: Whether the cloud carries the timestamp_difference feature.
      num_current_points: Number of current frame points, every point when omitted.
      seed: Seed of the random generator.

    Returns:
      PointCloud: The point cloud.
    """

    rng = np.random.default_rng(seed)
    feature_names = [
        PointFeatureName.X,
        PointFeatureName.Y,
        PointFeatureName.Z,
        PointFeatureName.INTENSITY,
    ]
    if with_time_lag:
        feature_names.append(PointFeatureName.TIMESTAMP_DIFFERENCE)
    features = rng.uniform(-50.0, 50.0, size=(num_points, len(feature_names)))
    features = features.astype(np.float32)
    features[:, 3] = rng.uniform(0.0, 255.0, size=num_points)
    if with_time_lag:
        features[:, 4] = 0.0
    return PointCloud(
        features=features,
        feature_names=tuple(feature_names),
        num_current_points=num_current_points if num_current_points is not None else num_points,
    )


def make_boxes3d(*, num_boxes: int = 4, seed: int = 0) -> Boxes3D:
    """
    Build random ground truth boxes.

    Args:
      num_boxes: Number of boxes.
      seed: Seed of the random generator.

    Returns:
      Boxes3D: The boxes.
    """

    rng = np.random.default_rng(seed)
    params = np.zeros((num_boxes, 9), dtype=np.float32)
    params[:, :3] = rng.uniform(-40.0, 40.0, size=(num_boxes, 3)).astype(np.float32)
    params[:, 3:6] = rng.uniform(0.5, 5.0, size=(num_boxes, 3)).astype(np.float32)
    params[:, 6] = rng.uniform(-np.pi, np.pi, size=num_boxes).astype(np.float32)
    params[:, 7:9] = rng.uniform(-2.0, 2.0, size=(num_boxes, 2)).astype(np.float32)
    return Boxes3D(
        params=params,
        labels=rng.integers(0, 3, size=num_boxes).astype(np.int64),
        names=tuple(["car"] * num_boxes),
        num_lidar_points=rng.integers(1, 50, size=num_boxes).astype(np.int64),
    )


def make_sample(
    *,
    record: DatasetRecord | None = None,
    data_root: str = "/data",
    points: PointCloud | None = None,
    boxes: Boxes3D | None = None,
    with_segment: bool = False,
    scene_token: str | None = "db/scene/0",
    seed: int = 0,
) -> Sample:
    """
    Build a sample around a synthetic record.

    Args:
      record: Dataset record of the sample, a default record when omitted.
      data_root: Root directory the record paths resolve against.
      points: Point cloud of the sample.
      boxes: Ground truth boxes of the sample.
      with_segment: Whether to attach random segmentation labels aligned with points.
      scene_token: Scene token of the sample.
      seed: Seed of the random generator.

    Returns:
      Sample: The sample.
    """

    if record is None:
        record = make_record()
    segment = None
    if with_segment:
        if points is None:
            raise ValueError("Segmentation labels require a point cloud.")
        rng = np.random.default_rng(seed)
        segment = SegmentationLabels(labels=rng.integers(0, 5, size=len(points)).astype(np.int64))
    meta = FrameMeta(
        sample_id=record.sample_id,
        scene_token=scene_token,
        timestamp_seconds=record.timestamp_seconds,
        ego2global=np.eye(4),
    )
    return Sample(
        record=record,
        data_root=data_root,
        meta=meta,
        points=points,
        boxes=boxes,
        segment=segment,
    )


def _taxonomy_parts(
    class_names: Sequence[str],
    name_mapping: Mapping[str, str] | None,
    coarsening: Mapping[str, str | None] | None,
) -> tuple[LabelVocabulary, Mapping[str, str | None], Mapping[str, Sequence[str]]]:
    """
    Build the vocabulary, the coarsening and one behaviour group per class of a taxonomy, the
    identity over the class names unless a vocabulary or a coarsening is given.
    """

    vocabulary = LabelVocabulary(
        name_mapping if name_mapping is not None else {name: name for name in class_names}
    )
    if coarsening is None:
        coarsening = {name: name for name in vocabulary.fine_names}
    class_groups = {f"grouped_{name}": [name] for name in class_names}
    return vocabulary, coarsening, class_groups


def make_label_taxonomy(
    class_names: Sequence[str] = ("car",),
    *,
    name_mapping: Mapping[str, str] | None = None,
    coarsening: Mapping[str, str | None] | None = None,
    ignore_index: int = -1,
) -> SegmentationTaxonomy:
    """
    Build a segmentation taxonomy, the identity over the class names unless a vocabulary or a
    coarsening is given, with one behaviour group per class.

    Args:
      class_names: Classes of the level, in index order.
      name_mapping: Raw label name to fine label name, the identity over the class names by
        default.
      coarsening: Fine label name to class name, the identity over the fine names by default.
      ignore_index: Label index of a label outside the level.

    Returns:
      SegmentationTaxonomy: The taxonomy.
    """

    vocabulary, coarsening, class_groups = _taxonomy_parts(class_names, name_mapping, coarsening)
    return SegmentationTaxonomy(
        vocabulary=vocabulary,
        class_names=list(class_names),
        coarsening=coarsening,
        ignore_index=ignore_index,
        class_groups=class_groups,
    )


def make_detection_taxonomy(
    class_names: Sequence[str] = ("car",),
    *,
    name_mapping: Mapping[str, str] | None = None,
    coarsening: Mapping[str, str | None] | None = None,
    ignore_index: int = -1,
) -> DetectionTaxonomy:
    """
    Build a detection taxonomy, the identity over the class names unless a vocabulary or a
    coarsening is given, with one behaviour group per class, a 100 m evaluation range and the
    static collision kind for every class.

    Args:
      class_names: Classes of the level, in index order.
      name_mapping: Raw label name to fine label name, the identity over the class names by
        default.
      coarsening: Fine label name to class name, the identity over the fine names by default.
      ignore_index: Label index of a label outside the level.

    Returns:
      DetectionTaxonomy: The taxonomy.
    """

    vocabulary, coarsening, class_groups = _taxonomy_parts(class_names, name_mapping, coarsening)
    return DetectionTaxonomy(
        vocabulary=vocabulary,
        class_names=list(class_names),
        coarsening=coarsening,
        ignore_index=ignore_index,
        class_groups=class_groups,
        eval_range={name: 100.0 for name in class_names},
        collision_kinds={name: "static" for name in class_names},
        vru_speeds={},
        partial_detection_classes=list(class_names),
        heatmap_pooling_classes=[],
    )


def make_database_taxonomy(class_names: Sequence[str] = ("car",)) -> DatabaseTaxonomy:
    """
    Build a database taxonomy whose detection and segmentation levels are the identity over
    the class names.

    Args:
      class_names: Classes of both levels, in index order.

    Returns:
      DatabaseTaxonomy: The database taxonomy.
    """

    return DatabaseTaxonomy(
        detection3d=make_detection_taxonomy(class_names),
        segmentation3d=make_label_taxonomy(class_names),
    )
