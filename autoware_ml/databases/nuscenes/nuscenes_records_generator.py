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

import logging
from typing import Any, Mapping, Sequence, Tuple

import numpy as np
from jaxtyping import Float64
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

from autoware_ml.databases.box3d_pipelines.box3d_pipeline import Box3DPipeline
from autoware_ml.databases.scenarios import ScenarioData
from autoware_ml.databases.schemas.box3d_schemas import Box3DDataModel
from autoware_ml.databases.schemas.camera_frames import CameraFrameDataModel
from autoware_ml.databases.schemas.category_mapping import CategoryMappingDataModel
from autoware_ml.databases.schemas.dataset_schemas import DatasetRecord
from autoware_ml.databases.schemas.frame_basic_metadata import FrameBasicMetadata
from autoware_ml.databases.schemas.lidar_frames import LidarFrameDataModel
from autoware_ml.databases.schemas.lidar_sources import LidarSourceDataModel
from autoware_ml.types.sensor import LidarChannel
from autoware_ml.types.spatial import CoordinateSystem
from autoware_ml.utils.dataset import convert_quaternion_to_matrix

logger = logging.getLogger(__name__)

# Canonical nuScenes camera channels in the devkit ordering.
_NUSCENES_CAMERA_CHANNELS = (
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)

_LIDARSEG_TABLE_NAME = "lidarseg"


class NuscenesRecordsGenerator:
    """
    RecordsGenerator for the nuScenes dataset. It converts the samples of the given scenes into
    DatasetRecords. All paths in the produced records are relative to the database root.
    """

    def __init__(
        self,
        database_root_path: str,
        version: str,
        scenario_data: Sequence[ScenarioData],
        ignore_label_index: int,
        box3d_pipelines: Sequence[Box3DPipeline],
    ) -> None:
        """
        Initialize NuscenesRecordsGenerator.

        Args:
          database_root_path: Root path of the nuScenes database.
          version: Version of the nuScenes database, for example v1.0-trainval.
          scenario_data: Scenario data of the scenes to process, one entry per scene.
          ignore_label_index: Label index to use for ignored labels in the box3d annotations.
          box3d_pipelines: List of box3d pipelines to process the box3d annotations.
        """

        self.database_root_path = database_root_path
        self.version = version
        self.scenario_data = scenario_data
        self.ignore_label_index = ignore_label_index
        self.box3d_pipelines = box3d_pipelines
        self.nuscenes_dataset = NuScenes(
            version=version, dataroot=database_root_path, verbose=False
        )
        self._scene_by_name = {scene["name"]: scene for scene in self.nuscenes_dataset.scene}
        self._has_lidarseg = _LIDARSEG_TABLE_NAME in self.nuscenes_dataset.table_names

    def generate_dataset_records(self) -> Sequence[DatasetRecord]:
        """
        Generate dataset records for every scene in the assigned scenario data.

        Returns:
          Sequence[DatasetRecord]: Sequence of dataset records.
        """

        records = []
        for scenario in self.scenario_data:
            records.extend(self._generate_scene_records(scenario))
        return records

    def _generate_scene_records(self, scenario: ScenarioData) -> Sequence[DatasetRecord]:
        """
        Generate dataset records for a single scene.

        Args:
          scenario: Scenario data of the scene.

        Returns:
          Sequence[DatasetRecord]: Sequence of dataset records of the scene.
        """

        if scenario.scenario_id not in self._scene_by_name:
            raise ValueError(
                f"Scene {scenario.scenario_id} does not exist in nuScenes {self.version}."
            )
        scene = self._scene_by_name[scenario.scenario_id]
        log_record = self.nuscenes_dataset.get("log", scene["log_token"])

        sample_tokens = []
        sample_token = scene["first_sample_token"]
        while sample_token:
            sample_tokens.append(sample_token)
            sample_token = self.nuscenes_dataset.get("sample", sample_token)["next"]

        records = []
        for sample_index in range(0, len(sample_tokens), scenario.sample_steps):
            sample = self.nuscenes_dataset.get("sample", sample_tokens[sample_index])
            records.append(
                self._extract_sample_record(
                    scenario=scenario,
                    log_record=log_record,
                    sample=sample,
                    sample_index=sample_index,
                )
            )
        return records

    def _extract_sample_record(
        self,
        scenario: ScenarioData,
        log_record: Mapping[str, Any],
        sample: Mapping[str, Any],
        sample_index: int,
    ) -> DatasetRecord:
        """
        Extract a dataset record from a nuScenes sample.

        Args:
          scenario: Scenario data of the scene the sample belongs to.
          log_record: Log record of the scene.
          sample: nuScenes sample record.
          sample_index: Sample index within the scene.

        Returns:
          DatasetRecord: Dataset record of the sample.
        """

        frame_basic_metadata = FrameBasicMetadata(
            scenario_id=scenario.scenario_id,
            sample_id=sample["token"],
            sample_index=sample_index,
            timestamp_seconds=sample["timestamp"] / 1e6,
            scenario_name=scenario.scenario_id,
            location=log_record["location"],
            vehicle_type=log_record["vehicle"],
        )

        lidar_frame_data_model, boxes_3d_data_model = self._extract_lidar_frame(
            sample, scenario.lidar_pointcloud_num_features
        )
        lidar_sweep_data_models = self._extract_lidar_sweeps(
            lidar_frame_data_model=lidar_frame_data_model,
            max_sweeps=scenario.max_sweeps,
            num_features=scenario.lidar_pointcloud_num_features,
        )
        camera_frame_data_models = self._extract_camera_frames(sample)
        lidar_source_data_models = self._extract_lidar_sources(sample)
        category_mapping_data_model = self._extract_category_mapping()

        return DatasetRecord(
            scenario_id=frame_basic_metadata.scenario_id,
            sample_id=frame_basic_metadata.sample_id,
            sample_index=frame_basic_metadata.sample_index,
            timestamp_seconds=frame_basic_metadata.timestamp_seconds,
            scenario_name=frame_basic_metadata.scenario_name,
            location=frame_basic_metadata.location,
            vehicle_type=frame_basic_metadata.vehicle_type,
            lidar_frames=[lidar_frame_data_model] + list(lidar_sweep_data_models),
            camera_frames=camera_frame_data_models,
            lidar_sources=lidar_source_data_models,
            category_mapping=category_mapping_data_model,
            boxes_3d=boxes_3d_data_model,
        )

    def _extract_lidar_frame(
        self, sample: Mapping[str, Any], num_features: int
    ) -> Tuple[LidarFrameDataModel, Sequence[Box3DDataModel]]:
        """
        Extract the keyframe lidar frame and its box annotations from a nuScenes sample.

        Args:
          sample: nuScenes sample record.
          num_features: Number of float32 features per point.

        Returns:
          Tuple of:
            LidarFrameDataModel: Lidar frame data model of the keyframe.
            Sequence[Box3DDataModel]: Box annotations of the keyframe in the sensor coordinate.
        """

        lidar_sample_data_token = sample["data"][LidarChannel.LIDAR_TOP]
        sd_record = self.nuscenes_dataset.get("sample_data", lidar_sample_data_token)
        cs_record = self.nuscenes_dataset.get(
            "calibrated_sensor", sd_record["calibrated_sensor_token"]
        )
        ego_pose_record = self.nuscenes_dataset.get("ego_pose", sd_record["ego_pose_token"])

        lidar_sensor_to_ego_pose_matrix = convert_quaternion_to_matrix(
            rotation_quaternion=Quaternion(cs_record["rotation"]),
            translation=np.asarray(cs_record["translation"], dtype=np.float64),
        )
        lidar_frame_ego_pose_to_global_matrix = convert_quaternion_to_matrix(
            rotation_quaternion=Quaternion(ego_pose_record["rotation"]),
            translation=np.asarray(ego_pose_record["translation"], dtype=np.float64),
        )

        lidar_frame_data_model = LidarFrameDataModel(
            lidar_frame_id=lidar_sample_data_token,
            lidar_keyframe=sd_record["is_key_frame"],
            lidar_sensor_id=cs_record["token"],
            lidar_sensor_channel_name=LidarChannel.LIDAR_TOP.value,
            lidar_timestamp_seconds=sd_record["timestamp"] / 1e6,
            lidar_pointcloud_path=sd_record["filename"],
            lidar_pointcloud_source_path=None,
            lidar_pointcloud_num_features=num_features,
            lidar_sensor_to_ego_pose_matrix=lidar_sensor_to_ego_pose_matrix,
            lidar_frame_ego_pose_to_global_matrix=lidar_frame_ego_pose_to_global_matrix,
            # Always the identity matrix for the main lidar sensor
            lidar_sensor_to_lidar_sweep_matrix=np.eye(4),
            lidar_pointcloud_semantic_mask_path=self._extract_semantic_mask_path(
                lidar_sample_data_token
            ),
        )
        boxes_3d_data_model = self._extract_boxes_3d_annotations(
            sample=sample,
            lidar_sensor_to_ego_pose_matrix=lidar_sensor_to_ego_pose_matrix,
            lidar_frame_ego_pose_to_global_matrix=lidar_frame_ego_pose_to_global_matrix,
        )
        return lidar_frame_data_model, boxes_3d_data_model

    def _extract_semantic_mask_path(self, lidar_sample_data_token: str) -> str | None:
        """
        Extract the lidarseg semantic mask path for a lidar sample data token.

        Args:
          lidar_sample_data_token: Lidar sample data token.

        Returns:
          str | None: Semantic mask path relative to the database root, or None when the
            database has no lidarseg annotations.
        """

        if not self._has_lidarseg:
            return None
        lidarseg_record = self.nuscenes_dataset.get(_LIDARSEG_TABLE_NAME, lidar_sample_data_token)
        return lidarseg_record["filename"]

    def _extract_boxes_3d_annotations(
        self,
        sample: Mapping[str, Any],
        lidar_sensor_to_ego_pose_matrix: Float64[np.ndarray, "4 4"],
        lidar_frame_ego_pose_to_global_matrix: Float64[np.ndarray, "4 4"],
    ) -> Sequence[Box3DDataModel]:
        """
        Extract box annotations from a nuScenes sample and process them with the pipeline.

        Args:
          sample: nuScenes sample record.
          lidar_sensor_to_ego_pose_matrix: Transformation matrix from the lidar sensor to the
            ego pose of the lidar frame.
          lidar_frame_ego_pose_to_global_matrix: Transformation matrix from the ego pose of the
            lidar frame to the global frame.

        Returns:
          Sequence[Box3DDataModel]: Sequence of Box3DDataModel in the sensor coordinate.
        """

        lidar_sample_data_token = sample["data"][LidarChannel.LIDAR_TOP]
        _, boxes_3d, _ = self.nuscenes_dataset.get_sample_data(lidar_sample_data_token)
        if not len(boxes_3d):
            return []

        # Rotation from the global frame into the lidar sensor frame for velocity vectors
        global_to_lidar_rotation = np.linalg.inv(
            lidar_sensor_to_ego_pose_matrix[:3, :3]
        ) @ np.linalg.inv(lidar_frame_ego_pose_to_global_matrix[:3, :3])

        boxes_3d_data_model = []
        sample_annotation_tokens = sample["anns"]
        for box_index, box3d in enumerate(boxes_3d):
            annotation_record = self.nuscenes_dataset.get(
                "sample_annotation", sample_annotation_tokens[box_index]
            )
            # nuScenes reports the box velocity in the global frame
            global_velocity = self.nuscenes_dataset.box_velocity(
                sample_annotation_tokens[box_index]
            )
            lidar_velocity = global_to_lidar_rotation @ np.asarray(
                global_velocity, dtype=np.float64
            )

            # Convert the box3d to the Box3DFieldIndex format, where the length and width are
            # swapped since in nuScenes, the shape is (width, length, height)
            box3d_params = np.asarray(
                [
                    box3d.center[0],
                    box3d.center[1],
                    box3d.center[2],
                    box3d.wlh[1],
                    box3d.wlh[0],
                    box3d.wlh[2],
                    box3d.orientation.yaw_pitch_roll[0],
                    lidar_velocity[0],
                    lidar_velocity[1],
                    lidar_velocity[2],
                ],
                dtype=np.float64,
            )
            box3d_valid = (
                annotation_record["num_lidar_pts"] + annotation_record["num_radar_pts"]
            ) > 0

            box_3d_attributes = set()
            for attribute_token in annotation_record["attribute_tokens"]:
                attribute_record = self.nuscenes_dataset.get("attribute", attribute_token)
                box_3d_attributes.add(attribute_record["name"])

            boxes_3d_data_model.append(
                Box3DDataModel(
                    box3d_params=box3d_params,
                    box3d_instance_id=annotation_record["instance_token"],
                    box3d_dataset_label_name=box3d.name,
                    box3d_label_name=box3d.name,
                    # Initially, set all label indices to the ignore label index
                    box3d_label_index=self.ignore_label_index,
                    box3d_num_lidar_points=annotation_record["num_lidar_pts"],
                    box3d_num_radar_points=annotation_record["num_radar_pts"],
                    box3d_valid=box3d_valid,
                    box3d_attributes=box_3d_attributes,
                    box3d_coordinate=CoordinateSystem.LIDAR_COMMON.name,
                )
            )

        # Process 3D boxes with the pipeline
        for box3d_pipeline in self.box3d_pipelines:
            boxes_3d_data_model = box3d_pipeline(boxes_3d_data_model)

        return boxes_3d_data_model

    def _extract_lidar_sweeps(
        self,
        lidar_frame_data_model: LidarFrameDataModel,
        max_sweeps: int,
        num_features: int,
    ) -> Sequence[LidarFrameDataModel]:
        """
        Extract multi-sweep lidar frames preceding a nuScenes sample.

        Args:
          lidar_frame_data_model: Lidar frame data model of the keyframe.
          max_sweeps: Max number of lidar sweeps to include.
          num_features: Number of float32 features per point.

        Returns:
          Sequence[LidarFrameDataModel]: Lidar sweep frame data models ordered from nearest
            to oldest.
        """

        lidar_frame_data_models = []
        current_sample_data_record = self.nuscenes_dataset.get(
            "sample_data", lidar_frame_data_model.lidar_frame_id
        )
        keyframe_lidar_to_global_matrix = (
            lidar_frame_data_model.lidar_frame_ego_pose_to_global_matrix
            @ lidar_frame_data_model.lidar_sensor_to_ego_pose_matrix
        )

        for _ in range(max_sweeps):
            # Stop processing if the current lidar sample data has no previous sample data
            if not current_sample_data_record["prev"]:
                break

            current_sample_data_record = self.nuscenes_dataset.get(
                "sample_data", current_sample_data_record["prev"]
            )
            current_cs_record = self.nuscenes_dataset.get(
                "calibrated_sensor",
                current_sample_data_record["calibrated_sensor_token"],
            )
            current_ego_pose_record = self.nuscenes_dataset.get(
                "ego_pose", current_sample_data_record["ego_pose_token"]
            )

            sweep_sensor_to_ego_pose_matrix = convert_quaternion_to_matrix(
                rotation_quaternion=Quaternion(current_cs_record["rotation"]),
                translation=np.asarray(current_cs_record["translation"], dtype=np.float64),
            )
            sweep_ego_pose_to_global_matrix = convert_quaternion_to_matrix(
                rotation_quaternion=Quaternion(current_ego_pose_record["rotation"]),
                translation=np.asarray(current_ego_pose_record["translation"], dtype=np.float64),
            )

            # Sweep -> sweep frame ego pose -> global -> keyframe lidar
            sweep_to_lidar_sensor_matrix = (
                np.linalg.inv(keyframe_lidar_to_global_matrix)
                @ sweep_ego_pose_to_global_matrix
                @ sweep_sensor_to_ego_pose_matrix
            )
            # Inverse it to obtain the transformation matrix from the lidar sensor to the sweep
            lidar_sensor_to_lidar_sweep_matrix = np.linalg.inv(sweep_to_lidar_sensor_matrix)

            lidar_frame_data_models.append(
                LidarFrameDataModel(
                    lidar_frame_id=current_sample_data_record["token"],
                    lidar_keyframe=current_sample_data_record["is_key_frame"],
                    lidar_sensor_id=current_cs_record["token"],
                    lidar_sensor_channel_name=lidar_frame_data_model.lidar_sensor_channel_name,
                    lidar_timestamp_seconds=current_sample_data_record["timestamp"] / 1e6,
                    lidar_pointcloud_path=current_sample_data_record["filename"],
                    lidar_pointcloud_source_path=None,  # Always None for lidar sweeps
                    lidar_pointcloud_num_features=num_features,
                    lidar_sensor_to_ego_pose_matrix=sweep_sensor_to_ego_pose_matrix,
                    lidar_frame_ego_pose_to_global_matrix=sweep_ego_pose_to_global_matrix,
                    lidar_sensor_to_lidar_sweep_matrix=lidar_sensor_to_lidar_sweep_matrix,
                    lidar_pointcloud_semantic_mask_path=None,  # Always None for lidar sweeps
                )
            )
        return lidar_frame_data_models

    def _extract_camera_frames(self, sample: Mapping[str, Any]) -> Sequence[CameraFrameDataModel]:
        """
        Extract camera frame records from a nuScenes sample.

        Args:
          sample: nuScenes sample record.

        Returns:
          Sequence[CameraFrameDataModel]: Camera frame data models of the sample in the
            canonical nuScenes camera ordering.
        """

        camera_frame_data_models = []
        for channel_name in _NUSCENES_CAMERA_CHANNELS:
            if channel_name not in sample["data"]:
                continue
            sample_data_token = sample["data"][channel_name]
            sd_record = self.nuscenes_dataset.get("sample_data", sample_data_token)
            cs_record = self.nuscenes_dataset.get(
                "calibrated_sensor", sd_record["calibrated_sensor_token"]
            )
            ego_pose_record = self.nuscenes_dataset.get("ego_pose", sd_record["ego_pose_token"])

            camera_intrinsic_matrix = np.asarray(cs_record["camera_intrinsic"], dtype=np.float64)
            if camera_intrinsic_matrix.shape != (3, 3):
                raise ValueError(
                    f"Camera intrinsic matrix of channel {channel_name} must be (3, 3), "
                    f"got {camera_intrinsic_matrix.shape}."
                )
            camera_sensor_to_ego_pose_matrix = convert_quaternion_to_matrix(
                rotation_quaternion=Quaternion(cs_record["rotation"]),
                translation=np.asarray(cs_record["translation"], dtype=np.float64),
            )
            camera_frame_ego_pose_to_global_matrix = convert_quaternion_to_matrix(
                rotation_quaternion=Quaternion(ego_pose_record["rotation"]),
                translation=np.asarray(ego_pose_record["translation"], dtype=np.float64),
            )

            camera_frame_data_models.append(
                CameraFrameDataModel(
                    camera_frame_id=sample_data_token,
                    camera_keyframe=sd_record["is_key_frame"],
                    camera_sensor_id=cs_record["token"],
                    camera_sensor_channel_name=channel_name,
                    camera_timestamp_seconds=sd_record["timestamp"] / 1e6,
                    camera_image_path=sd_record["filename"],
                    camera_image_width=sd_record["width"],
                    camera_image_height=sd_record["height"],
                    camera_intrinsic_matrix=camera_intrinsic_matrix,
                    # nuScenes images are pre-undistorted so the distortion fields are empty
                    camera_distortion_coefficients=[],
                    camera_distortion_model="",
                    camera_sensor_to_ego_pose_matrix=camera_sensor_to_ego_pose_matrix,
                    camera_frame_ego_pose_to_global_matrix=camera_frame_ego_pose_to_global_matrix,
                )
            )
        return camera_frame_data_models

    def _extract_lidar_sources(self, sample: Mapping[str, Any]) -> Sequence[LidarSourceDataModel]:
        """
        Extract lidar sources metadata from a nuScenes sample.

        Args:
          sample: nuScenes sample record.

        Returns:
          Sequence[LidarSourceDataModel]: Lidar source data models, one entry for the top lidar.
        """

        lidar_sample_data_token = sample["data"][LidarChannel.LIDAR_TOP]
        sd_record = self.nuscenes_dataset.get("sample_data", lidar_sample_data_token)
        cs_record = self.nuscenes_dataset.get(
            "calibrated_sensor", sd_record["calibrated_sensor_token"]
        )
        return [
            LidarSourceDataModel(
                channel_name=LidarChannel.LIDAR_TOP.value,
                sensor_token=cs_record["sensor_token"],
                translation=np.asarray(cs_record["translation"], dtype=np.float64),
                rotation=Quaternion(cs_record["rotation"]).rotation_matrix,
            )
        ]

    def _extract_category_mapping(self) -> CategoryMappingDataModel:
        """
        Extract the category mapping from the nuScenes lidarseg taxonomy.

        Returns:
          CategoryMappingDataModel: Category mapping of the database. Empty when the database
            has no lidarseg annotations.
        """

        if not self._has_lidarseg:
            return CategoryMappingDataModel(category_names=[], category_indices=[])

        category_names = []
        category_indices = []
        for category_record in self.nuscenes_dataset.category:
            category_names.append(category_record["name"])
            category_indices.append(category_record["index"])
        return CategoryMappingDataModel(
            category_names=category_names,
            category_indices=category_indices,
        )
