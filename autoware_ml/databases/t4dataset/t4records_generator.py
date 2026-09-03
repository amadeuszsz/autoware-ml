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

"""Record extraction of one T4dataset scenario through t4-devkit.

One record is emitted per kept sample. It carries the sample's own lidar frame first and up
to max_sweeps preceding frames, every camera image of the sample, the calibration of every
lidar sensor, the segmentation category table of the scene and the box annotations after
the box pipelines. Every path is relative to the database root.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from jaxtyping import Float64
from t4_devkit import Tier4
from t4_devkit.common.timestamp import microseconds2seconds
from t4_devkit.dataclass.box import Box3D
from t4_devkit.schema import (
    CalibratedSensor,
    EgoPose,
    LidarSeg,
    Sample,
    SampleAnnotation,
    SampleData,
    SchemaName,
    Sensor,
    SensorModality,
)

from autoware_ml.databases.box3d_pipelines.box3d_label_resolver import Box3DLabelResolver
from autoware_ml.databases.scenarios import ScenarioData
from autoware_ml.databases.schemas.box3d_schemas import Box3DDataModel
from autoware_ml.databases.schemas.camera_frames import CameraFrameDataModel
from autoware_ml.databases.schemas.category_mapping import CategoryMappingDataModel
from autoware_ml.databases.schemas.dataset_schemas import DatasetRecord
from autoware_ml.databases.schemas.lidar_frames import LidarFrameDataModel
from autoware_ml.databases.schemas.lidar_sources import LidarSourceDataModel
from autoware_ml.databases.taxonomy import LabelTaxonomy
from autoware_ml.geometry.utils import points_in_boxes_3d
from autoware_ml.types.geometry import Box3DFieldIndex
from autoware_ml.types.spatial import CoordinateSystem
from autoware_ml.utils.dataset import convert_quaternion_to_matrix

logger = logging.getLogger(__name__)

FLOAT32_BYTES = 4
# T4dataset stores radial and tangential coefficients, the model OpenCV calls plumb bob.
CAMERA_DISTORTION_MODEL = "plumb_bob"


class T4RecordsGenerator:
    """Build the dataset records of one T4dataset scenario."""

    def __init__(
        self,
        database_root_path: str,
        scenario_data: ScenarioData,
        lidar_channel: str,
        box3d_label_resolver: Box3DLabelResolver,
        segmentation_taxonomy: LabelTaxonomy,
        recompute_boxes3d_lidar_points_num: bool,
    ) -> None:
        """
        Initialize T4RecordsGenerator.

        Args:
          database_root_path: Root path of the T4 database.
          scenario_data: Scenario to extract, carrying the parameters of its dataset.
          lidar_channel: Sensor channel of the lidar frame every sample is built around.
          box3d_label_resolver: Resolver baking the label of every box through the taxonomy
            and the box pipelines.
          segmentation_taxonomy: Taxonomy the mask categories of a scene with semantic masks
            must be listed in.
          recompute_boxes3d_lidar_points_num: Whether to recount the lidar points inside every
            box from the point cloud, after the pipelines ran. This reads every point cloud
            and slows generation down considerably.
        """

        self.database_root_path = Path(database_root_path)
        self.scenario_data = scenario_data
        self.lidar_channel = lidar_channel
        self.box3d_label_resolver = box3d_label_resolver
        self.segmentation_taxonomy = segmentation_taxonomy
        self.recompute_boxes3d_lidar_points_num = recompute_boxes3d_lidar_points_num
        self.num_features = scenario_data.lidar_pointcloud_num_features

        self.scene_dir = (
            f"{scenario_data.dataset_name}/{scenario_data.scenario_id}/"
            f"{scenario_data.scenario_version}"
        )
        self.scene_root = self.database_root_path / self.scene_dir
        if not self.scene_root.is_dir():
            raise FileNotFoundError(f"Scene directory {self.scene_root} does not exist.")
        self.t4 = Tier4(
            data_root=str(
                self.database_root_path / scenario_data.dataset_name / scenario_data.scenario_id
            ),
            revision=scenario_data.scenario_version,
            verbose=False,
        )

        self.scenario_name = self._extract_scenario_name()
        self.lidarseg_by_sample_data = self._index_lidarseg()
        self.lidar_sources = self._extract_lidar_sources()
        self.category_mapping = self._extract_category_mapping()

    def generate_dataset_records(self) -> Sequence[DatasetRecord]:
        """
        Generate the dataset records of the scenario, in time order.

        Returns:
          Sequence[DatasetRecord]: Sequence of dataset records.
        """

        logger.info(f"Generating dataset records for scenario {self.scene_dir}")
        samples = sorted(self.t4.sample, key=lambda sample: sample.timestamp)
        return [
            self._extract_record(samples[sample_index], sample_index)
            for sample_index in self._sample_indices(samples)
        ]

    def _relative_path(self, filename: str) -> str:
        """Database root relative path of a file stored relative to the scene root."""
        return f"{self.scene_dir}/{filename}"

    def _require_file(self, filename: str) -> int:
        """
        Size in bytes of a file stored relative to the scene root.

        Args:
          filename: Path relative to the scene root.

        Returns:
          int: Size of the file in bytes.
        """

        path = self.scene_root / filename
        if not path.is_file():
            raise FileNotFoundError(f"{self.scene_dir}: missing file {filename}")
        return path.stat().st_size

    def _sample_indices(self, samples: Sequence[Sample]) -> Sequence[int]:
        """
        Positions of the samples the dataset trains on.

        A dataset declaring semantic masks trains on the masked samples only, so the others
        are left out. A scene of such a dataset without a single mask is rejected.

        Args:
          samples: Samples of the scene in time order.

        Returns:
          Sequence[int]: Positions of the kept samples.
        """

        indices = list(range(len(samples)))
        if self.scenario_data.semantic_masks:
            indices = [index for index in indices if self._is_masked(samples[index])]
            if not indices:
                raise ValueError(
                    f"{self.scene_dir}: dataset {self.scenario_data.dataset_name} declares "
                    "semantic masks but the scene carries none."
                )
        return indices[:: self.scenario_data.sample_steps]

    def _lidar_token(self, sample: Sample) -> str:
        """Sample data token of the lidar frame of a sample."""
        if self.lidar_channel not in sample.data:
            raise ValueError(
                f"{self.scene_dir}: sample {sample.token} has no {self.lidar_channel} frame."
            )
        return sample.data[self.lidar_channel]

    def _is_masked(self, sample: Sample) -> bool:
        """Whether the lidar frame of a sample carries a semantic mask."""
        return self._lidar_token(sample) in self.lidarseg_by_sample_data

    def _extract_scenario_name(self) -> str:
        """Name of the single scene of the scenario directory."""
        if len(self.t4.scene) != 1:
            raise ValueError(f"{self.scene_dir}: expected exactly one scene record.")
        return self.t4.scene[0].name

    def _index_lidarseg(self) -> Mapping[str, LidarSeg]:
        """Semantic mask records keyed by the annotated sample data token."""
        records: dict[str, LidarSeg] = {}
        for record in self.t4.lidarseg:
            if record.sample_data_token in records:
                raise ValueError(
                    f"{self.scene_dir}: several lidarseg records for sample data "
                    f"{record.sample_data_token}."
                )
            records[record.sample_data_token] = record
        return records

    def _extract_record(self, sample: Sample, sample_index: int) -> DatasetRecord:
        """
        Extract the dataset record of one sample.

        Args:
          sample: T4 sample.
          sample_index: Position of the sample in the scene.

        Returns:
          DatasetRecord: Dataset record of the sample.
        """

        lidar_token = self._lidar_token(sample)
        sd_record: SampleData = self.t4.get(SchemaName.SAMPLE_DATA, lidar_token)
        if not sd_record.is_valid:
            raise ValueError(f"{self.scene_dir}: lidar sample data {lidar_token} is invalid.")

        lidar_frame = self._extract_lidar_frame(
            sd_record, lidar_sensor_to_lidar_sweep_matrix=np.eye(4), annotated=True
        )
        lidar_sweeps = self._extract_lidar_sweeps(sd_record, lidar_frame)
        boxes_3d = self._extract_boxes_3d(sample, lidar_token, lidar_frame)
        camera_frames = self._extract_camera_frames(sample)

        return DatasetRecord(
            scenario_id=self.scenario_data.scenario_id,
            sample_id=sample.token,
            sample_index=sample_index,
            timestamp_seconds=microseconds2seconds(sample.timestamp),
            location=self.scenario_data.location,
            vehicle_type=self.scenario_data.vehicle_type,
            scenario_name=self.scenario_name,
            lidar_frames=[lidar_frame, *lidar_sweeps],
            camera_frames=camera_frames,
            lidar_sources=self.lidar_sources,
            category_mapping=self.category_mapping,
            boxes_3d=boxes_3d,
        )

    def _sensor_to_global_matrix(self, sd_record: SampleData) -> Float64[np.ndarray, "4 4"]:
        """Transform from the sensor frame of a sample data record to the global frame."""
        cs_record: CalibratedSensor = self.t4.get(
            SchemaName.CALIBRATED_SENSOR, sd_record.calibrated_sensor_token
        )
        ego_pose_record: EgoPose = self.t4.get(SchemaName.EGO_POSE, sd_record.ego_pose_token)
        ego_pose_to_global = convert_quaternion_to_matrix(
            rotation_quaternion=ego_pose_record.rotation, translation=ego_pose_record.translation
        )
        sensor_to_ego_pose = convert_quaternion_to_matrix(
            rotation_quaternion=cs_record.rotation, translation=cs_record.translation
        )
        return ego_pose_to_global @ sensor_to_ego_pose

    def _point_count(self, filename: str) -> int:
        """
        Number of points of a point cloud file stored relative to the scene root.

        Args:
          filename: Path relative to the scene root.

        Returns:
          int: Number of points.
        """

        size = self._require_file(filename)
        record_size = FLOAT32_BYTES * self.num_features
        if size % record_size != 0:
            raise ValueError(
                f"{self.scene_dir}: {filename} holds {size} bytes, not a multiple of "
                f"{self.num_features} float32 features."
            )
        return size // record_size

    def _extract_lidar_frame(
        self,
        sd_record: SampleData,
        lidar_sensor_to_lidar_sweep_matrix: Float64[np.ndarray, "4 4"],
        annotated: bool,
    ) -> LidarFrameDataModel:
        """
        Extract one lidar frame. Sweeps carry no source metadata and no mask, the loader reads
        those only for the sample's own frame.

        Args:
          sd_record: Sample data record of the lidar frame.
          lidar_sensor_to_lidar_sweep_matrix: Transform from the sensor frame of the sample's
            own lidar frame into the sensor frame of this frame.
          annotated: Whether this is the sample's own frame.

        Returns:
          LidarFrameDataModel: Lidar frame data model.
        """

        cs_record: CalibratedSensor = self.t4.get(
            SchemaName.CALIBRATED_SENSOR, sd_record.calibrated_sensor_token
        )
        ego_pose_record: EgoPose = self.t4.get(SchemaName.EGO_POSE, sd_record.ego_pose_token)
        num_points = self._point_count(sd_record.filename)

        return LidarFrameDataModel(
            lidar_frame_id=sd_record.token,
            lidar_keyframe=sd_record.is_key_frame,
            lidar_sensor_id=cs_record.token,
            lidar_sensor_channel_name=sd_record.channel,
            lidar_timestamp_seconds=microseconds2seconds(sd_record.timestamp),
            lidar_pointcloud_path=self._relative_path(sd_record.filename),
            lidar_pointcloud_source_path=self._source_path(sd_record) if annotated else None,
            lidar_pointcloud_num_features=self.num_features,
            lidar_sensor_to_ego_pose_matrix=convert_quaternion_to_matrix(
                rotation_quaternion=cs_record.rotation, translation=cs_record.translation
            ),
            lidar_frame_ego_pose_to_global_matrix=convert_quaternion_to_matrix(
                rotation_quaternion=ego_pose_record.rotation,
                translation=ego_pose_record.translation,
            ),
            lidar_sensor_to_lidar_sweep_matrix=lidar_sensor_to_lidar_sweep_matrix,
            lidar_pointcloud_semantic_mask_path=(
                self._semantic_mask_path(sd_record, num_points) if annotated else None
            ),
        )

    def _source_path(self, sd_record: SampleData) -> str | None:
        """
        Path of the per sensor source metadata of a lidar frame.

        Args:
          sd_record: Sample data record of the lidar frame.

        Returns:
          str | None: Root relative path, or None when the frame declares no source metadata.
        """

        if not sd_record.info_filename:
            return None
        self._require_file(sd_record.info_filename)
        return self._relative_path(sd_record.info_filename)

    def _semantic_mask_path(self, sd_record: SampleData, num_points: int) -> str | None:
        """
        Path of the semantic mask of a lidar frame, checked against the point count.

        Args:
          sd_record: Sample data record of the lidar frame.
          num_points: Number of points of the frame.

        Returns:
          str | None: Root relative path, or None when the frame carries no mask.
        """

        if sd_record.token not in self.lidarseg_by_sample_data:
            return None
        mask = self.lidarseg_by_sample_data[sd_record.token]
        num_labels = self._require_file(mask.filename)
        if num_labels != num_points:
            raise ValueError(
                f"{self.scene_dir}: {mask.filename} holds {num_labels} labels for "
                f"{num_points} points."
            )
        return self._relative_path(mask.filename)

    def _extract_lidar_sweeps(
        self, sd_record: SampleData, lidar_frame: LidarFrameDataModel
    ) -> Sequence[LidarFrameDataModel]:
        """
        Extract the preceding lidar frames of a sample, nearest first, at most max_sweeps of
        them. A frame the dataset flags invalid is left out.

        Args:
          sd_record: Sample data record of the sample's own lidar frame.
          lidar_frame: Lidar frame data model of the sample's own frame.

        Returns:
          Sequence[LidarFrameDataModel]: Lidar sweep frames.
        """

        sample_sensor_to_global = (
            lidar_frame.lidar_frame_ego_pose_to_global_matrix
            @ lidar_frame.lidar_sensor_to_ego_pose_matrix
        )
        sweeps: list[LidarFrameDataModel] = []
        token = sd_record.prev
        while token and len(sweeps) < self.scenario_data.max_sweeps:
            sweep_record: SampleData = self.t4.get(SchemaName.SAMPLE_DATA, token)
            if sweep_record.is_valid:
                # Sample sensor -> global -> sweep sensor
                lidar_sensor_to_lidar_sweep_matrix = (
                    np.linalg.inv(self._sensor_to_global_matrix(sweep_record))
                    @ sample_sensor_to_global
                )
                sweeps.append(
                    self._extract_lidar_frame(
                        sweep_record,
                        lidar_sensor_to_lidar_sweep_matrix=lidar_sensor_to_lidar_sweep_matrix,
                        annotated=False,
                    )
                )
            token = sweep_record.prev
        return sweeps

    def _extract_boxes_3d(
        self, sample: Sample, lidar_token: str, lidar_frame: LidarFrameDataModel
    ) -> Sequence[Box3DDataModel]:
        """
        Extract the box annotations of a sample in the lidar sensor frame and process them
        with the box pipelines.

        Args:
          sample: T4 sample.
          lidar_token: Sample data token of the sample's own lidar frame.
          lidar_frame: Lidar frame data model of the sample's own frame.

        Returns:
          Sequence[Box3DDataModel]: Box annotations after the pipelines.
        """

        _, boxes, _ = self.t4.get_sample_data(
            lidar_token,
            selected_ann_tokens=list(sample.ann_3ds),
            as_3d=True,
            as_sensor_coord=True,
        )
        if len(boxes) != len(sample.ann_3ds):
            raise ValueError(
                f"{self.scene_dir}: sample {sample.token} has {len(sample.ann_3ds)} annotations "
                f"but the devkit returned {len(boxes)} boxes."
            )
        boxes_3d = [
            self._build_box_3d(annotation_token, box)
            for annotation_token, box in zip(sample.ann_3ds, boxes, strict=True)
        ]
        boxes_3d = self.box3d_label_resolver(boxes_3d)
        if self.recompute_boxes3d_lidar_points_num:
            boxes_3d = self._recount_lidar_points(boxes_3d, lidar_frame)
        return boxes_3d

    def _build_box_3d(self, annotation_token: str, box: Box3D) -> Box3DDataModel:
        """
        Build one box data model before its label is resolved by the pipelines.

        Args:
          annotation_token: Sample annotation token of the box.
          box: Box in the lidar sensor frame.

        Returns:
          Box3DDataModel: Box data model.
        """

        annotation: SampleAnnotation = self.t4.get(SchemaName.SAMPLE_ANNOTATION, annotation_token)
        # T4dataset stores the size as (width, length, height)
        width, length, height = box.size
        velocity = (
            np.full(3, np.nan, dtype=np.float64)
            if box.velocity is None
            else np.asarray(box.velocity, dtype=np.float64)
        )
        box3d_params = np.asarray(
            [*box.position, length, width, height, box.rotation.yaw_pitch_roll[0], *velocity],
            dtype=np.float64,
        )
        if box3d_params.shape != (len(Box3DFieldIndex),):
            raise ValueError(
                f"{self.scene_dir}: expected {len(Box3DFieldIndex)} box parameters, "
                f"built {box3d_params.shape}."
            )
        return Box3DDataModel(
            box3d_params=box3d_params,
            box3d_instance_id=annotation.instance_token,
            box3d_dataset_label_name=box.semantic_label.name,
            box3d_label_name=box.semantic_label.name,
            # The label resolver bakes the fine name and the class, every box starts ignored
            box3d_label_index=self.box3d_label_resolver.ignore_index,
            box3d_num_lidar_points=int(annotation.num_lidar_pts),
            box3d_num_radar_points=int(annotation.num_radar_pts),
            box3d_valid=annotation.num_lidar_pts > 0,
            box3d_attributes=set(box.semantic_label.attributes),
            box3d_coordinate=CoordinateSystem.LIDAR_COMMON.name,
        )

    def _recount_lidar_points(
        self, boxes_3d: Sequence[Box3DDataModel], lidar_frame: LidarFrameDataModel
    ) -> Sequence[Box3DDataModel]:
        """
        Recount the lidar points inside every box from the point cloud of the frame.

        Args:
          boxes_3d: Box annotations after the pipelines.
          lidar_frame: Lidar frame data model of the sample's own frame.

        Returns:
          Sequence[Box3DDataModel]: Boxes with the recounted number of lidar points.
        """

        if not len(boxes_3d):
            return []
        points = np.fromfile(
            self.database_root_path / lidar_frame.lidar_pointcloud_path, dtype=np.float32
        ).reshape(-1, self.num_features)[:, :3]
        box_params = np.stack([box.box3d_params for box in boxes_3d]).astype(np.float32)
        points_in_boxes = points_in_boxes_3d(torch.from_numpy(points), torch.from_numpy(box_params))
        counts = points_in_boxes.sum(dim=1).tolist()
        return [
            box.create_new_data_model(box3d_num_lidar_points=int(count))
            for box, count in zip(boxes_3d, counts, strict=True)
        ]

    def _extract_camera_frames(self, sample: Sample) -> Sequence[CameraFrameDataModel]:
        """
        Extract every camera image of a sample. The lidar to camera transform is not stored,
        the reader composes it from the pose matrices of the lidar frame and the camera frame.

        Args:
          sample: T4 sample.

        Returns:
          Sequence[CameraFrameDataModel]: Camera frames of the sample ordered by channel name.
        """

        camera_frames: list[CameraFrameDataModel] = []
        for channel_name, token in sorted(sample.data.items()):
            sd_record: SampleData = self.t4.get(SchemaName.SAMPLE_DATA, token)
            if sd_record.modality != SensorModality.CAMERA:
                continue
            self._require_file(sd_record.filename)
            cs_record: CalibratedSensor = self.t4.get(
                SchemaName.CALIBRATED_SENSOR, sd_record.calibrated_sensor_token
            )
            ego_pose_record: EgoPose = self.t4.get(SchemaName.EGO_POSE, sd_record.ego_pose_token)
            camera_intrinsic_matrix = np.asarray(cs_record.camera_intrinsic, dtype=np.float64)
            if camera_intrinsic_matrix.shape != (3, 3):
                raise ValueError(
                    f"{self.scene_dir}: camera {channel_name} intrinsic matrix must be (3, 3), "
                    f"got {camera_intrinsic_matrix.shape}."
                )
            distortion = np.asarray(cs_record.camera_distortion, dtype=np.float64).tolist()
            camera_frames.append(
                CameraFrameDataModel(
                    camera_frame_id=sd_record.token,
                    camera_keyframe=sd_record.is_key_frame,
                    camera_sensor_id=cs_record.token,
                    camera_sensor_channel_name=channel_name,
                    camera_timestamp_seconds=microseconds2seconds(sd_record.timestamp),
                    camera_image_path=self._relative_path(sd_record.filename),
                    camera_image_width=int(sd_record.width),
                    camera_image_height=int(sd_record.height),
                    camera_intrinsic_matrix=camera_intrinsic_matrix,
                    camera_distortion_coefficients=distortion,
                    camera_distortion_model=CAMERA_DISTORTION_MODEL if distortion else "",
                    camera_sensor_to_ego_pose_matrix=convert_quaternion_to_matrix(
                        rotation_quaternion=cs_record.rotation, translation=cs_record.translation
                    ),
                    camera_frame_ego_pose_to_global_matrix=convert_quaternion_to_matrix(
                        rotation_quaternion=ego_pose_record.rotation,
                        translation=ego_pose_record.translation,
                    ),
                )
            )
        return camera_frames

    def _extract_lidar_sources(self) -> Sequence[LidarSourceDataModel]:
        """
        Extract the calibration of every lidar sensor of the scene, ordered by channel name.

        A scene may store one calibration record per frame instead of one per sensor, so the
        records of a channel are collapsed into one. A channel whose records disagree has no
        single calibration for the scene and is rejected.

        Returns:
          Sequence[LidarSourceDataModel]: Lidar sources of the scene.
        """

        sources: dict[str, LidarSourceDataModel] = {}
        for cs_record in self.t4.calibrated_sensor:
            sensor_record: Sensor = self.t4.get(SchemaName.SENSOR, cs_record.sensor_token)
            if sensor_record.modality != SensorModality.LIDAR:
                continue
            sensor_to_ego_pose = convert_quaternion_to_matrix(
                rotation_quaternion=cs_record.rotation, translation=cs_record.translation
            )
            source = LidarSourceDataModel(
                channel_name=sensor_record.channel,
                sensor_token=sensor_record.token,
                translation=sensor_to_ego_pose[:3, 3],
                rotation=sensor_to_ego_pose[:3, :3],
            )
            if sensor_record.channel not in sources:
                sources[sensor_record.channel] = source
                continue
            known = sources[sensor_record.channel]
            if (
                known.sensor_token != source.sensor_token
                or not np.array_equal(known.rotation, source.rotation)
                or not np.array_equal(known.translation, source.translation)
            ):
                raise ValueError(
                    f"{self.scene_dir}: conflicting calibrations for {sensor_record.channel}."
                )
        return [sources[channel] for channel in sorted(sources)]

    def _extract_category_mapping(self) -> CategoryMappingDataModel:
        """
        Extract the segmentation category table of the scene. A scene with semantic masks
        must name only categories the segmentation vocabulary lists, so a new category is
        discovered when the table is generated.

        Returns:
          CategoryMappingDataModel: Category names and their label indices.
        """

        category_names = [category.name for category in self.t4.category]
        if self.scenario_data.semantic_masks:
            unlisted = self.segmentation_taxonomy.vocabulary.unlisted(category_names)
            if unlisted:
                raise ValueError(
                    f"{self.scene_dir}: the segmentation vocabulary does not list the mask "
                    f"categories {unlisted}."
                )
        return CategoryMappingDataModel(
            category_names=category_names,
            category_indices=[int(category.index) for category in self.t4.category],
        )
