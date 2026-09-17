import logging
from types import MappingProxyType
from typing import Sequence

from jaxtyping import Float32
import polars as pl
import numpy as np
import torch

from autoware_ml.databases.schemas.image_frames import ImageFrameDataModel
from autoware_ml.databases.schemas.lidar_frames import LidarFrameDataModel, LidarFrameDatasetSchema
from autoware_ml.databases.schemas.dataset_schemas import DatasetTableSchema
from autoware_ml.dataclasses.batch.frame_meta import FrameMetaSample
from autoware_ml.dataclasses.batch.sample_batch import ModelGTSample
from autoware_ml.dataclasses.geometry.images import ImageSample
from autoware_ml.dataclasses.geometry.point_clouds import LiDARPointCloudSample
from autoware_ml.datamodule.base_dataset import (
    BaseDataset,
)
from autoware_ml.datamodule.base_dataset_task import BaseDatasetTask
from autoware_ml.datamodule.t4dataset.frame_meta import scene_dir_fragment
from autoware_ml.transforms.base import TransformsCompose
from autoware_ml.types.tasks import TaskType


logger = logging.getLogger(__name__)


class T4Dataset(BaseDataset):
    """
    A dataset class that supports multiple tasks.
    It extends MultiTaskDatasetInterface to include implementation of data retrieval for multiple
    tasks in a single interface.
    """

    def __init__(
        self,
        database_root_path: str,
        max_num_3d_gt_bboxes: int,
        dataset_records_dataframe: pl.DataFrame | None,
        transforms: TransformsCompose | None,
        dataset_tasks: MappingProxyType[TaskType | str, BaseDatasetTask],
    ) -> None:
        """
        Initialize the T4Dataset class.
        Args:
          max_num_3d_gt_bboxes: Maximum number of 3D ground truth bounding boxes in the dataset.
            This is allowed to be 0 if the dataset does not contain any 3D ground truth
            bounding boxes or it does not need to run 3D detection tasks.
          dataset_records_dataframe: Polars DataFrame of dataset records to be used in
            the multi-task dataset.
          transforms: Global transforms to be applied to the dataset records.
          dataset_tasks: Every task dataset that is part of the multi-task dataset, mapped by
            task type.
        """
        super().__init__(
            database_root_path=database_root_path,
            max_num_3d_gt_bboxes=max_num_3d_gt_bboxes,
            dataset_records_dataframe=dataset_records_dataframe,
            transforms=transforms,
        )

        # Convert the dataset_tasks to TaskType: BaseDatasetTask mapping if the keys are strings
        self.dataset_tasks: MappingProxyType[TaskType, BaseDatasetTask] = MappingProxyType(
            {
                TaskType(key) if isinstance(key, str) else key: value
                for key, value in dataset_tasks.items()
            }
        )
        logger.info(
            f"Initialized T4Dataset with {len(self.dataset_tasks)} "
            f"task datasets: {list(self.dataset_tasks.keys())} "
            f"transforms: {self.transforms} and max_num_3d_gt_bboxes: {self.max_num_3d_gt_bboxes}"
        )

    def get_data_sample(self, index: int) -> ModelGTSample:
        """
        Process the dataset records dataframe for multiple tasks in the T4 dataset.

        Args:
          index: Index of the specific record to be processed.

        Returns:
          ModelGTSample: Processed multi-task data row, mapped by task type.
        """
        data_samples = {}
        for task_type, dataset_task in self.dataset_tasks.items():
            data_samples[task_type] = dataset_task.get_data_sample(index)

        # Retrieve general data row for the given index from the dataset records dataframe
        lidar_pointcloud_samples = self.get_lidar_pointcloud_data_samples(index)

        # Retrieve the detection3d_gt_bboxes_3d and segmentation3d_gt_sample from the data_samples dictionary
        detection3d_gt_sample: ModelGTSample | None = data_samples.get(
            TaskType.DETECTION3D, None
        )
        if detection3d_gt_sample is not None:
            detection3d_gt_bboxes_3d = detection3d_gt_sample.detection3d_gt_bboxes_3d
        else:
            detection3d_gt_bboxes_3d = None

        segmentation3d_multi_task_gt_sample: ModelGTSample | None = data_samples.get(
            TaskType.SEGMENTATION3D, None
        )
        if segmentation3d_multi_task_gt_sample is not None:
            segmentation3d_gt_sample = segmentation3d_multi_task_gt_sample.segmentation3d_gt_sample
        else:
            segmentation3d_gt_sample = None

        # Merge the data samples from different tasks into a single multi-task data row
        return ModelGTSample(
            lidar_point_cloud_samples=lidar_pointcloud_samples,
            image_samples=self.get_image_data_samples(index),
            point_cloud_data=None,  # point cloud data will be populated in the transform pipeline
            camera_image_data=None,
            detection3d_gt_bboxes_3d=detection3d_gt_bboxes_3d,
            segmentation3d_gt_sample=segmentation3d_gt_sample,
            frame_meta=self.get_frame_meta_sample(index),
        )

    def get_frame_meta_sample(self, idx: int) -> FrameMetaSample:
        """
        Retrieve the evaluation metadata of the frame, the map pose and the scene identifier
        the region and collision filters resolve the lanelet map with.

        Args:
          idx: Index of the specific record to be processed.

        Returns:
          FrameMetaSample: Metadata of the keyframe lidar frame.
        """
        lidar_frame = LidarFrameDataModel.load_from_dictionary(
            self.dataset_records_dataframe.item(idx, DatasetTableSchema.LIDAR_FRAMES.name)[0]
        )
        return FrameMetaSample(
            # The boxes and the points live in the lidar sensor frame, so the map transform
            # composes the sensor mounting with the ego pose of the frame
            ego2global=torch.tensor(
                lidar_frame.lidar_frame_ego_pose_to_global_matrix_fp32
                @ lidar_frame.lidar_sensor_to_ego_pose_matrix_fp32,
                dtype=torch.float32,
            ),
            scene_token=scene_dir_fragment(
                lidar_frame.lidar_pointcloud_relative_path, str(self.database_root_path)
            ),
        )

    def get_image_data_samples(self, idx: int) -> Sequence[ImageSample] | None:
        """
        Retrieve the keyframe image of every camera of the record.

        Args:
          idx: Index of the specific record to be processed.

        Returns:
          Sequence[ImageSample] | None: One sample per camera, None when the record carries no
            image frames.
        """
        image_frames = self.dataset_records_dataframe.item(
            idx, DatasetTableSchema.IMAGE_FRAMES.name
        )
        if not image_frames:
            return None

        image_samples = []
        for camera_frames in image_frames:
            # The keyframe of a camera comes first, its sweeps after it
            image_frame = ImageFrameDataModel.load_from_dictionary(camera_frames[0])
            if image_frame.lidar2cam_fp32 is None or image_frame.lidar2img_fp32 is None:
                raise ValueError(
                    f"The image frame {image_frame.image_frame_id} of record {idx} carries no "
                    "lidar to camera calibration, so it cannot be projected."
                )
            image_samples.append(
                ImageSample(
                    image_path=self._update_frame_path(image_frame.image_path),
                    camera_name=image_frame.image_sensor_channel_name,
                    timestamp=image_frame.image_timestamp_seconds,
                    camera_intrinsic=torch.tensor(image_frame.cam2img_fp32, dtype=torch.float32),
                    lidar2cam=torch.tensor(image_frame.lidar2cam_fp32, dtype=torch.float32),
                    lidar2image=torch.tensor(image_frame.lidar2img_fp32, dtype=torch.float32),
                    distortion_model=image_frame.image_distortion_model,
                    distortion_coefficients=torch.tensor(
                        image_frame.image_distortion_coefficients, dtype=torch.float32
                    ),
                )
            )
        return image_samples

    def _update_frame_path(self, frame_path: str) -> str:
        """
        Remove the absolute path prefix from a frame path and return the updated path with the
        dataset root. Note that this is dataset-specific.
        """
        # Return the relative path from the frame path
        relative_path = "/".join(frame_path.split("/")[-6:])
        return str(self.database_root_path / relative_path)

    def get_lidar_pointcloud_data_samples(self, idx: int) -> Sequence[LiDARPointCloudSample]:
        """
        Retrieve the lidar point cloud data row for the given index.

        Args:
          idx: Index of the specific record to be processed.
        """
        # Retrieve the lidar point cloud data row for the given index from the dataset records dataframe
        lidar_pointcloud_metadata_samples = self.dataset_records_dataframe.item(
            idx, DatasetTableSchema.LIDAR_FRAMES.name
        )
        lidar_pointcloud_samples = []
        for lidar_pointcloud_metadata in lidar_pointcloud_metadata_samples:
            lidar_sensor_to_ego_pose_matrix: Float32[np.ndarray, "4 4"] = lidar_pointcloud_metadata[
                LidarFrameDatasetSchema.lidar_sensor_to_ego_pose_matrix.name
            ]
            lidar_to_ego_pose_to_global_matrix: Float32[np.ndarray, "4 4"] = (
                lidar_pointcloud_metadata[
                    LidarFrameDatasetSchema.lidar_frame_ego_pose_to_global_matrix.name
                ]
            )
            lidar_sensor_to_lidar_sweep_matrix: Float32[np.ndarray, "4 4"] = (
                lidar_pointcloud_metadata[
                    LidarFrameDatasetSchema.lidar_sensor_to_lidar_sweep_matrix.name
                ]
            )
            lidar_pointcloud_path = self._update_frame_path(
                lidar_pointcloud_metadata[LidarFrameDatasetSchema.lidar_pointcloud_path.name]
            )

            lidar_pointcloud_samples.append(
                LiDARPointCloudSample(
                    point_cloud_path=lidar_pointcloud_path,
                    timestamp=lidar_pointcloud_metadata[
                        LidarFrameDatasetSchema.lidar_timestamp_seconds.name
                    ],
                    sensor_to_ego_pose_matrix=torch.tensor(
                        lidar_sensor_to_ego_pose_matrix,
                        dtype=torch.float32,
                    ),
                    lidar_to_ego_pose_to_global_matrix=torch.tensor(
                        lidar_to_ego_pose_to_global_matrix,
                        dtype=torch.float32,
                    ),
                    lidar_sensor_to_lidar_sweep_matrix=torch.tensor(
                        lidar_sensor_to_lidar_sweep_matrix,
                        dtype=torch.float32,
                    ),
                )
            )
        return lidar_pointcloud_samples

    def assign_dataset_records(self, dataset_records_dataframe: pl.DataFrame) -> None:
        """
        Recursively assign the dataset records dataframe to each task dataset as well and
        perform their pre_filtering .

        Args:
            dataset_records_dataframe: Polars DataFrame of dataset records.
        """
        self.dataset_records_dataframe = dataset_records_dataframe
        for dataset_task in self.dataset_tasks.values():
            filtered_dataset_records_dataframe = dataset_task.pre_filter_dataset_records(
                dataset_records_dataframe
            )
            dataset_task.dataset_records_dataframe = filtered_dataset_records_dataframe
