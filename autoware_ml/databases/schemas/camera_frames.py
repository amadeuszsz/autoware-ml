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

from dataclasses import dataclass
from typing import Mapping, Any, Sequence

import numpy as np
import polars as pl
from jaxtyping import Float32, Float64
from pydantic import BaseModel, ConfigDict

from autoware_ml.databases.schemas.base_schemas import (
    BaseFieldSchema,
    DatasetTableColumn,
    DataModelInterface,
)


@dataclass(frozen=True)
class CameraFrameDatasetSchema(BaseFieldSchema):
    """
    Dataclass to define polars schema for columns related to cameras.
    """

    camera_frame_id = DatasetTableColumn("camera_frame_id", pl.String)
    camera_keyframe = DatasetTableColumn("camera_keyframe", pl.Boolean)
    camera_sensor_id = DatasetTableColumn("camera_sensor_id", pl.String)
    camera_sensor_channel_name = DatasetTableColumn("camera_sensor_channel_name", pl.String)
    camera_timestamp_seconds = DatasetTableColumn("camera_timestamp_seconds", pl.Float64)
    camera_image_path = DatasetTableColumn("camera_image_path", pl.String)
    camera_image_width = DatasetTableColumn("camera_image_width", pl.Int32)
    camera_image_height = DatasetTableColumn("camera_image_height", pl.Int32)
    camera_intrinsic_matrix = DatasetTableColumn(
        "camera_intrinsic_matrix", pl.Array(pl.Float32, shape=(3, 3))
    )
    camera_distortion_coefficients = DatasetTableColumn(
        "camera_distortion_coefficients", pl.List(pl.Float64)
    )
    camera_distortion_model = DatasetTableColumn("camera_distortion_model", pl.String)
    camera_sensor_to_ego_pose_matrix = DatasetTableColumn(
        "camera_sensor_to_ego_pose_matrix", pl.Array(pl.Float32, shape=(4, 4))
    )
    camera_frame_ego_pose_to_global_matrix = DatasetTableColumn(
        "camera_frame_ego_pose_to_global_matrix", pl.Array(pl.Float32, shape=(4, 4))
    )


class CameraFrameDataModel(BaseModel, DataModelInterface):
    """
    Camera frame data model that can be shared by multiple datasets. It saves the metadata of a
    camera frame captured at the same sample as the lidar keyframe.

    Attributes:
      camera_frame_id: Camera frame ID.
      camera_keyframe: Whether this camera frame is a keyframe.
      camera_sensor_id: Camera sensor ID.
      camera_sensor_channel_name: Camera channel name.
      camera_timestamp_seconds: Camera timestamp in seconds.
      camera_image_path: Camera image path.
      camera_image_width: Camera image width in pixels.
      camera_image_height: Camera image height in pixels.
      camera_intrinsic_matrix: Camera intrinsic matrix.
      camera_distortion_coefficients: Camera distortion coefficients. The length depends on the
        distortion model.
      camera_distortion_model: Camera distortion model name.
      camera_sensor_to_ego_pose_matrix: Transformation matrix from the camera sensor to the ego pose of
        this camera frame.
      camera_frame_ego_pose_to_global_matrix: Transformation matrix from the ego pose of this
        camera frame to the global frame.
    """

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    camera_frame_id: str
    camera_keyframe: bool
    camera_sensor_id: str
    camera_sensor_channel_name: str
    camera_timestamp_seconds: float
    camera_image_path: str
    camera_image_width: int
    camera_image_height: int
    camera_intrinsic_matrix: Float64[np.ndarray, "3 3"]
    camera_distortion_coefficients: Sequence[float]
    camera_distortion_model: str
    camera_sensor_to_ego_pose_matrix: Float64[np.ndarray, "4 4"]
    camera_frame_ego_pose_to_global_matrix: Float64[np.ndarray, "4 4"]

    @property
    def camera_image_relative_path(self) -> str:
        """
        Parse the camera image path to {database_version}/{scene_id}/
        {dataset_version}/data/{camera_token}/{frame}.jpg from path.

        Returns:
          str: Camera image relative path.
        """

        return "/".join(self.camera_image_path.split("/")[-6:])

    @property
    def camera_intrinsic_matrix_fp32(self) -> Float32[np.ndarray, "3 3"]:
        """
        Convert the camera intrinsic matrix to float32.

        Returns:
          Float32[np.ndarray, "3 3"]: Camera intrinsic matrix.
        """

        return self.camera_intrinsic_matrix.astype(np.float32)

    @property
    def camera_sensor_to_ego_pose_matrix_fp32(self) -> Float32[np.ndarray, "4 4"]:
        """
        Convert the camera to ego pose matrix to float32.

        Returns:
          Float32[np.ndarray, "4 4"]: Camera to ego pose matrix.
        """

        return self.camera_sensor_to_ego_pose_matrix.astype(np.float32)

    @property
    def camera_frame_ego_pose_to_global_matrix_fp32(self) -> Float32[np.ndarray, "4 4"]:
        """
        Convert the camera frame ego pose to global matrix to float32.

        Returns:
          Float32[np.ndarray, "4 4"]: Camera frame ego pose to global matrix.
        """

        return self.camera_frame_ego_pose_to_global_matrix.astype(np.float32)

    def to_dictionary(self) -> Mapping[str, Any]:
        """
        Convert the camera frame data model to a dictionary.

        Returns:
          Mapping[str, Any]: Dictionary representation of the camera frame data model.
        """

        return {
            CameraFrameDatasetSchema.camera_frame_id.name: self.camera_frame_id,
            CameraFrameDatasetSchema.camera_keyframe.name: self.camera_keyframe,
            CameraFrameDatasetSchema.camera_sensor_id.name: self.camera_sensor_id,
            CameraFrameDatasetSchema.camera_sensor_channel_name.name: self.camera_sensor_channel_name,
            CameraFrameDatasetSchema.camera_timestamp_seconds.name: self.camera_timestamp_seconds,
            CameraFrameDatasetSchema.camera_image_path.name: self.camera_image_path,
            CameraFrameDatasetSchema.camera_image_width.name: self.camera_image_width,
            CameraFrameDatasetSchema.camera_image_height.name: self.camera_image_height,
            CameraFrameDatasetSchema.camera_intrinsic_matrix.name: self.camera_intrinsic_matrix_fp32,
            CameraFrameDatasetSchema.camera_distortion_coefficients.name: list(
                self.camera_distortion_coefficients
            ),
            CameraFrameDatasetSchema.camera_distortion_model.name: self.camera_distortion_model,
            CameraFrameDatasetSchema.camera_sensor_to_ego_pose_matrix.name: self.camera_sensor_to_ego_pose_matrix_fp32,
            CameraFrameDatasetSchema.camera_frame_ego_pose_to_global_matrix.name: self.camera_frame_ego_pose_to_global_matrix_fp32,
        }

    @classmethod
    def load_from_dictionary(cls, data_model: Mapping[str, Any]) -> CameraFrameDataModel:
        """
        Load the camera frame data model and decode it to the corresponding CameraFrameDataModel
        from a dictionary, which is deserialized from a Polars dataframe.

        Args:
          data_model: Dictionary representation of the camera frame data model, which is
          deserialized from a Polars dataframe.

        Returns:
          CameraFrameDataModel: CameraFrameDataModel object.
        """

        return cls(
            camera_frame_id=data_model[CameraFrameDatasetSchema.camera_frame_id.name],
            camera_keyframe=data_model[CameraFrameDatasetSchema.camera_keyframe.name],
            camera_sensor_id=data_model[CameraFrameDatasetSchema.camera_sensor_id.name],
            camera_sensor_channel_name=data_model[
                CameraFrameDatasetSchema.camera_sensor_channel_name.name
            ],
            camera_timestamp_seconds=data_model[
                CameraFrameDatasetSchema.camera_timestamp_seconds.name
            ],
            camera_image_path=data_model[CameraFrameDatasetSchema.camera_image_path.name],
            camera_image_width=data_model[CameraFrameDatasetSchema.camera_image_width.name],
            camera_image_height=data_model[CameraFrameDatasetSchema.camera_image_height.name],
            camera_intrinsic_matrix=np.asarray(
                data_model[CameraFrameDatasetSchema.camera_intrinsic_matrix.name],
                dtype=np.float64,
            ),
            camera_distortion_coefficients=list(
                data_model[CameraFrameDatasetSchema.camera_distortion_coefficients.name]
            ),
            camera_distortion_model=data_model[
                CameraFrameDatasetSchema.camera_distortion_model.name
            ],
            camera_sensor_to_ego_pose_matrix=np.asarray(
                data_model[CameraFrameDatasetSchema.camera_sensor_to_ego_pose_matrix.name],
                dtype=np.float64,
            ),
            camera_frame_ego_pose_to_global_matrix=np.asarray(
                data_model[CameraFrameDatasetSchema.camera_frame_ego_pose_to_global_matrix.name],
                dtype=np.float64,
            ),
        )
