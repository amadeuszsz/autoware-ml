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

"""Camera image loading transforms."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
from jaxtyping import Float32, Float64, UInt8

from autoware_ml.databases.schemas.camera_frames import CameraFrameDataModel
from autoware_ml.datamodule.samples.camera import ImageSet
from autoware_ml.datamodule.samples.sample import Sample
from autoware_ml.transforms.base import BaseTransform
from autoware_ml.transforms.point_cloud.loading import (
    keyframe_lidar_frame,
    resolve_frame_path,
)

COLOR_TYPES = frozenset({"rgb", "bgr"})


def camera_frame_by_channel(sample: Sample, channel_name: str) -> CameraFrameDataModel:
    """Get the camera frame of one channel from the sample record.

    Args:
        sample: Sample holding the dataset record.
        channel_name: Camera channel name of the requested frame.

    Returns:
        The camera frame of the channel.

    Raises:
        ValueError: If the record has no camera frames or the channel is missing.
    """
    if sample.record.camera_frames is None:
        raise ValueError(f"The record of sample {sample.meta.sample_id} has no camera frames.")
    for camera_frame in sample.record.camera_frames:
        if camera_frame.camera_sensor_channel_name == channel_name:
            return camera_frame
    raise ValueError(
        f"The record of sample {sample.meta.sample_id} has no camera frame for channel "
        f"'{channel_name}'."
    )


def load_frame_image(
    data_root: str, camera_frame: CameraFrameDataModel
) -> UInt8[np.ndarray, "height width 3"]:
    """Load the BGR image of one camera frame.

    Args:
        data_root: Root directory of the dataset files.
        camera_frame: Camera frame data model of the frame.

    Returns:
        UInt8[np.ndarray, "height width 3"]: BGR image in height, width, channels layout.

    Raises:
        FileNotFoundError: If the image file cannot be read.
    """
    path = resolve_frame_path(data_root, camera_frame.camera_image_path)
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return image


class LoadImageFromFile(BaseTransform):
    """Load the image of the calibration camera into the calibration state.

    The transform reads the camera frame of the record whose channel matches the camera name of
    the calibration state and loads its image.
    """

    _required_fields = ["calibration"]

    def __init__(self, *, color_type: str = "rgb") -> None:
        """Initialize the LoadImageFromFile transform.

        Args:
            color_type: Output color format, "rgb" or "bgr".
        """
        if color_type.lower() not in COLOR_TYPES:
            raise ValueError(
                f"color_type must be one of {sorted(COLOR_TYPES)}, got {color_type!r}."
            )
        self.color_type = color_type.lower()

    def transform(self, sample: Sample) -> Sample:
        """Load the image of the calibration camera.

        Args:
            sample: Sample holding the dataset record and the calibration state.

        Returns:
            Sample with the loaded calibration image.
        """
        camera_frame = camera_frame_by_channel(sample, sample.calibration.camera_name)
        image = load_frame_image(sample.data_root, camera_frame)
        if self.color_type == "rgb":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        calibration = sample.calibration.model_copy(update={"image": image.astype(np.float32)})
        return sample.model_copy(update={"calibration": calibration})


class LoadMultiViewImagesFromFiles(BaseTransform):
    """Load synchronized multiview images and derive the camera projection matrices.

    The transform reads the camera frames of the record in the configured channel order and
    composes the lidar to camera matrix through both ego poses, the one at the lidar timestamp
    and the one at the camera timestamp, so ego motion between the captures is accounted for:

        lidar2cam = inv(cam2ego) @ inv(ego2global_cam) @ ego2global_lidar @ lidar2ego
    """

    def __init__(self, *, camera_order: Sequence[str], normalize_to_unit: bool = True) -> None:
        """Initialize the LoadMultiViewImagesFromFiles transform.

        Args:
            camera_order: Camera channel names in the order the image set exposes them. Every
                named channel must exist in the record.
            normalize_to_unit: Whether to divide pixel values by 255.
        """
        self.camera_order = tuple(camera_order)
        if not self.camera_order:
            raise ValueError("camera_order must name at least one camera channel.")
        self.normalize_to_unit = normalize_to_unit

    def transform(self, sample: Sample) -> Sample:
        """Load the images and camera matrices of every configured channel.

        Args:
            sample: Sample holding the dataset record.

        Returns:
            Sample with the loaded image set.
        """
        keyframe = keyframe_lidar_frame(sample)
        lidar2ego = keyframe.lidar_sensor_to_ego_pose_matrix
        ego2global_lidar = keyframe.lidar_frame_ego_pose_to_global_matrix

        images = []
        intrinsics = []
        lidar2cam = []
        lidar2img = []
        for channel_name in self.camera_order:
            camera_frame = camera_frame_by_channel(sample, channel_name)
            image = load_frame_image(sample.data_root, camera_frame)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
            if self.normalize_to_unit:
                image = image / 255.0
            images.append(np.transpose(image, (2, 0, 1)))

            camera_matrix = np.eye(4, dtype=np.float32)
            camera_matrix[:3, :3] = camera_frame.camera_intrinsic_matrix_fp32
            intrinsics.append(camera_matrix)

            camera_transform = self._compose_lidar2cam(camera_frame, lidar2ego, ego2global_lidar)
            lidar2cam.append(camera_transform)
            lidar2img.append(camera_matrix @ camera_transform)

        image_set = ImageSet(
            images=np.stack(images, axis=0),
            camera_names=self.camera_order,
            camera_intrinsics=np.stack(intrinsics, axis=0),
            lidar2cam=np.stack(lidar2cam, axis=0),
            lidar2img=np.stack(lidar2img, axis=0),
        )
        return sample.model_copy(update={"images": image_set})

    @staticmethod
    def _compose_lidar2cam(
        camera_frame: CameraFrameDataModel,
        lidar2ego: Float64[np.ndarray, "4 4"],
        ego2global_lidar: Float64[np.ndarray, "4 4"],
    ) -> Float32[np.ndarray, "4 4"]:
        """Compose the lidar to camera matrix through both ego poses.

        Args:
            camera_frame: Camera frame providing the camera pose and its ego pose.
            lidar2ego: Lidar sensor to ego pose matrix at the lidar timestamp.
            ego2global_lidar: Ego pose to global matrix at the lidar timestamp.

        Returns:
            Float32[np.ndarray, "4 4"]: The lidar to camera matrix.
        """
        cam2ego = camera_frame.camera_sensor_to_ego_pose_matrix
        ego2global_cam = camera_frame.camera_frame_ego_pose_to_global_matrix
        lidar2cam = (
            np.linalg.inv(cam2ego) @ np.linalg.inv(ego2global_cam) @ ego2global_lidar @ lidar2ego
        )
        return lidar2cam.astype(np.float32)
