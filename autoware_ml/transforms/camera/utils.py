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

"""Shared helpers of the calibration camera transforms."""

from __future__ import annotations

from autoware_ml.utils.calibration import CalibrationData


def copy_calibration_data(data: CalibrationData) -> CalibrationData:
    """Create a deep copy of a calibration data instance.

    CalibrationData is a mutable dataclass, so transforms that update the calibration copy it
    first and leave the instance of the input sample untouched.

    Args:
        data: Calibration data to copy.

    Returns:
        Independent copy with copied arrays.
    """
    return CalibrationData(
        camera_matrix=data.camera_matrix.copy(),
        distortion_coefficients=data.distortion_coefficients.copy(),
        lidar_to_camera_transformation=data.lidar_to_camera_transformation.copy(),
        distortion_model=data.distortion_model,
        noise=None if data.noise is None else data.noise.copy(),
        new_camera_matrix=None if data.new_camera_matrix is None else data.new_camera_matrix.copy(),
    )
