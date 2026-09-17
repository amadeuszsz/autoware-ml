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

"""Pure geometry helpers shared by the transforms."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def rotation_matrix(axis: str, angle: float) -> npt.NDArray[np.float32]:
    """Build a 3x3 rotation matrix for a single axis.

    Args:
        axis: Rotation axis, one of x, y or z.
        angle: Rotation in radians.

    Returns:
        npt.NDArray[np.float32]: The rotation matrix.

    Raises:
        NotImplementedError: If the axis is not one of x, y or z.
    """
    cos, sin = np.cos(angle), np.sin(angle)
    if axis == "x":
        return np.array([[1, 0, 0], [0, cos, -sin], [0, sin, cos]], dtype=np.float32)
    if axis == "y":
        return np.array([[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]], dtype=np.float32)
    if axis == "z":
        return np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]], dtype=np.float32)
    raise NotImplementedError(f"Unsupported rotation axis: {axis}")
