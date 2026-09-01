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

import numpy as np
from jaxtyping import Float64
from pydantic import BaseModel, ConfigDict, model_validator


class FrameMeta(BaseModel):
    """
    Frame metadata of one sample, used by evaluation, map based metric filters, and temporal
    models. The metadata is derived from the dataset record when the sample is created.

    Attributes:
      sample_id: Unique ID of the sample within the dataset.
      scene_token: Scene identifier used to resolve per scene resources such as lanelet maps.
        None when the dataset has no scene resources.
      timestamp_seconds: Timestamp of the sample in seconds.
      ego2global: Transformation matrix from the ego frame to the global frame at the sample
        timestamp. None when the dataset has no ego poses.
      location: Location where the sample was recorded, when known.
      vehicle_type: Type of the recording vehicle, when known.
      prev_exists: Whether the previous sample of the scene exists, used by temporal models.
    """

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    sample_id: str
    scene_token: str | None
    timestamp_seconds: float
    ego2global: Float64[np.ndarray, "4 4"] | None
    location: str | None = None
    vehicle_type: str | None = None
    prev_exists: bool | None = None

    @model_validator(mode="after")
    def validate_meta(self) -> FrameMeta:
        """
        Validate the ego pose matrix.

        Returns:
          FrameMeta: The validated metadata.
        """

        if self.ego2global is not None and self.ego2global.shape != (4, 4):
            raise ValueError(
                f"The ego2global matrix must have shape (4, 4), got {self.ego2global.shape}."
            )
        return self
