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

"""Typed decoded predictions of the 3D detection heads."""

from __future__ import annotations

from jaxtyping import Float32, Int64
from pydantic import BaseModel, ConfigDict
from torch import Tensor


class Detection3DPrediction(BaseModel):
    """Decoded detections of one sample.

    Attributes:
      bboxes_3d: Box parameters of every detection, following Box3DFieldIndex without the
        vertical velocity.
      scores_3d: Confidence score of every detection.
      labels_3d: Label index of every detection.
    """

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    bboxes_3d: Float32[Tensor, "num_detections num_box_params"]
    scores_3d: Float32[Tensor, " num_detections"]
    labels_3d: Int64[Tensor, " num_detections"]
