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

"""
Modules to save raw outputs from a polyline map head.
"""

from __future__ import annotations

from jaxtyping import Bool, Float
from pydantic import BaseModel, ConfigDict, model_validator

import torch


class OnlineMapHeadOutputs(BaseModel):
    """
    Dataclass to save the raw outputs of a polyline map head.

    Polylines are BEV points in meters in the frame of the (augmented) point cloud, flattened to
    ``(x_0, y_0, x_1, y_1, ...)``.

    Attributes:
      classification: Class logits of every query at every decoder layer.
      prediction: Polyline of every query at every decoder layer.
      instance_feature: Last layer feature of every query.
    """

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    classification: Float[torch.Tensor, "batch_size num_layers num_queries num_classes"]
    prediction: Float[torch.Tensor, "batch_size num_layers num_queries polyline_dims"]
    instance_feature: Float[torch.Tensor, "batch_size num_queries embed_dims"]

    @model_validator(mode="after")
    def _check_shapes(self) -> OnlineMapHeadOutputs:
        """Check that every tensor describes the same queries."""
        if self.classification.shape[:3] != self.prediction.shape[:3]:
            raise ValueError(
                "classification and prediction must share batch, layers and queries, got "
                f"{tuple(self.classification.shape)} and {tuple(self.prediction.shape)}."
            )
        batch_size, _, num_queries = self.prediction.shape[:3]
        if tuple(self.instance_feature.shape[:2]) != (batch_size, num_queries):
            raise ValueError(
                "instance_feature must share batch and queries with prediction, got "
                f"{tuple(self.instance_feature.shape)} and {tuple(self.prediction.shape)}."
            )
        return self

    def select_samples(self, mask: Bool[torch.Tensor, " batch_size"]) -> OnlineMapHeadOutputs:
        """
        Keep the outputs of the samples a mask selects.

        Args:
            mask: True for every sample to keep.

        Returns:
            The outputs of the selected samples.
        """
        return OnlineMapHeadOutputs(
            classification=self.classification[mask],
            prediction=self.prediction[mask],
            instance_feature=self.instance_feature[mask],
        )
