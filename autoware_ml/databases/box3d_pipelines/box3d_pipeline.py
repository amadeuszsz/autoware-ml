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

from typing import Sequence

from autoware_ml.databases.schemas.box3d_schemas import Box3DDataModel
from autoware_ml.databases.taxonomy import LabelTaxonomy


class Box3DPipeline:
    """
    Base class for box 3D pipelines. A pipeline runs on boxes carrying their fine label name,
    after the vocabulary resolved the raw names and before the class indices of the level are
    assigned.
    """

    def __call__(self, boxes3d_data_model: Sequence[Box3DDataModel]) -> Sequence[Box3DDataModel]:
        """Process the boxes 3D."""
        raise NotImplementedError("Subclass must implement this method")

    def validate_taxonomy(self, taxonomy: LabelTaxonomy) -> None:
        """
        Check that the pipeline is consistent with the taxonomy the boxes are baked with. A
        pipeline whose behaviour depends on label names rejects a taxonomy it would corrupt.

        Args:
          taxonomy: Taxonomy the boxes are baked with.
        """

    def __str__(self) -> str:
        """
        String representation of the pipeline, used for logging.

        Returns:
          str: String representation of the pipeline.
        """
        return self.__class__.__name__
