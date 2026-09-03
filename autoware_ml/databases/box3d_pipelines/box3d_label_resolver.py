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

"""Baking of the box labels when the record table is generated."""

from __future__ import annotations

from typing import Sequence

from autoware_ml.databases.box3d_pipelines.box3d_pipeline import Box3DPipeline
from autoware_ml.databases.schemas.box3d_schemas import Box3DDataModel
from autoware_ml.databases.taxonomy import LabelTaxonomy


class Box3DLabelResolver:
    """
    Resolve the labels of the boxes of one sample in three stages: the vocabulary turns the
    raw dataset name of every box into its fine name, the box pipelines run on the fine names,
    and the taxonomy assigns the class index of its level to every box that survives. The
    stored label name is the fine name, so every table keeps the finest label the corpus
    supports whatever level it was baked for. A box whose raw name the vocabulary places
    outside every level is not stored, and a raw name the vocabulary does not list is an error.
    """

    def __init__(self, taxonomy: LabelTaxonomy, box3d_pipelines: Sequence[Box3DPipeline]) -> None:
        """
        Initialize the resolver.

        Args:
          taxonomy: Taxonomy the boxes are baked with.
          box3d_pipelines: Pipelines run on the boxes between name and index resolution.
        """

        for pipeline in box3d_pipelines:
            pipeline.validate_taxonomy(taxonomy)
        self._taxonomy = taxonomy
        self._box3d_pipelines = tuple(box3d_pipelines)

    @property
    def taxonomy(self) -> LabelTaxonomy:
        """Taxonomy the boxes are baked with."""
        return self._taxonomy

    @property
    def ignore_index(self) -> int:
        """Label index of a box outside the classes of the level."""
        return self._taxonomy.ignore_index

    def __call__(self, boxes3d_data_model: Sequence[Box3DDataModel]) -> list[Box3DDataModel]:
        """
        Resolve the labels of the boxes of one sample.

        Args:
          boxes3d_data_model: Boxes carrying their raw dataset label name.

        Returns:
          list[Box3DDataModel]: Boxes carrying their fine label name and class index.
        """

        fine_names = [
            self._taxonomy.fine_name(box.box3d_dataset_label_name) for box in boxes3d_data_model
        ]
        boxes = [
            box.create_new_data_model(
                box3d_label_name=fine_name, box3d_label_index=self._taxonomy.ignore_index
            )
            for box, fine_name in zip(boxes3d_data_model, fine_names)
            if fine_name is not None
        ]
        for pipeline in self._box3d_pipelines:
            boxes = list(pipeline(boxes))
        return [
            box.create_new_data_model(
                box3d_label_index=self._taxonomy.class_index(box.box3d_label_name)
            )
            for box in boxes
        ]
