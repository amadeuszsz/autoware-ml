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

"""Segmentation annotation loading transforms."""

from __future__ import annotations

import numpy as np
from jaxtyping import Int64

from autoware_ml.databases.taxonomy import LabelTaxonomy
from autoware_ml.datamodule.samples.sample import Sample
from autoware_ml.datamodule.samples.segmentation3d import SegmentationLabels
from autoware_ml.transforms.base import BaseTransform
from autoware_ml.transforms.point_cloud.loading import keyframe_lidar_frame, resolve_frame_path


class LoadSeg3DAnnotations(BaseTransform):
    """Load per point semantic labels from the mask file of the sample record.

    The semantic mask of the keyframe lidar frame stores one raw category index per current
    frame point. The category mapping of the record names every raw index, and the
    segmentation taxonomy of the database resolves every category name to its training label.
    A category outside the taxonomy, a raw index the record does not name, and every point of
    a record whose category mapping is empty take the ignore index of the taxonomy.

    The produced segment covers the current frame points only. For a densified cloud
    PreparePointSegInput pads the labels to the full point count, so no transform that filters
    or reorders points may run in between.
    """

    _required_fields = ["points"]

    def __init__(self, *, taxonomy: LabelTaxonomy, dtype: str = "uint8") -> None:
        """Initialize the LoadSeg3DAnnotations transform.

        Args:
            taxonomy: Segmentation taxonomy of the database the records come from.
            dtype: Raw label dtype stored on disk.
        """
        self.taxonomy = taxonomy
        self.dtype = np.dtype(dtype)

    @property
    def ignore_index(self) -> int:
        """Training label of a point outside the taxonomy."""
        return self.taxonomy.ignore_index

    def transform(self, sample: Sample) -> Sample:
        """Load the semantic labels of the current frame from the mask file of the record.

        Args:
            sample: Sample holding the dataset record and a loaded point cloud.

        Returns:
            Sample with the current frame segmentation labels.
        """
        keyframe = keyframe_lidar_frame(sample)
        if keyframe.lidar_pointcloud_semantic_mask_path is None:
            raise ValueError(
                f"The record of sample {sample.meta.sample_id} has no semantic mask path on "
                "its keyframe lidar frame."
            )
        path = resolve_frame_path(sample.data_root, keyframe.lidar_pointcloud_semantic_mask_path)
        raw_labels = np.fromfile(path, dtype=self.dtype).astype(np.int64)

        lookup = self._category_lookup(sample)
        labels = np.full(raw_labels.shape, self.ignore_index, dtype=np.int64)
        valid = (raw_labels >= 0) & (raw_labels < lookup.shape[0])
        labels[valid] = lookup[raw_labels[valid]]
        return sample.model_copy(update={"segment": SegmentationLabels(labels=labels)})

    def _category_lookup(self, sample: Sample) -> Int64[np.ndarray, " lookup_size"]:
        """Build the raw label lookup table from the category mapping of the record.

        Args:
            sample: Sample holding the dataset record.

        Returns:
            Int64[np.ndarray, " lookup_size"]: Training label per raw label.
        """
        category_mapping = sample.record.category_mapping
        if category_mapping is None:
            raise ValueError(
                f"The record of sample {sample.meta.sample_id} carries no category mapping, "
                "so its semantic mask cannot be resolved."
            )
        lookup_size = 0
        if len(category_mapping.category_indices):
            lookup_size = max(category_mapping.category_indices) + 1
        lookup = np.full(lookup_size, fill_value=self.ignore_index, dtype=np.int64)
        for category_name, raw_label in zip(
            category_mapping.category_names, category_mapping.category_indices
        ):
            lookup[raw_label] = self.taxonomy.resolve_index(category_name)
        return lookup
