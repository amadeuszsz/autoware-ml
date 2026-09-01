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

from autoware_ml.datamodule.samples.sample import Sample
from autoware_ml.datamodule.samples.segmentation3d import SegmentationLabels
from autoware_ml.transforms.base import BaseTransform
from autoware_ml.transforms.point_cloud.loading import keyframe_lidar_frame, resolve_frame_path


class LoadSeg3DAnnotations(BaseTransform):
    """Load per point semantic labels from the mask file of the sample record.

    The semantic mask of the keyframe lidar frame stores one raw label per current frame point.
    The raw labels are remapped to training labels in one of two exclusive modes. With
    label_mapping the stored values are raw integer labels and the mapping translates them
    directly, the nuScenes lidarseg layout. With class_mapping the stored values index the
    category mapping of the record and the mapping translates the category names, the T4
    layout. Values without a mapping entry become ignore_index.

    The produced segment covers the current frame points only. For a densified cloud
    PreparePointSegInput pads the labels to the full point count, so no transform that filters
    or reorders points may run in between.
    """

    _required_fields = ["points"]

    def __init__(
        self,
        *,
        dtype: str = "uint8",
        label_mapping: dict[int, int] | None = None,
        max_label: int | None = None,
        class_mapping: dict[str, int] | None = None,
        ignore_index: int = -1,
    ) -> None:
        """Initialize the LoadSeg3DAnnotations transform.

        Args:
            dtype: Raw label dtype stored on disk.
            label_mapping: Optional raw label to training label mapping.
            max_label: Optional maximum raw label used to size the lookup table.
            class_mapping: Optional category name to training label mapping.
            ignore_index: Ignore label used for unknown categories.
        """
        if (label_mapping is None) == (class_mapping is None):
            raise ValueError(
                "LoadSeg3DAnnotations requires exactly one of 'label_mapping' "
                "(raw int to train label, e.g. nuScenes) or 'class_mapping' "
                "(category name to train label resolved through the category mapping of the "
                f"record, e.g. T4), got label_mapping={label_mapping is not None}, "
                f"class_mapping={class_mapping is not None}."
            )
        self.dtype = np.dtype(dtype)
        self.label_mapping = label_mapping
        self.max_label = max_label
        self.class_mapping = class_mapping
        self.ignore_index = ignore_index

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

        if self.class_mapping is not None:
            lookup = self._category_lookup(sample)
        else:
            lookup = self._label_lookup()

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
                "LoadSeg3DAnnotations was configured with 'class_mapping' but the record "
                f"of sample {sample.meta.sample_id} carries no category mapping to remap "
                "from. Provide the category mapping, or configure 'label_mapping' for raw "
                "integer masks."
            )
        lookup_size = 0
        if len(category_mapping.category_indices):
            lookup_size = max(category_mapping.category_indices) + 1
        lookup = np.full(lookup_size, fill_value=self.ignore_index, dtype=np.int64)
        for category_name, raw_label in zip(
            category_mapping.category_names, category_mapping.category_indices
        ):
            lookup[raw_label] = self.class_mapping.get(category_name, self.ignore_index)
        return lookup

    def _label_lookup(self) -> Int64[np.ndarray, " lookup_size"]:
        """Build the raw label lookup table from the configured label mapping.

        Returns:
            Int64[np.ndarray, " lookup_size"]: Training label per raw label.
        """
        lookup_size = (
            self.max_label + 1 if self.max_label is not None else max(self.label_mapping) + 1
        )
        lookup = np.full(lookup_size, fill_value=self.ignore_index, dtype=np.int64)
        for source_label, target_label in self.label_mapping.items():
            lookup[int(source_label)] = int(target_label)
        return lookup
