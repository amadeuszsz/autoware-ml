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

"""3D bounding-box annotation loading transforms."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from autoware_ml.datamodule.samples.boxes3d import NUM_BOX_PARAMS, Boxes3D
from autoware_ml.datamodule.samples.sample import Sample
from autoware_ml.transforms.base import BaseTransform
from autoware_ml.transforms.boxes3d.annotations import (
    box_is_physical,
    normalize_filter_attributes,
    resolve_box_class,
    sanitize_box_params,
)


class LoadDet3DAnnotations(BaseTransform):
    """Load the 3D box annotations of the sample record into detection targets.

    The record carries the class index of every box already baked by the database taxonomy.
    The transform drops the boxes outside the trained classes, the boxes matching a class and
    attribute exclusion, and the boxes whose parameters are not physically trainable, and
    packs the survivors into the boxes of the sample with the stored label index and the
    trained class name at that index.
    """

    def __init__(
        self,
        *,
        class_names: Sequence[str],
        ignore_label_index: int,
        filter_attributes: Sequence[Sequence[str]] | None = None,
    ) -> None:
        """Initialize the LoadDet3DAnnotations transform.

        Args:
            class_names: Trained class names, in index order.
            ignore_label_index: Label index of a box whose class is not trained.
            filter_attributes: Class and attribute name pairs whose boxes are dropped.
        """
        if not len(class_names):
            raise ValueError("LoadDet3DAnnotations requires at least one class name.")
        self.class_names = tuple(class_names)
        self.ignore_label_index = ignore_label_index
        self.filter_attributes = normalize_filter_attributes(filter_attributes)

    def transform(self, sample: Sample) -> Sample:
        """Convert the stored box annotations into detection target boxes.

        Args:
            sample: Sample whose record carries 3D box annotations.

        Returns:
            Sample with the loaded detection boxes.
        """
        boxes_3d = sample.record.boxes_3d
        if boxes_3d is None:
            raise ValueError(
                f"The record of sample {sample.meta.sample_id} carries no 3D box annotations."
            )

        params_rows: list[np.ndarray] = []
        labels: list[int] = []
        names: list[str] = []
        num_lidar_points: list[int] = []
        for box in boxes_3d:
            class_name = resolve_box_class(
                box,
                class_names=self.class_names,
                ignore_label_index=self.ignore_label_index,
                filter_attributes=self.filter_attributes,
            )
            if class_name is None:
                continue
            params = sanitize_box_params(box.box3d_params)
            if not box_is_physical(params):
                continue
            params_rows.append(params)
            labels.append(int(box.box3d_label_index))
            names.append(class_name)
            num_lidar_points.append(int(box.box3d_num_lidar_points))

        if params_rows:
            params_matrix = np.stack(params_rows).astype(np.float32)
        else:
            params_matrix = np.zeros((0, NUM_BOX_PARAMS), dtype=np.float32)
        boxes = Boxes3D(
            params=params_matrix,
            labels=np.array(labels, dtype=np.int64),
            names=tuple(names),
            num_lidar_points=np.array(num_lidar_points, dtype=np.int64),
        )
        return sample.replace(boxes=boxes)
