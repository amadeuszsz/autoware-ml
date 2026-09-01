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

import logging
from collections.abc import Mapping, Sequence

import numpy as np

from autoware_ml.datamodule.samples.boxes3d import NUM_BOX_PARAMS, Boxes3D
from autoware_ml.datamodule.samples.sample import Sample
from autoware_ml.transforms.base import BaseTransform
from autoware_ml.transforms.boxes3d.annotations import (
    box_is_physical,
    normalize_filter_attributes,
    resolve_detection_class,
    sanitize_box_params,
)

logger = logging.getLogger(__name__)


class LoadDet3DAnnotations(BaseTransform):
    """Load the 3D box annotations of the sample record into detection targets.

    The transform reads the stored box annotations of the record, resolves every box through
    the configured taxonomy, drops boxes whose class is outside the detector classes or whose
    parameters are not physically trainable, and packs the survivors into the boxes of the
    sample with one label index per detector class.
    """

    def __init__(
        self,
        *,
        class_names: Sequence[str],
        name_mapping: Mapping[str, str | None] | None = None,
        filter_attributes: Sequence[Sequence[str]] | None = None,
    ) -> None:
        """Initialize the LoadDet3DAnnotations transform.

        Args:
            class_names: Detector class names in label order.
            name_mapping: Optional raw dataset category to detector class mapping. Values set
                to None drop the corresponding raw category.
            filter_attributes: Class and attribute name pairs whose boxes are dropped.
        """
        self.class_names = tuple(str(name) for name in class_names)
        if not self.class_names:
            raise ValueError("LoadDet3DAnnotations requires at least one detector class name.")
        self.name_mapping = dict(name_mapping) if name_mapping is not None else None
        self.filter_attributes = normalize_filter_attributes(filter_attributes)
        self._log_dropped_mapping_targets()

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
        names: list[str] = []
        num_lidar_points: list[int] = []
        for box in boxes_3d:
            canonical = resolve_detection_class(
                box,
                class_names=self.class_names,
                name_mapping=self.name_mapping,
                filter_attributes=self.filter_attributes,
            )
            if canonical is None:
                continue
            params = sanitize_box_params(box.box3d_params)
            if not box_is_physical(params):
                continue
            params_rows.append(params)
            names.append(canonical)
            num_lidar_points.append(int(box.box3d_num_lidar_points))

        if params_rows:
            params_matrix = np.stack(params_rows).astype(np.float32)
        else:
            params_matrix = np.zeros((0, NUM_BOX_PARAMS), dtype=np.float32)
        name_to_label = {name: index for index, name in enumerate(self.class_names)}
        boxes = Boxes3D(
            params=params_matrix,
            labels=np.array([name_to_label[name] for name in names], dtype=np.int64),
            names=tuple(names),
            num_lidar_points=np.array(num_lidar_points, dtype=np.int64),
        )
        return sample.model_copy(update={"boxes": boxes})

    def _log_dropped_mapping_targets(self) -> None:
        """Log mapping targets dropped because they are not detector classes.

        A name_mapping target absent from class_names is treated as an intentional drop
        (for example mapping trailer to the non-target class trailer so standalone trailers
        are excluded). Such boxes are dropped by resolve_detection_class, this only surfaces
        them once at construction.
        """
        if self.name_mapping is None:
            return
        dropped_targets = sorted(
            {
                str(mapped_name)
                for mapped_name in self.name_mapping.values()
                if mapped_name is not None and str(mapped_name) not in self.class_names
            }
        )
        if dropped_targets:
            logger.info(
                "name_mapping targets not in class_names will be dropped: %s", dropped_targets
            )
