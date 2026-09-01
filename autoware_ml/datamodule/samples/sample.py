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
from jaxtyping import Bool, Int64
from pydantic import BaseModel, ConfigDict, model_validator

from autoware_ml.databases.schemas.dataset_schemas import DatasetRecord
from autoware_ml.datamodule.samples.boxes3d import Boxes3D
from autoware_ml.datamodule.samples.calibration import CalibrationSample
from autoware_ml.datamodule.samples.camera import ImageSet
from autoware_ml.datamodule.samples.meta import FrameMeta
from autoware_ml.datamodule.samples.point_cloud import PointCloud
from autoware_ml.datamodule.samples.segmentation3d import SegmentationLabels


class Sample(BaseModel):
    """
    Typed sample of one frame. A dataset seeds the sample with the dataset record and the frame
    metadata, and the transform pipeline fills the task fields. Every task field is optional, a
    task is active when its field is set. Transforms never mutate a sample, they return a new
    one.

    Attributes:
      record: Dataset record the sample was created from. Loading transforms read their inputs
        from the record.
      data_root: Root directory the relative paths of the record resolve against.
      meta: Frame metadata of the sample.
      points: Point cloud of the sample. None until points are loaded.
      boxes: Detection ground truth boxes. None when detection is inactive.
      segment: Semantic segmentation labels aligned with points. None when segmentation is
        inactive.
      images: Multiview camera images. None when the camera modality is inactive.
      calibration: Calibration status task state. None when the calibration task is inactive.
    """

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    record: DatasetRecord
    data_root: str
    meta: FrameMeta
    points: PointCloud | None = None
    boxes: Boxes3D | None = None
    segment: SegmentationLabels | None = None
    images: ImageSet | None = None
    calibration: CalibrationSample | None = None

    @model_validator(mode="after")
    def validate_sample(self) -> Sample:
        """
        Validate the alignment between the task fields.

        Returns:
          Sample: The validated sample.
        """

        if self.segment is not None:
            if self.points is None:
                raise ValueError("Segmentation labels require a loaded point cloud.")
            if len(self.segment) != len(self.points):
                raise ValueError(
                    f"Segmentation labels cover {len(self.segment)} points but the point cloud "
                    f"holds {len(self.points)} points."
                )
        return self

    def filter_points(self, mask: Bool[np.ndarray, " num_points"]) -> Sample:
        """
        Create a sample keeping only the masked points. The segmentation labels are filtered
        together with the point cloud so both stay aligned.

        Args:
          mask: Boolean mask of the points to keep.

        Returns:
          Sample: Sample with the filtered points.
        """

        if self.points is None:
            raise ValueError("Cannot filter points before the point cloud is loaded.")
        update = {"points": self.points.filter(mask)}
        if self.segment is not None:
            update["segment"] = self.segment.filter(mask)
        return self.model_copy(update=update)

    def reorder_points(self, indices: Int64[np.ndarray, " num_points"]) -> Sample:
        """
        Create a sample with reordered points. The segmentation labels are reordered together
        with the point cloud so both stay aligned.

        Args:
          indices: Permutation of the point indices.

        Returns:
          Sample: Sample with the reordered points.
        """

        if self.points is None:
            raise ValueError("Cannot reorder points before the point cloud is loaded.")
        update = {"points": self.points.reorder(indices)}
        if self.segment is not None:
            update["segment"] = self.segment.reorder(indices)
        return self.model_copy(update=update)
