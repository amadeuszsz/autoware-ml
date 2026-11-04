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

"""Online map targets of a record, the lanelet2 map elements around the ego."""

import math
from collections.abc import Mapping, Sequence

import numpy as np
import numpy.typing as npt
import polars as pl

from autoware_ml.databases.schemas.dataset_schemas import DatasetTableSchema
from autoware_ml.databases.schemas.lidar_frames import LidarFrameDatasetSchema
from autoware_ml.dataclasses.batch.online_map import OnlineMapGTSample, VectorMapSpec
from autoware_ml.datamodule.t4dataset.frame_meta import scene_dir_fragment
from autoware_ml.geometry.vector_map import MapClassSource, VectorMap
from autoware_ml.metrics.geometry.lanelet import T4LaneletMapResolver


class OnlineMapTargets:
    """Read the classed map elements around every record from the lanelet2 map of its scene.

    The motion task builds it over its records for the online map. Every scene must ship its
    map. The elements are cut to a square around the frame that holds the mapped region under
    any augmentation within the margin.
    """

    def __init__(
        self,
        database_root_path: str,
        dataset_records_dataframe: pl.DataFrame,
        poses: npt.NDArray[np.float64],
        *,
        classes: Mapping[str, Mapping[str, Sequence[str]]],
        roi: Sequence[float],
        num_points: int,
        min_length: float,
        augmentation_margin: float,
        map_root_path: str | None = None,
    ) -> None:
        """
        Resolve the map of every record.

        Args:
          database_root_path: Root directory of the database.
          dataset_records_dataframe: Records of the corpus with their lidar frames.
          poses: Transform of every record from the frame of its point cloud to the global frame.
          classes: Source of every class, ``line_types`` or ``regions``, in class index order.
          roi: Mapped region of the augmented frame ``(x_min, y_min, x_max, y_max)``.
          num_points: Points every element is resampled to.
          min_length: Shortest piece of an element the region cut keeps, in meters.
          augmentation_margin: Distance the augmentation may move the region corners by.
          map_root_path: Root holding the scene directories with the maps, when the maps live
            apart from the point clouds. None reads them from the database root.

        Raises:
          ValueError: If the region, the point count, the length or the margin is invalid.
        """
        x_min, y_min, x_max, y_max = (float(value) for value in roi)
        if not (x_min < x_max and y_min < y_max):
            raise ValueError(f"roi must be (x_min, y_min, x_max, y_max), got {roi}.")
        if num_points < 2 or min_length < 0 or augmentation_margin < 0:
            raise ValueError(
                f"Map elements need num_points >= 2, min_length >= 0 and augmentation_margin "
                f">= 0, got {num_points}, {min_length} and {augmentation_margin}."
            )
        self.class_sources = {name: MapClassSource(**source) for name, source in classes.items()}
        self.spec = VectorMapSpec(
            roi=(x_min, y_min, x_max, y_max),
            num_points=num_points,
            min_length=min_length,
            crop_half_size=math.hypot(max(-x_min, x_max), max(-y_min, y_max)) + augmentation_margin,
        )
        records = dataset_records_dataframe
        self.poses = poses
        resolve = T4LaneletMapResolver(
            database_root_path if map_root_path is None else map_root_path
        )
        scenes = (
            records.select(
                pl.col(DatasetTableSchema.LIDAR_FRAMES.name)
                .list.first()
                .struct.field(LidarFrameDatasetSchema.lidar_pointcloud_relative_path.name)
            )
            .to_series()
            .map_elements(scene_dir_fragment, return_dtype=pl.String)
        )
        osm_paths = {scene: resolve(scene) for scene in scenes.unique()}
        self.osm_paths = [osm_paths[scene] for scene in scenes]

    def get_gt_sample(self, idx: int) -> OnlineMapGTSample:
        """
        Cut the map elements around one record.

        Args:
          idx: Index of the record.

        Returns:
          OnlineMapGTSample: The map elements of the record.
        """
        vector_map = VectorMap.from_osm(self.osm_paths[idx], self.class_sources)
        elements = vector_map.crop(self.poses[idx], self.spec.crop_half_size)
        return OnlineMapGTSample(elements=elements, spec=self.spec)
