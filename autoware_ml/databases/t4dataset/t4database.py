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

"""T4dataset database, generating one record table per set of scenario lists."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from autoware_ml.databases.base_database import BaseDatabase, run_record_workers
from autoware_ml.databases.box3d_pipelines.box3d_label_resolver import Box3DLabelResolver
from autoware_ml.databases.box3d_pipelines.box3d_pipeline import Box3DPipeline
from autoware_ml.databases.scenarios import ScenarioData
from autoware_ml.databases.schemas.dataset_schemas import DatasetRecord
from autoware_ml.databases.t4dataset.t4records_generator import T4RecordsGenerator
from autoware_ml.databases.t4dataset.t4scenarios import T4Scenarios
from autoware_ml.databases.taxonomy import DatabaseTaxonomy, LabelTaxonomy
from autoware_ml.types.sensor import LidarChannel


@dataclass(frozen=True)
class T4RecordsGeneratorWorkerParams:
    """
    Parameters of one scenario processed by a T4RecordsGenerator worker.

    Attributes:
      database_root_path: Root path of the T4 database.
      scenario_data: Scenario to extract.
      lidar_channel: Sensor channel of the lidar frame every sample is built around.
      box3d_label_resolver: Resolver baking the label of every box.
      segmentation_taxonomy: Taxonomy the mask categories of the scene must be listed in.
      recompute_boxes3d_lidar_points_num: Whether to recount the lidar points inside every box.
    """

    database_root_path: str
    scenario_data: ScenarioData
    lidar_channel: str
    box3d_label_resolver: Box3DLabelResolver
    segmentation_taxonomy: LabelTaxonomy
    recompute_boxes3d_lidar_points_num: bool


def _apply_t4_records_generator(
    worker_params: T4RecordsGeneratorWorkerParams,
) -> Sequence[DatasetRecord]:
    """
    Generate the records of one scenario in a worker process.

    Args:
      worker_params: Parameters of the scenario.

    Returns:
      Sequence[DatasetRecord]: Records of the scenario.
    """

    generator = T4RecordsGenerator(
        database_root_path=worker_params.database_root_path,
        scenario_data=worker_params.scenario_data,
        lidar_channel=worker_params.lidar_channel,
        box3d_label_resolver=worker_params.box3d_label_resolver,
        segmentation_taxonomy=worker_params.segmentation_taxonomy,
        recompute_boxes3d_lidar_points_num=worker_params.recompute_boxes3d_lidar_points_num,
    )
    return generator.generate_dataset_records()


class T4Database(BaseDatabase):
    """T4dataset database. One worker extracts one scenario."""

    def __init__(
        self,
        version: str,
        root_path: str,
        scenarios: Mapping[str, T4Scenarios],
        cache_path: str,
        cache_file_prefix_name: str,
        num_workers: int,
        taxonomy: DatabaseTaxonomy,
        box3d_pipelines: Sequence[Box3DPipeline],
        lidar_channel: str,
        recompute_boxes3d_lidar_points_num: bool = False,
    ) -> None:
        """
        Initialize the T4 database. Please refer to BaseDatabase for the shared arguments.

        Args:
          version: Version of the database.
          root_path: Root path where the actual annotation files are stored.
          scenarios: Scenario configurations of every scenario group, keyed by group name.
          cache_path: Directory the record table is written to.
          cache_file_prefix_name: Prefix of the record table file.
          num_workers: Number of worker processes used to generate the records.
          taxonomy: Taxonomies the box labels are baked with and the mask categories are
            resolved with.
          box3d_pipelines: Box pipelines applied to the box annotations of every sample.
          lidar_channel: Sensor channel of the lidar frame every sample is built around.
          recompute_boxes3d_lidar_points_num: Whether to recount the lidar points inside every
            box from the point cloud after the pipelines ran.
        """

        self._lidar_channel = LidarChannel(lidar_channel).value
        self._recompute_boxes3d_lidar_points_num = recompute_boxes3d_lidar_points_num
        super().__init__(
            version=version,
            root_path=root_path,
            scenarios=scenarios,
            cache_path=cache_path,
            cache_file_prefix_name=cache_file_prefix_name,
            num_workers=num_workers,
            taxonomy=taxonomy,
            box3d_pipelines=box3d_pipelines,
        )

    @property
    def lidar_channel(self) -> str:
        """Sensor channel of the lidar frame every sample is built around."""
        return self._lidar_channel

    def description_fields(self) -> dict[str, str]:
        """Shared fields plus the T4 specific generation parameters."""
        fields = super().description_fields()
        fields["lidar_channel"] = self._lidar_channel
        fields["recompute_boxes3d_lidar_points_num"] = str(self._recompute_boxes3d_lidar_points_num)
        return fields

    def generate_records(
        self, scenario_data: Mapping[str, ScenarioData]
    ) -> Sequence[DatasetRecord]:
        """
        Generate the records of every scenario, one worker per scenario.

        Args:
          scenario_data: Dictionary of scenario ID to scenario data.

        Returns:
          Sequence[DatasetRecord]: Records of every scenario.
        """

        worker_params = [
            T4RecordsGeneratorWorkerParams(
                database_root_path=str(self.root_path),
                scenario_data=scenario,
                lidar_channel=self._lidar_channel,
                box3d_label_resolver=self.box3d_label_resolver,
                segmentation_taxonomy=self.taxonomy.segmentation3d,
                recompute_boxes3d_lidar_points_num=self._recompute_boxes3d_lidar_points_num,
            )
            for scenario in scenario_data.values()
        ]
        return run_record_workers(_apply_t4_records_generator, worker_params, self.num_workers)
