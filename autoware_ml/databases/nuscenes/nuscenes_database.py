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

"""nuScenes database, generating one record table per devkit version."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from autoware_ml.databases.base_database import BaseDatabase, run_record_workers
from autoware_ml.databases.box3d_pipelines.box3d_pipeline import Box3DPipeline
from autoware_ml.databases.nuscenes.nuscenes_records_generator import NuscenesRecordsGenerator
from autoware_ml.databases.nuscenes.nuscenes_scenarios import NuscenesScenarios
from autoware_ml.databases.scenarios import ScenarioData
from autoware_ml.databases.schemas.dataset_schemas import DatasetRecord


@dataclass(frozen=True)
class NuscenesRecordsGeneratorWorkerParams:
    """
    Parameters for a chunk of nuScenes scenes to be processed by NuscenesRecordsGenerator.

    Attributes:
      database_root_path: Root path of the nuScenes database.
      nuscenes_version: Version of the nuScenes devkit tables.
      scenario_data: Scenario data of the scenes assigned to this worker.
      ignore_label_index: Label index to use for ignored labels.
      box3d_pipelines: List of box 3D pipelines to process the box 3D annotations.
    """

    database_root_path: str
    nuscenes_version: str
    scenario_data: Sequence[ScenarioData]
    ignore_label_index: int
    box3d_pipelines: Sequence[Box3DPipeline]


def _apply_nuscenes_records_generator(
    worker_params: NuscenesRecordsGeneratorWorkerParams,
) -> Sequence[DatasetRecord]:
    """
    Generate the records of one chunk of scenes in a worker process.

    Args:
      worker_params: Parameters of the chunk.

    Returns:
      Sequence[DatasetRecord]: Records of the chunk.
    """

    generator = NuscenesRecordsGenerator(
        database_root_path=worker_params.database_root_path,
        version=worker_params.nuscenes_version,
        scenario_data=worker_params.scenario_data,
        ignore_label_index=worker_params.ignore_label_index,
        box3d_pipelines=worker_params.box3d_pipelines,
    )
    return generator.generate_dataset_records()


class NuscenesDatabase(BaseDatabase):
    """
    nuScenes database. Every worker loads one devkit instance and extracts a chunk of scenes,
    because loading the devkit tables dominates the cost of one scene.
    """

    def __init__(
        self,
        version: str,
        root_path: str,
        scenarios: Mapping[str, NuscenesScenarios],
        cache_path: str,
        cache_file_prefix_name: str,
        num_workers: int,
        class_names: Sequence[str],
        label_remapper: Mapping[str, str],
        ignore_label_index: int,
        box3d_pipelines: Sequence[Box3DPipeline],
        nuscenes_version: str,
    ) -> None:
        """
        Initialize the nuScenes database. Please refer to BaseDatabase for the shared arguments.

        Args:
          version: Version of the database.
          root_path: Root path where the actual annotation files are stored.
          scenarios: Scenario configurations of every scenario group, keyed by group name.
          cache_path: Directory the record table is written to.
          cache_file_prefix_name: Prefix of the record table file.
          num_workers: Number of worker processes. Every worker loads its own devkit instance,
            so keep the count moderate.
          class_names: Class names the box labels are resolved against.
          label_remapper: Mapping from raw dataset label names to class names.
          ignore_label_index: Label index of a box whose class is not trained.
          box3d_pipelines: Box pipelines applied to the box annotations of every sample.
          nuscenes_version: Version of the nuScenes devkit tables, for example v1.0-trainval.
        """

        for scenarios_obj in scenarios.values():
            for dataset_params in scenarios_obj.dataset_params:
                if dataset_params.dataset_name != nuscenes_version:
                    raise ValueError(
                        f"Scenario dataset name {dataset_params.dataset_name} does not match "
                        f"the nuScenes version {nuscenes_version}."
                    )
        self._nuscenes_version = nuscenes_version
        super().__init__(
            version=version,
            root_path=root_path,
            scenarios=scenarios,
            cache_path=cache_path,
            cache_file_prefix_name=cache_file_prefix_name,
            num_workers=num_workers,
            class_names=class_names,
            label_remapper=label_remapper,
            ignore_label_index=ignore_label_index,
            box3d_pipelines=box3d_pipelines,
        )

    @property
    def nuscenes_version(self) -> str:
        """Version of the nuScenes devkit tables."""
        return self._nuscenes_version

    def description_fields(self) -> dict[str, str]:
        """Shared fields plus the nuScenes devkit version."""
        fields = super().description_fields()
        fields["nuscenes_version"] = self._nuscenes_version
        return fields

    def generate_records(
        self, scenario_data: Mapping[str, ScenarioData]
    ) -> Sequence[DatasetRecord]:
        """
        Generate the records of every scene, chunking the scenes across the workers.

        Args:
          scenario_data: Dictionary of scenario ID to scenario data.

        Returns:
          Sequence[DatasetRecord]: Records of every scene.
        """

        scenario_list = list(scenario_data.values())
        num_chunks = min(self.num_workers, len(scenario_list))
        worker_params = [
            NuscenesRecordsGeneratorWorkerParams(
                database_root_path=str(self.root_path),
                nuscenes_version=self._nuscenes_version,
                scenario_data=list(chunk),
                ignore_label_index=self.ignore_label_index,
                box3d_pipelines=self.box3d_pipelines,
            )
            for chunk in np.array_split(scenario_list, num_chunks)
        ]
        return run_record_workers(
            _apply_nuscenes_records_generator, worker_params, self.num_workers
        )
