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

import logging
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Sequence
from pathlib import Path
from types import MappingProxyType

import numpy as np
import polars as pl
from tqdm import tqdm

from autoware_ml.databases.nuscenes.nuscenes_records_generator import (
    NuscenesRecordsGenerator,
)
from autoware_ml.databases.nuscenes.nuscenes_scenarios import NuscenesScenarios
from autoware_ml.databases.scenarios import ScenarioData
from autoware_ml.databases.schemas.dataset_schemas import DatasetRecord, DatasetTableSchema
from autoware_ml.databases.box3d_pipelines.box3d_pipeline import Box3DPipeline

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NuscenesRecordsGeneratorWorkerParams:
    """
    Parameters for a chunk of nuScenes scenes to be processed by NuscenesRecordsGenerator.

    Attributes:
      database_root_path: Root path of the nuScenes database.
      nuscenes_version: Version of the nuScenes database.
      scenario_data: Scenario data of the scenes assigned to this worker.
      lidar_pointcloud_num_features: Number of features in the lidar pointcloud.
      ignore_label_index: Label index to use for ignored labels.
      box3d_pipelines: List of box 3D pipelines to process the box 3D annotations.
    """

    database_root_path: str
    nuscenes_version: str
    scenario_data: Sequence[ScenarioData]
    lidar_pointcloud_num_features: int
    ignore_label_index: int
    box3d_pipelines: Sequence[Box3DPipeline]


def _apply_nuscenes_records_generator(
    worker_params: NuscenesRecordsGeneratorWorkerParams,
) -> Sequence[DatasetRecord]:
    """
    Submit nuScenes records generator to the worker pool for a worker to process.

    Args:
      worker_params: nuScenes records generator worker parameters.

    Returns:
      Sequence[DatasetRecord]: Sequence of dataset records.
    """

    nuscenes_records_generator = NuscenesRecordsGenerator(
        database_root_path=worker_params.database_root_path,
        version=worker_params.nuscenes_version,
        scenario_data=worker_params.scenario_data,
        lidar_pointcloud_num_features=worker_params.lidar_pointcloud_num_features,
        ignore_label_index=worker_params.ignore_label_index,
        box3d_pipelines=worker_params.box3d_pipelines,
    )
    return nuscenes_records_generator.generate_dataset_records()


class NuscenesRecordsWriter:
    """Write the dataset record table of a nuScenes version.

    The table is a plain parquet file named by the caller, generated once and shared by
    every model that trains on it. Nothing about the training configuration enters the
    file name, and the records carry their own database name and split.
    """

    def __init__(
        self,
        nuscenes_version: str,
        root_path: str,
        scenarios: MappingProxyType[str, NuscenesScenarios],
        out_file: str,
        num_workers: int,
        class_names: Sequence[str],
        ignore_label_index: int,
        label_remapper: MappingProxyType[str, str] | None,
        lidar_pointcloud_num_features: int,
        box3d_pipelines: Sequence[Box3DPipeline],
    ) -> None:
        """
        Initialize the nuScenes records writer.

        Args:
          nuscenes_version: Version of the nuScenes devkit tables, for example v1.0-trainval.
          root_path: Root path where the actual annotation files are stored.
          scenarios: Scenario configurations for each scenario group.
          out_file: Parquet file to write.
          num_workers: Number of workers to use. Every worker loads its own devkit instance,
            so keep the count moderate.
          class_names: List of class names, used for category mapping.
          ignore_label_index: Index to use for ignored labels.
          label_remapper: Mapping to remap label names, if needed.
          lidar_pointcloud_num_features: Number of features in the lidar pointcloud.
          box3d_pipelines: List of box 3D pipelines to process the box 3D annotations.
        """
        self._nuscenes_version = nuscenes_version
        self._root_path = Path(root_path)
        self._scenarios = scenarios
        self._out_file = Path(out_file)
        self._num_workers = num_workers
        self._class_names = class_names
        self._ignore_label_index = ignore_label_index
        self._label_remapper = label_remapper
        self._lidar_pointcloud_num_features = lidar_pointcloud_num_features
        self._box3d_pipelines = box3d_pipelines

        for scenarios_obj in scenarios.values():
            for dataset_param in scenarios_obj.dataset_params:
                if dataset_param.dataset_name != nuscenes_version:
                    raise ValueError(
                        f"Scenario dataset name {dataset_param.dataset_name} does not match "
                        f"the nuScenes version {nuscenes_version}."
                    )

    def __str__(self) -> str:
        """String representation used in logs."""
        return (
            f"NuscenesRecordsWriter(nuscenes_version={self._nuscenes_version}, "
            f"root_path={self._root_path}, out_file={self._out_file}, "
            f"class_names={self._class_names}, label_remapper={self._label_remapper}, "
            f"ignore_label_index={self._ignore_label_index}, "
            f"box3d_pipelines=[{', '.join([str(pipeline) for pipeline in self._box3d_pipelines])}])"
        )

    def unique_scenario_data(self) -> Sequence[ScenarioData]:
        """Scenario data of every scene, one entry per scene."""
        unique: dict[str, ScenarioData] = {}
        for scenarios_obj in self._scenarios.values():
            for split_scenarios in scenarios_obj.scenario_data.values():
                for scenario in split_scenarios:
                    unique[scenario.scenario_id] = scenario
        return list(unique.values())

    def write(self) -> Path:
        """Generate the records and write them to the configured parquet file.

        Returns:
          Path: The written file.
        """
        start_time = time.perf_counter()
        scenario_data = self.unique_scenario_data()
        logger.info(f"Processing a total of {len(scenario_data)} scenes in nuScenes")

        records = self._run_nuscenes_records_generator(scenario_data)
        logger.info(f"Processed {len(records)} records")

        self._out_file.parent.mkdir(parents=True, exist_ok=True)
        frame = pl.DataFrame(
            [record.to_dictionary() for record in records],
            schema=DatasetTableSchema.to_polars_schema(),
        )
        frame.write_parquet(self._out_file)
        elapsed = time.perf_counter() - start_time
        logger.info(f"Wrote {self._out_file} in {elapsed:.4f} seconds")
        return self._out_file

    def _run_nuscenes_records_generator(
        self, scenario_data: Sequence[ScenarioData]
    ) -> Sequence[DatasetRecord]:
        """
        Generate the records of every scene, chunking the scenes across workers. Every worker
        loads one devkit instance and processes its chunk of scenes.

        Args:
          scenario_data: Scenario data of every scene.

        Returns:
          Sequence[DatasetRecord]: Sequence of dataset records.
        """
        scenario_list = list(scenario_data)
        num_chunks = min(self._num_workers, len(scenario_list)) if scenario_list else 0
        if num_chunks == 0:
            return []

        scenario_chunks = [list(chunk) for chunk in np.array_split(scenario_list, num_chunks)]
        worker_params = [
            NuscenesRecordsGeneratorWorkerParams(
                database_root_path=str(self._root_path),
                nuscenes_version=self._nuscenes_version,
                scenario_data=chunk,
                lidar_pointcloud_num_features=self._lidar_pointcloud_num_features,
                ignore_label_index=self._ignore_label_index,
                box3d_pipelines=self._box3d_pipelines,
            )
            for chunk in scenario_chunks
        ]

        flatten_records: list[DatasetRecord] = []
        if self._num_workers > 1:
            with ProcessPoolExecutor(max_workers=self._num_workers) as executor:
                futures = executor.map(_apply_nuscenes_records_generator, worker_params)
                for result in tqdm(futures, total=len(worker_params)):
                    flatten_records.extend(result)
            return flatten_records
        for worker_param in tqdm(worker_params, total=len(worker_params)):
            flatten_records.extend(_apply_nuscenes_records_generator(worker_param))
        return flatten_records

