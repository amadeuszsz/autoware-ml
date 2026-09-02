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

"""Shared implementation of a dataset database.

The base class owns everything that does not depend on the annotation format: the database
definition and its hash, the cache file the record table is written to, the parallel run of
the record generators, and reading the table back. A dataset family only implements how the
records of its scenarios are generated.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Mapping, Sequence, TypeVar

import polars as pl
from tqdm import tqdm

from autoware_ml.databases.box3d_pipelines.box3d_pipeline import Box3DPipeline
from autoware_ml.databases.scenarios import ScenarioData, Scenarios
from autoware_ml.databases.schemas.dataset_schemas import DatasetRecord, DatasetTableSchema

logger = logging.getLogger(__name__)

WorkerParams = TypeVar("WorkerParams")


def run_record_workers(
    function: Callable[[WorkerParams], Sequence[DatasetRecord]],
    worker_params: Sequence[WorkerParams],
    num_workers: int,
) -> list[DatasetRecord]:
    """
    Run a record generation function over every parameter set and flatten the results.

    Args:
      function: Module level function generating the records of one parameter set.
      worker_params: One parameter set per unit of work.
      num_workers: Number of worker processes, the calling process alone when 1.

    Returns:
      list[DatasetRecord]: Records of every parameter set, in parameter order.
    """

    records: list[DatasetRecord] = []
    if num_workers > 1:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for result in tqdm(executor.map(function, worker_params), total=len(worker_params)):
                records.extend(result)
        return records
    for params in tqdm(worker_params, total=len(worker_params)):
        records.extend(function(params))
    return records


class BaseDatabase:
    """Definition of a base database class that will be inherited by every dataset type."""

    def __init__(
        self,
        version: str,
        root_path: str,
        scenarios: Mapping[str, Scenarios],
        cache_path: str,
        cache_file_prefix_name: str,
        num_workers: int,
        class_names: Sequence[str],
        label_remapper: Mapping[str, str],
        ignore_label_index: int,
        box3d_pipelines: Sequence[Box3DPipeline],
    ) -> None:
        """
        Initialize BaseDatabase.

        Args:
          version: Version of the database.
          root_path: Root path where the actual annotation files are stored.
          scenarios: Scenario configurations of every scenario group, keyed by group name.
          cache_path: Directory the record table is written to.
          cache_file_prefix_name: Prefix of the record table file, the file is
            <cache_file_prefix_name>_<database_hash>.parquet.
          num_workers: Number of worker processes used to generate the records.
          class_names: Class names the box labels are resolved against.
          label_remapper: Mapping from raw dataset label names to class names.
          ignore_label_index: Label index of a box whose class is not trained.
          box3d_pipelines: Box pipelines applied to the box annotations of every sample.
        """

        if not len(scenarios):
            raise ValueError("A database requires at least one scenario group.")
        if not len(class_names):
            raise ValueError("A database requires at least one class name.")
        if num_workers < 1:
            raise ValueError(f"num_workers must be at least 1, got {num_workers}.")
        if not cache_file_prefix_name:
            raise ValueError("cache_file_prefix_name must not be empty.")

        self._version = version
        self._root_path = Path(root_path)
        self._scenarios = dict(scenarios)
        self._cache_path = Path(cache_path)
        self._cache_file_prefix_name = cache_file_prefix_name
        self._num_workers = num_workers
        self._class_names = list(class_names)
        self._label_remapper = dict(label_remapper)
        self._ignore_label_index = ignore_label_index
        self._box3d_pipelines = list(box3d_pipelines)
        logger.info(f"Database initialized: {self}")

    def description_fields(self) -> dict[str, str]:
        """
        Fields of the database definition in string form, in a fixed order. Subclasses extend
        the mapping with their own parameters so they take part in the database hash.

        Returns:
          dict[str, str]: Field name to string value.
        """

        scenarios = ", ".join(
            f"{group}: {scenarios}" for group, scenarios in self._scenarios.items()
        )
        return {
            "version": self._version,
            "root_path": str(self._root_path),
            "cache_path": str(self._cache_path),
            "cache_file_prefix_name": self._cache_file_prefix_name,
            "class_names": str(self._class_names),
            "label_remapper": str(self._label_remapper),
            "ignore_label_index": str(self._ignore_label_index),
            "box3d_pipelines": f"[{', '.join(str(pipeline) for pipeline in self._box3d_pipelines)}]",
            "scenarios": f"({scenarios})",
        }

    def __str__(self) -> str:
        """
        String representation of the database, the input of the database hash.

        Returns:
          str: String representation of the database.
        """

        fields = ", ".join(f"{name}={value}" for name, value in self.description_fields().items())
        return f"{self.__class__.__name__}({fields})"

    def __eq__(self, other: object) -> bool:
        """
        Compare two databases by their string representation.

        Returns:
          bool: True if the databases are equal, False otherwise.
        """

        return type(self) is type(other) and str(self) == str(other)

    def __hash__(self) -> int:
        """
        Hash the database by its string representation.

        Returns:
          int: Hash of the database.
        """

        return hash(str(self))

    @property
    def version(self) -> str:
        """Version of the database."""
        return self._version

    @property
    def root_path(self) -> Path:
        """Root directory the record paths resolve against."""
        return self._root_path

    @property
    def cache_path(self) -> Path:
        """Directory the record table is written to."""
        return self._cache_path

    @property
    def num_workers(self) -> int:
        """Number of worker processes used to generate the records."""
        return self._num_workers

    @property
    def scenarios(self) -> Mapping[str, Scenarios]:
        """Scenarios of every scenario group, keyed by group name."""
        return self._scenarios

    @property
    def class_names(self) -> Sequence[str]:
        """Class names the box labels are resolved against."""
        return self._class_names

    @property
    def label_remapper(self) -> Mapping[str, str]:
        """Mapping from raw dataset label names to class names."""
        return self._label_remapper

    @property
    def ignore_label_index(self) -> int:
        """Label index of a box whose class is not trained."""
        return self._ignore_label_index

    @property
    def box3d_pipelines(self) -> Sequence[Box3DPipeline]:
        """Box pipelines applied to the box annotations of every sample."""
        return self._box3d_pipelines

    @property
    def database_hash(self) -> str:
        """
        Hash of the database definition and the table schema. Any change to the scenarios,
        the taxonomy, the pipelines or the schema selects a different record table.

        Returns:
          str: Hex digest of the hash.
        """

        hash_input = str(self) + str(self.get_polars_schema())
        return hashlib.sha256(hash_input.encode("utf-8")).hexdigest()

    @property
    def cache_file_path(self) -> Path:
        """Record table file of the database."""
        return self._cache_path / f"{self._cache_file_prefix_name}_{self.database_hash}.parquet"

    def get_polars_schema(self) -> pl.Schema:
        """
        Get the polars schema of the record table.

        Returns:
          pl.Schema: Polars schema.
        """

        return DatasetTableSchema.to_polars_schema()

    def get_unique_scenario_data(self) -> Mapping[str, ScenarioData]:
        """
        Get all scenario data from all scenario groups and keep their order the same.

        Returns:
          Mapping[str, ScenarioData]: Dictionary of scenario ID to scenario data.
        """

        unique_scenarios: dict[str, ScenarioData] = {}
        for scenarios in self._scenarios.values():
            for scenario in scenarios.get_all_scenario_data():
                if scenario.scenario_id not in unique_scenarios:
                    unique_scenarios[scenario.scenario_id] = scenario
        return unique_scenarios

    def generate_records(
        self, scenario_data: Mapping[str, ScenarioData]
    ) -> Sequence[DatasetRecord]:
        """
        Generate the records of every scenario.

        Args:
          scenario_data: Dictionary of scenario ID to scenario data.

        Returns:
          Sequence[DatasetRecord]: Records of every scenario.
        """

        raise NotImplementedError("Subclasses must implement generate_records!")

    def process_scenario_records(self) -> None:
        """Generate the record table of the database unless it exists."""

        cache_file_path = self.cache_file_path
        if cache_file_path.exists():
            logger.info(f"Record table {cache_file_path} already exists, skipping generation")
            return

        start_time = time.perf_counter()
        unique_scenario_data = self.get_unique_scenario_data()
        logger.info(f"Processing {len(unique_scenario_data)} unique scenarios of {self._version}")
        records = self.generate_records(unique_scenario_data)
        if not len(records):
            raise ValueError(f"Database {self._version} produced no records.")

        frame = pl.DataFrame(
            [record.to_dictionary() for record in records], schema=self.get_polars_schema()
        )
        cache_file_path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(cache_file_path)
        elapsed = time.perf_counter() - start_time
        logger.info(
            f"Wrote {frame.height} records of {self._version} to {cache_file_path} "
            f"in {elapsed:.1f} seconds"
        )

    def load_polars_scenario_dataframe(self) -> pl.DataFrame:
        """
        Load the record table of the database as a dataframe.

        Returns:
          pl.DataFrame: Polars dataframe of the records.

        Raises:
          FileNotFoundError: If the record table has not been generated.
        """

        cache_file_path = self.cache_file_path
        if not cache_file_path.exists():
            raise FileNotFoundError(
                f"Record table {cache_file_path} does not exist. Run process_scenario_records() "
                f"or the generate-dataset command for database {self._version} first."
            )
        frame = pl.read_parquet(cache_file_path, schema=self.get_polars_schema())
        logger.info(f"Loaded {frame.height} records of {self._version} from {cache_file_path}")
        return frame

    def load_scenario_records(self) -> Sequence[DatasetRecord]:
        """
        Load the record table of the database as dataset records.

        Returns:
          Sequence[DatasetRecord]: Dataset records.
        """

        frame = self.load_polars_scenario_dataframe()
        return [DatasetRecord.load_from_dictionary(row) for row in frame.iter_rows(named=True)]
