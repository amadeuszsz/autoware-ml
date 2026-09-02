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

"""Database fixtures for datamodule unit tests.

The fixtures drive the real database machinery. A fake database writes records built in
memory to its hashed record table and reads them back like a generated one, and its scenario
lists are yaml files parsed by T4Scenarios, so the tests exercise the splitter and the cache
path instead of faking them.
"""

from __future__ import annotations

import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import polars as pl
import yaml

from autoware_ml.databases.base_database import BaseDatabase
from autoware_ml.databases.scenarios import DatasetParams, ScenarioData
from autoware_ml.databases.schemas.box3d_schemas import Box3DDataModel
from autoware_ml.databases.schemas.dataset_schemas import DatasetRecord, DatasetTableSchema
from autoware_ml.databases.t4dataset.t4scenarios import T4Scenarios
from autoware_ml.datamodule.sources import DatasetSource
from autoware_ml.testing.factories import make_record

FAKE_DATASET_NAME = "db-0"
SCENARIO_VERSION = "0"


class FakeDatabase(BaseDatabase):
    """Database serving records built in memory through the real cache path."""

    def __init__(
        self,
        records: Sequence[DatasetRecord],
        scenarios: Mapping[str, T4Scenarios],
        root_path: Path,
        cache_path: Path,
        version: str = "fake-v1",
        class_names: Sequence[str] = ("car",),
    ) -> None:
        """
        Initialize the fake database.

        Args:
          records: Records the database generates.
          scenarios: Scenarios of every scenario group.
          root_path: Root directory the record paths resolve against.
          cache_path: Directory the record table is written to.
          version: Version of the database.
          class_names: Class names of the database taxonomy.
        """

        self._records = list(records)
        self.generate_calls = 0
        super().__init__(
            version=version,
            root_path=str(root_path),
            scenarios=scenarios,
            cache_path=str(cache_path),
            cache_file_prefix_name="database",
            num_workers=1,
            class_names=list(class_names),
            label_remapper={name: name for name in class_names},
            ignore_label_index=-1,
            box3d_pipelines=[],
        )

    def generate_records(
        self, scenario_data: Mapping[str, ScenarioData]
    ) -> Sequence[DatasetRecord]:
        """Return the in memory records and count the call."""
        self.generate_calls += 1
        return self._records


def write_scenario_list(
    directory: Path,
    splits: Mapping[str, Sequence[str]],
    dataset_name: str = FAKE_DATASET_NAME,
) -> Path:
    """
    Write a scenario list in the perception-devops yaml form.

    Args:
      directory: Directory receiving the yaml file.
      splits: Split name to scenario IDs.
      dataset_name: Name of the dataset, the stem of the yaml file.

    Returns:
      Path: The written yaml file.
    """

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{dataset_name}.yaml"
    entries = {
        split: [f"{scenario_id}/{SCENARIO_VERSION}" for scenario_id in scenario_ids]
        for split, scenario_ids in splits.items()
    }
    path.write_text(yaml.safe_dump(entries), encoding="utf-8")
    return path


def make_scenarios(
    directory: Path,
    splits: Mapping[str, Sequence[str]],
    dataset_name: str = FAKE_DATASET_NAME,
) -> T4Scenarios:
    """
    Build T4 scenarios over a scenario list written to the directory.

    Args:
      directory: Directory receiving the scenario list.
      splits: Split name to scenario IDs.
      dataset_name: Name of the dataset.

    Returns:
      T4Scenarios: The scenarios.
    """

    write_scenario_list(directory, splits, dataset_name)
    return T4Scenarios(
        scenario_root_path=directory,
        dataset_params=[
            DatasetParams(
                dataset_name=dataset_name,
                max_sweeps=0,
                sample_steps=1,
                lidar_pointcloud_num_features=5,
            )
        ],
    )


def make_database(
    tmp_path: Path,
    records: Sequence[DatasetRecord] = (),
    splits: Mapping[str, Sequence[str]] | None = None,
    version: str = "fake-v1",
    class_names: Sequence[str] = ("car",),
) -> FakeDatabase:
    """
    Build a fake database whose record table is not generated yet.

    Args:
      tmp_path: Directory receiving the scenario lists and the record table, and serving
        as the data root.
      records: Records the database generates.
      splits: Split name to scenario IDs, every scenario of the records in train when omitted.
      version: Version of the database, also separating its scenario lists from those of
        another fake database in the same directory.
      class_names: Class names of the database taxonomy.

    Returns:
      FakeDatabase: The database.
    """

    records = list(records)
    if splits is None:
        splits = {"train": sorted({record.scenario_id for record in records})}
    scenarios = {"group": make_scenarios(tmp_path / "scenarios" / version, splits)}
    return FakeDatabase(
        records=records,
        scenarios=scenarios,
        root_path=tmp_path,
        cache_path=tmp_path / "cache",
        version=version,
        class_names=class_names,
    )


def records_dataframe(records: Sequence[DatasetRecord]) -> pl.DataFrame:
    """
    Build the polars dataframe a record table holds for the given records.

    Args:
      records: Dataset records of the dataframe.

    Returns:
      pl.DataFrame: Dataframe following the dataset table schema.
    """

    return pl.DataFrame(
        [record.to_dictionary() for record in records],
        schema=DatasetTableSchema.to_polars_schema(),
    )


def make_stored_record(
    *,
    scenario_id: str = "scene-0",
    sample_id: str = "sample-0",
    sample_index: int = 0,
    boxes_3d: Sequence[Box3DDataModel] | None = None,
    camera_frames=None,
) -> DatasetRecord:
    """
    Build a record that survives the polars round trip. The record always carries a
    category mapping because a record without one cannot be loaded back from a table row.

    Args:
      scenario_id: Scenario ID of the record.
      sample_id: Sample ID of the record.
      sample_index: Sample index of the record.
      boxes_3d: Box annotations of the record.
      camera_frames: Camera frames of the record.

    Returns:
      DatasetRecord: The dataset record.
    """

    record = make_record(
        scenario_id=scenario_id,
        sample_id=sample_id,
        boxes_3d=boxes_3d,
        camera_frames=camera_frames,
        category_names=("car",),
        category_indices=(0,),
    )
    return record.model_copy(update={"sample_index": sample_index})


def make_source(
    tmp_path: Path | None = None, records: Sequence[DatasetRecord] = (), **source_kwargs
) -> DatasetSource:
    """
    Build a dataset source backed by a fake database.

    Args:
      tmp_path: Directory of the database, or None for a temporary directory.
      records: Records the database generates.
      source_kwargs: DatasetSource fields except the database.

    Returns:
      DatasetSource: The dataset source.
    """

    root = Path(tempfile.mkdtemp()) if tmp_path is None else tmp_path
    return DatasetSource(database=make_database(root, records), **source_kwargs)
