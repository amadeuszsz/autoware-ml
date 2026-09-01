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

"""Record table fixtures for datamodule unit tests.

The tests read the same parquet a generated table is, so the fixtures write one to a
temporary directory instead of faking the read path.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Sequence

import polars as pl

from autoware_ml.databases.record_table import RecordTable
from autoware_ml.databases.schemas.box3d_schemas import Box3DDataModel
from autoware_ml.databases.schemas.dataset_schemas import DatasetRecord, DatasetTableSchema
from autoware_ml.datamodule.sources import DatasetSource
from autoware_ml.testing.factories import make_record


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


def write_record_table(
    tmp_path: Path,
    records: Sequence[DatasetRecord] = (),
    *,
    name: str = "records.parquet",
    databases: Sequence[str] = (),
) -> RecordTable:
    """
    Write the records to a parquet file and return the table reading it.

    Args:
      tmp_path: Directory receiving the parquet file and serving as the data root.
      records: Dataset records of the table.
      name: File name of the table.
      databases: Databases the table selects, or empty for every database.

    Returns:
      RecordTable: Table over the written file.
    """

    path = tmp_path / name
    records_dataframe(records).write_parquet(path)
    return RecordTable(path=str(path), data_root=str(tmp_path), databases=tuple(databases))


def make_stored_record(
    *,
    scenario_id: str = "scene-0",
    sample_id: str = "sample-0",
    sample_index: int = 0,
    database: str = "db-0",
    split: str = "train",
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
      database: Database the record belongs to.
      split: Split the record belongs to.
      boxes_3d: Box annotations of the record.
      camera_frames: Camera frames of the record.

    Returns:
      DatasetRecord: The dataset record.
    """

    record = make_record(
        scenario_id=scenario_id,
        sample_id=sample_id,
        database=database,
        split=split,
        boxes_3d=boxes_3d,
        camera_frames=camera_frames,
        category_names=("car",),
        category_indices=(0,),
    )
    return record.model_copy(update={"sample_index": sample_index})


def make_source(
    tmp_path: Path | None = None, records: Sequence[DatasetRecord] = (), **source_kwargs
):
    """
    Build a dataset source backed by a written record table.

    Args:
      tmp_path: Directory receiving the parquet file, or None for a temporary directory.
      records: Dataset records of the table.
      source_kwargs: DatasetSource fields except the record table.

    Returns:
      DatasetSource: The dataset source.
    """

    root = Path(tempfile.mkdtemp()) if tmp_path is None else tmp_path
    return DatasetSource(records=write_record_table(root, records), **source_kwargs)
