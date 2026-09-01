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

"""Read access to a generated dataset record table.

A record table is a parquet file of dataset records produced outside this repository,
by t4dataset-generator for T4dataset. It carries its own splits and database names, so
selecting data here is a filter and nothing is derived from a scenario list, a cache
hash, or the dataset annotations.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from autoware_ml.databases.schemas.dataset_schemas import DatasetTableSchema

logger = logging.getLogger(__name__)

SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class RecordTable:
    """One parquet file of dataset records, optionally narrowed to some databases.

    Attributes:
        path: Parquet file holding the records.
        data_root: Directory the record paths resolve against.
        databases: Databases to keep, or empty to keep every database of the table.
    """

    path: str
    data_root: str
    databases: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Validate the table location and the database selection."""
        if isinstance(self.databases, str):
            # A bare string would iterate as characters and filter on them, and it is what
            # an unfilled mandatory value reaches this constructor as.
            raise TypeError(
                f"databases must be a list of database names, got the string "
                f"{self.databases!r}."
            )
        if not Path(self.path).is_file():
            raise FileNotFoundError(f"Record table {self.path} does not exist.")
        if not Path(self.data_root).is_dir():
            raise FileNotFoundError(f"Data root {self.data_root} does not exist.")

    def __str__(self) -> str:
        """String representation used in logs."""
        databases = ", ".join(self.databases) if self.databases else "every database"
        return f"RecordTable(path={self.path}, data_root={self.data_root}, {databases})"

    def scan(self) -> pl.LazyFrame:
        """Scan the table, keeping only the configured databases.

        Returns:
            pl.LazyFrame: Lazy frame over the selected records.
        """
        frame = pl.scan_parquet(self.path)
        self._require_columns(frame.collect_schema().names())
        if not self.databases:
            return frame
        self._require_databases(frame)
        return frame.filter(pl.col(DatasetTableSchema.DATABASE.name).is_in(list(self.databases)))

    def load(self, split: str) -> pl.DataFrame:
        """Load the records of one split, ordered by scenario and sample.

        Args:
            split: Split to load, train, val or test.

        Returns:
            pl.DataFrame: Records of the split.

        Raises:
            ValueError: If the split is unknown or holds no records.
        """
        if split not in SPLITS:
            raise ValueError(f"Unknown split {split!r}, expected one of {SPLITS}.")
        records = (
            self.scan()
            .filter(pl.col(DatasetTableSchema.SPLIT.name) == split)
            .sort([DatasetTableSchema.SCENARIO_ID.name, DatasetTableSchema.SAMPLE_INDEX.name])
            .collect()
        )
        if records.is_empty():
            raise ValueError(f"{self} holds no {split} records.")
        logger.info(f"Loaded {records.height} {split} records from {self.path}")
        return records

    def _require_columns(self, columns: Sequence[str]) -> None:
        """Reject a table that does not carry the dataset record columns."""
        missing = [
            name for name in DatasetTableSchema.to_polars_schema() if name not in set(columns)
        ]
        if missing:
            raise ValueError(f"Record table {self.path} is missing the columns {missing}.")

    def _require_databases(self, frame: pl.LazyFrame) -> None:
        """Reject a selection naming a database the table does not hold."""
        column = DatasetTableSchema.DATABASE.name
        available = set(frame.select(column).unique().collect()[column].to_list())
        unknown = sorted(set(self.databases) - available)
        if unknown:
            raise ValueError(
                f"Record table {self.path} holds no records of {unknown}, "
                f"it holds {sorted(available)}."
            )
