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

"""Protocol of a dataset database.

A database turns the raw annotations of the scenarios it lists into a record table, a
parquet file named after the hash of the database definition, and reads that table back.
Training and the generation entrypoint only depend on this protocol.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Protocol, Sequence

import polars as pl

from autoware_ml.databases.scenarios import ScenarioData, Scenarios
from autoware_ml.databases.schemas.dataset_schemas import DatasetRecord


class DatabaseInterface(Protocol):
    """Protocol for database classes that defines the common interface for every dataset type."""

    def __str__(self) -> str:
        """String representation of the database, the input of the database hash."""
        ...

    def __hash__(self) -> int:
        """Hash the database by its string representation."""
        ...

    def __eq__(self, other: object) -> bool:
        """Compare two databases by their string representation."""
        ...

    @property
    def version(self) -> str:
        """Version of the database."""
        ...

    @property
    def root_path(self) -> Path:
        """Root directory the record paths resolve against."""
        ...

    @property
    def scenarios(self) -> Mapping[str, Scenarios]:
        """Scenarios of every scenario group, keyed by group name."""
        ...

    @property
    def class_names(self) -> Sequence[str]:
        """Class names the box labels are resolved against."""
        ...

    @property
    def label_remapper(self) -> Mapping[str, str]:
        """Mapping from raw dataset label names to class names."""
        ...

    @property
    def ignore_label_index(self) -> int:
        """Label index of a box whose class is not trained."""
        ...

    @property
    def database_hash(self) -> str:
        """Hash of the database definition and the table schema."""
        ...

    @property
    def cache_file_path(self) -> Path:
        """Record table file of the database."""
        ...

    def get_unique_scenario_data(self) -> Mapping[str, ScenarioData]:
        """Scenario data of every scenario across all groups, keyed by scenario ID."""
        ...

    def process_scenario_records(self) -> None:
        """Generate the record table of the database unless it exists."""
        ...

    def load_polars_scenario_dataframe(self) -> pl.DataFrame:
        """Load the record table of the database as a dataframe."""
        ...

    def load_scenario_records(self) -> Sequence[DatasetRecord]:
        """Load the record table of the database as dataset records."""
        ...
