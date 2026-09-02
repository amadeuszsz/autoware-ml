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

"""Split assignment of dataset records from the scenario lists of a database."""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping

import polars as pl

from autoware_ml.databases.scenarios import Scenarios
from autoware_ml.databases.schemas.dataset_schemas import DatasetTableSchema
from autoware_ml.types.dataset import SplitType


class ScenarioSplitter:
    """
    Split the records of a database into train, val and test by the scenario lists of the
    database. A scenario listed in two splits would put the same frames on both sides of a
    split boundary, so it is rejected.
    """

    def __str__(self) -> str:
        """
        String representation of the splitter.

        Returns:
          str: String representation of the splitter.
        """

        return f"{self.__class__.__name__}()"

    def split_by_polars_dataframe(
        self,
        dataset_records_dataframe: pl.DataFrame,
        scenarios: Mapping[str, Scenarios],
    ) -> Mapping[SplitType, pl.DataFrame]:
        """
        Split the records into train, val and test by scenario ID.

        Args:
          dataset_records_dataframe: Records of the database.
          scenarios: Scenarios of every scenario group of the database.

        Returns:
          Mapping[SplitType, pl.DataFrame]: Records of every split declared by the scenarios.
        """

        scenario_ids_by_split: dict[SplitType, set[str]] = defaultdict(set)
        for scenarios_obj in scenarios.values():
            for split, scenario_data_list in scenarios_obj.scenario_data.items():
                scenario_ids_by_split[split].update(
                    scenario_data.scenario_id for scenario_data in scenario_data_list
                )

        split_of_scenario: dict[str, SplitType] = {}
        for split, scenario_ids in scenario_ids_by_split.items():
            for scenario_id in scenario_ids:
                if scenario_id in split_of_scenario:
                    raise ValueError(
                        f"Scenario {scenario_id} is listed in both {split_of_scenario[scenario_id]} "
                        f"and {split}."
                    )
                split_of_scenario[scenario_id] = split

        scenario_column = pl.col(DatasetTableSchema.SCENARIO_ID.name)
        return {
            split: dataset_records_dataframe.filter(scenario_column.is_in(sorted(scenario_ids)))
            for split, scenario_ids in scenario_ids_by_split.items()
        }
