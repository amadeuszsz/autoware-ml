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
from collections import defaultdict
from typing import Mapping, Sequence

import yaml
from pydantic import model_validator

from autoware_ml.databases.scenarios import DatasetParams, ScenarioData, Scenarios
from autoware_ml.types.dataset import SplitType

logger = logging.getLogger(__name__)

_SPLITS = (SplitType.TRAIN, SplitType.VAL, SplitType.TEST)


class T4Scenarios(Scenarios):
    """
    T4Scenarios class inherits from Scenarios and defines the logic for building
    scenario data for T4dataset. Every dataset is described by one yaml file below the
    scenario root, named after the dataset, holding the scenario entries of every split.
    """

    @model_validator(mode="after")
    def build_scenarios(self) -> T4Scenarios:
        """
        Build scenarios from the scenario lists of every dataset, and
        overwrite the scenario_data attribute.

        Returns:
          T4Scenarios: T4Scenarios class instance.
        """

        scenario_data: dict[SplitType, list[ScenarioData]] = defaultdict(list)
        for dataset_params in self.dataset_params:
            db_yaml_path = self.scenario_root_path / f"{dataset_params.dataset_name}.yaml"
            if not db_yaml_path.is_file():
                raise FileNotFoundError(f"Scenario list {db_yaml_path} does not exist.")
            logger.info(f"Loading scenario list {db_yaml_path}")
            with open(db_yaml_path, "r") as file:
                db_scenarios = yaml.safe_load(file)
            if not isinstance(db_scenarios, Mapping):
                raise ValueError(f"Scenario list {db_yaml_path} must hold a mapping of splits.")

            for split, scenarios in self._build_scenario_splits(
                db_scenarios, dataset_params
            ).items():
                scenario_data[split] += scenarios

        object.__setattr__(self, "scenario_data", dict(scenario_data))
        for split, scenarios in scenario_data.items():
            logger.info(f"Loaded total of {len(scenarios)} scenarios for split {split}")
        return self

    @staticmethod
    def _build_scenario_data(scenario_entry: str, dataset_params: DatasetParams) -> ScenarioData:
        """
        Build scenario data from one scenario list entry.

        An entry is either <scenario_id>/<version> or
        <scenario_id>/<version>/<location>/<vehicle_type>/<status>.

        Args:
          scenario_entry: Scenario list entry.
          dataset_params: Parameters of the dataset the entry belongs to.

        Returns:
          ScenarioData: Scenario data.
        """

        parts = scenario_entry.split("/")
        if len(parts) == 5:
            scenario_id, version, location, vehicle_type, _ = parts
        elif len(parts) == 2:
            scenario_id, version = parts
            location = vehicle_type = None
        else:
            raise ValueError(
                f"Invalid scenario entry {scenario_entry!r} of dataset "
                f"{dataset_params.dataset_name}, expected <scenario_id>/<version> or "
                "<scenario_id>/<version>/<location>/<vehicle_type>/<status>."
            )

        return ScenarioData.from_dataset_params(
            dataset_params,
            scenario_id=scenario_id,
            scenario_version=version,
            vehicle_type=vehicle_type,
            location=location,
        )

    def _build_scenario_splits(
        self, db_scenarios: Mapping[str, Sequence[str]], dataset_params: DatasetParams
    ) -> Mapping[SplitType, Sequence[ScenarioData]]:
        """
        Build the scenario data of every split from one scenario list.

        Args:
          db_scenarios: Dictionary of split name to scenario entries.
          dataset_params: Parameters of the dataset the list belongs to.

        Returns:
          Mapping[SplitType, Sequence[ScenarioData]]: Dictionary of split to scenario data.
        """

        scenario_splits: dict[SplitType, list[ScenarioData]] = {}
        for split in _SPLITS:
            entries = db_scenarios.get(split.value, [])
            if entries is None:
                entries = []
            if not isinstance(entries, Sequence) or isinstance(entries, str):
                raise ValueError(
                    f"Split {split.value} of dataset {dataset_params.dataset_name} must be a list "
                    f"of scenario entries, got {type(entries).__name__}."
                )
            scenario_splits[split] = [
                self._build_scenario_data(scenario_entry=str(entry), dataset_params=dataset_params)
                for entry in entries
            ]
        return scenario_splits
