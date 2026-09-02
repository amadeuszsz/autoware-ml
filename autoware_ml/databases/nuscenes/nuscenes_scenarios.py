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

from nuscenes.utils import splits
from pydantic import model_validator

from autoware_ml.databases.scenarios import DatasetParams, ScenarioData, Scenarios
from autoware_ml.types.dataset import SplitType

logger = logging.getLogger(__name__)

# Official nuScenes scene splits per devkit version. The val scenes double as the test
# split because the official test split ships without annotations.
_NUSCENES_VERSION_SPLITS: Mapping[str, Mapping[SplitType, Sequence[str]]] = {
    "v1.0-trainval": {
        SplitType.TRAIN: splits.train,
        SplitType.VAL: splits.val,
        SplitType.TEST: splits.val,
    },
    "v1.0-mini": {
        SplitType.TRAIN: splits.mini_train,
        SplitType.VAL: splits.mini_val,
        SplitType.TEST: splits.mini_val,
    },
    "v1.0-test": {
        SplitType.TEST: splits.test,
    },
}


class NuscenesScenarios(Scenarios):
    """
    NuscenesScenarios class inherits from Scenarios and defines the logic for building
    scenario data for the nuScenes dataset. One scenario corresponds to one nuScenes scene and
    the split membership follows the official devkit scene splits.
    """

    @model_validator(mode="after")
    def build_scenarios(self) -> NuscenesScenarios:
        """
        Build scenarios from the official nuScenes scene splits, and overwrite the
        scenario_data attribute.

        Returns:
          NuscenesScenarios: NuscenesScenarios class instance.
        """

        scenario_data: dict[SplitType, list[ScenarioData]] = defaultdict(list)
        for dataset_params in self.dataset_params:
            version_splits = self._resolve_version_splits(dataset_params)
            for split, scene_names in version_splits.items():
                scenario_data[split] += [
                    self._build_scenario_data(scene_name, dataset_params)
                    for scene_name in scene_names
                ]

        object.__setattr__(self, "scenario_data", dict(scenario_data))
        for split, scenarios in scenario_data.items():
            logger.info(f"Loaded total of {len(scenarios)} scenarios for split {split}")
        return self

    @staticmethod
    def _resolve_version_splits(
        dataset_params: DatasetParams,
    ) -> Mapping[SplitType, Sequence[str]]:
        """
        Resolve the official scene splits for a nuScenes version.

        Args:
          dataset_params: Dataset parameters whose dataset_name is the nuScenes version.

        Returns:
          Mapping[SplitType, Sequence[str]]: Dictionary of SplitType to scene names.
        """

        if dataset_params.dataset_name not in _NUSCENES_VERSION_SPLITS:
            raise ValueError(
                f"Unsupported nuScenes version: {dataset_params.dataset_name}. "
                f"Available: {list(_NUSCENES_VERSION_SPLITS)}"
            )
        return _NUSCENES_VERSION_SPLITS[dataset_params.dataset_name]

    @staticmethod
    def _build_scenario_data(scene_name: str, dataset_params: DatasetParams) -> ScenarioData:
        """
        Build scenario data from a nuScenes scene name and dataset parameters.

        Args:
          scene_name: Name of the nuScenes scene.
          dataset_params: Dataset parameters.

        Returns:
          ScenarioData: Scenario data.
        """

        return ScenarioData.from_dataset_params(
            dataset_params,
            scenario_id=scene_name,
            scenario_version=dataset_params.dataset_name,
            vehicle_type=None,
            location=None,
        )
