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

"""Scenario metadata of a database.

A database is described by immutable scenario objects: the parameters of every dataset it
holds, the scenarios of every split, and the group a scenario list belongs to. Every object
has a deterministic string form, which is what the database hash is built from, so any
change here selects a different record table.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Mapping, Sequence

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

from autoware_ml.types.dataset import SplitType


def path_adapter(path: str | Path) -> Path:
    """
    Adapter for pathlib. If the path is a string, convert it
    to a Path object.

    Args:
      path: Path to be adapted, can be a string or a Path object.

    Returns:
      Path: Adapted path.
    """

    if isinstance(path, str):
        return Path(path)
    return path


PathAdapter = Annotated[Path, BeforeValidator(path_adapter)]


class DatasetParams(BaseModel):
    """
    Parameters of one dataset, applied to every scenario listed for it.

    Attributes:
      dataset_name: Name of the dataset, the directory below the database root and the stem
        of its scenario list.
      max_sweeps: Maximum number of preceding lidar frames stored per sample.
      sample_steps: Keep every n-th sample of a scenario.
      lidar_pointcloud_num_features: Number of float32 features per point in the point
        cloud files of the dataset.
      semantic_masks: Whether only the samples carrying a semantic segmentation mask are
        kept, so a corpus labelled at a lower rate than it was recorded trains on its
        labelled samples only.
    """

    model_config = ConfigDict(frozen=True, strict=True)

    dataset_name: str
    max_sweeps: int = Field(ge=0)
    sample_steps: int = Field(ge=1)
    lidar_pointcloud_num_features: int = Field(ge=3)
    semantic_masks: bool = False

    def __str__(self) -> str:
        """String representation of the dataset parameters."""
        return (
            f"DatasetParams(dataset_name={self.dataset_name}, "
            f"max_sweeps={self.max_sweeps}, "
            f"sample_steps={self.sample_steps}, "
            f"lidar_pointcloud_num_features={self.lidar_pointcloud_num_features}, "
            f"semantic_masks={self.semantic_masks})"
        )

    def __eq__(self, other: object) -> bool:
        """Compare two dataset parameter sets by their string form."""
        return isinstance(other, DatasetParams) and str(self) == str(other)

    def __hash__(self) -> int:
        """Hash the dataset parameters by their string form."""
        return hash(str(self))


class ScenarioData(BaseModel):
    """
    Class to store the scenario data for a single scenario.
    Note that one ScenarioData object can have multiple samples/frames
    in the scenario.

    Attributes:
      dataset_name: Name of the dataset.
      scenario_id: ID of the scenario.
      scenario_version: Version of the scenario.
      max_sweeps: Maximum number of sweeps to include.
      sample_steps: Number of steps to sample.
      lidar_pointcloud_num_features: Number of float32 features per point.
      semantic_masks: Whether only the samples carrying a semantic mask are kept.
      vehicle_type: Type of the vehicle.
      location: Location of the scenario.
    """

    model_config = ConfigDict(frozen=True, strict=True)

    dataset_name: str
    scenario_id: str
    scenario_version: str
    max_sweeps: int
    sample_steps: int
    lidar_pointcloud_num_features: int
    semantic_masks: bool
    vehicle_type: str | None = None
    location: str | None = None

    @classmethod
    def from_dataset_params(
        cls,
        dataset_params: DatasetParams,
        scenario_id: str,
        scenario_version: str,
        vehicle_type: str | None,
        location: str | None,
    ) -> ScenarioData:
        """
        Build the scenario data of one scenario from the parameters of its dataset.

        Args:
          dataset_params: Parameters of the dataset the scenario belongs to.
          scenario_id: ID of the scenario.
          scenario_version: Version of the scenario.
          vehicle_type: Type of the vehicle, or None when the scenario list does not say.
          location: Location of the scenario, or None when the scenario list does not say.

        Returns:
          ScenarioData: Scenario data.
        """

        return cls(
            dataset_name=dataset_params.dataset_name,
            scenario_id=scenario_id,
            scenario_version=scenario_version,
            max_sweeps=dataset_params.max_sweeps,
            sample_steps=dataset_params.sample_steps,
            lidar_pointcloud_num_features=dataset_params.lidar_pointcloud_num_features,
            semantic_masks=dataset_params.semantic_masks,
            vehicle_type=vehicle_type,
            location=location,
        )

    def __str__(self) -> str:
        """
        String representation of the scenario data.

        Returns:
          str: String representation of the scenario data.
        """

        return (
            f"ScenarioData(dataset_name={self.dataset_name}, "
            f"scenario_id={self.scenario_id}, "
            f"scenario_version={self.scenario_version}, "
            f"max_sweeps={self.max_sweeps}, "
            f"sample_steps={self.sample_steps}, "
            f"lidar_pointcloud_num_features={self.lidar_pointcloud_num_features}, "
            f"semantic_masks={self.semantic_masks}, "
            f"vehicle_type={self.vehicle_type}, "
            f"location={self.location})"
        )

    def __eq__(self, other: object) -> bool:
        """Compare two scenario data by their string form."""
        return isinstance(other, ScenarioData) and str(self) == str(other)

    def __hash__(self) -> int:
        """Hash the scenario data by its string form."""
        return hash(str(self))


class Scenarios(BaseModel):
    """
    Scenario datasets class. This class is used to store the scenario data for a dataset.

    Attributes:
      scenario_root_path: Root path where the scenario yaml files are stored.
      dataset_params: Parameters for the dataset.
      scenario_data: Dictionary of split type to a list of ScenarioData.
    """

    model_config = ConfigDict(frozen=True, strict=True)

    scenario_root_path: PathAdapter
    dataset_params: Sequence[DatasetParams]
    scenario_data: Mapping[SplitType, Sequence[ScenarioData]] | None = None

    def __str__(self) -> str:
        """
        String representation of the scenarios.

        Returns:
          str: String representation of the scenarios.
        """

        dataset_params = ", ".join(str(dataset_param) for dataset_param in self.dataset_params)
        scenario_data = ", ".join(
            f"{split}: [{', '.join(str(scenario) for scenario in scenarios)}]"
            for split, scenarios in self.scenario_data.items()
        )
        return (
            f"{self.__class__.__name__}(scenario_root_path={self.scenario_root_path}, "
            f"dataset_params=[{dataset_params}], "
            f"scenario_data=({scenario_data}))"
        )

    def __eq__(self, other: object) -> bool:
        """Compare two scenario sets by their string form."""
        return isinstance(other, Scenarios) and str(self) == str(other)

    def __hash__(self) -> int:
        """Hash the scenarios by their string form."""
        return hash(str(self))

    @model_validator(mode="after")
    def build_scenarios(self) -> Scenarios:
        """
        Definition of the logic to build Scenarios for a dataset.

        Returns:
          Scenarios: Scenarios class instance.
        """

        raise NotImplementedError("Subclasses must implement build_scenarios()!")

    def get_all_scenario_data(self) -> Sequence[ScenarioData]:
        """
        Get all scenario data from all splits.

        Returns:
          Sequence[ScenarioData]: Sequence of scenario data.
        """

        return [scenario_data for split in self.scenario_data.values() for scenario_data in split]
