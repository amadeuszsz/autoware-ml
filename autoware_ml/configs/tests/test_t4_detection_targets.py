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
"""Config level checks of the detection targets the T4dataset configs select."""

import numpy as np
import pytest
from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig

from autoware_ml.configs.resolvers import register_config_resolvers
from autoware_ml.databases.schemas.box3d_schemas import Box3DDataModel
from autoware_ml.transforms.boxes3d.annotations import (
    normalize_filter_attributes,
    resolve_box_class,
)

CONFIGS = (
    "tasks/detection3d/ptv3/voxel012_122m_t4dataset_j6gen2",
    "tasks/multi/ptv3/voxel012_122m_t4dataset_j6gen2",
)
TWO_WHEELERS = ("bicycle", "motorcycle")
# The rider state is annotated under the current name and under the per class names the
# older corpus releases carry, so an exclusion has to name every spelling in use.
RIDER_STATES = (
    "two_wheel_vehicle_state.without_rider",
    "cycle_state.without_rider",
    "motorcycle_state.without_rider",
)
IGNORE_INDEX = -1


def _config(name: str) -> DictConfig:
    register_config_resolvers()
    GlobalHydra.instance().clear()
    with initialize_config_module(version_base=None, config_module="autoware_ml.configs"):
        return compose(config_name=name)


def _load_steps(pipeline) -> list[DictConfig]:
    return [
        step for step in pipeline if str(step.get("_target_", "")).endswith("LoadDet3DAnnotations")
    ]


def _box(class_name: str, class_names: tuple[str, ...], attributes: set[str]) -> Box3DDataModel:
    return Box3DDataModel(
        box3d_params=np.zeros(10, dtype=np.float64),
        box3d_instance_id="instance",
        box3d_dataset_label_name=class_name,
        box3d_label_name=class_name,
        box3d_label_index=class_names.index(class_name),
        box3d_num_lidar_points=32,
        box3d_num_radar_points=0,
        box3d_valid=True,
        box3d_attributes=attributes,
        box3d_coordinate="base_link",
    )


@pytest.mark.parametrize("name", CONFIGS)
def test_parked_and_rider_less_two_wheelers_are_excluded(name: str) -> None:
    # A parked or rider-less two wheeler is street furniture, not a road user, so it must
    # not enter the detection targets under any spelling the corpora use.
    exclusions = normalize_filter_attributes(_config(name).dataset.detection3d.filter_attributes)
    for class_name in TWO_WHEELERS:
        assert (class_name, "vehicle_state.parked") in exclusions
        assert [state for state in RIDER_STATES if (class_name, state) in exclusions]


@pytest.mark.parametrize("name", CONFIGS)
def test_every_split_selects_the_same_targets(name: str) -> None:
    # Training on one target set and scoring on another silently shifts every metric, so
    # the frame sampler and all pipelines read one exclusion list.
    cfg = _config(name)
    expected = normalize_filter_attributes(cfg.dataset.detection3d.filter_attributes)
    assert (
        normalize_filter_attributes(cfg.datamodule.train_frame_sampling.filter_attributes)
        == expected
    )
    for split in ("train_transforms", "test_transforms"):
        steps = _load_steps(cfg.datamodule[split].pipeline)
        assert steps, f"{split} loads no detection annotations"
        for step in steps:
            assert normalize_filter_attributes(step.filter_attributes) == expected


@pytest.mark.parametrize("class_name", TWO_WHEELERS)
def test_the_exclusions_reject_the_boxes_they_name(class_name: str) -> None:
    cfg = _config(CONFIGS[0])
    class_names = tuple(cfg.dataset.detection3d.class_names)
    exclusions = normalize_filter_attributes(cfg.dataset.detection3d.filter_attributes)
    ridden = _box(class_name, class_names, {"two_wheel_vehicle_state.with_rider"})
    assert (
        resolve_box_class(
            ridden,
            class_names=class_names,
            ignore_label_index=IGNORE_INDEX,
            filter_attributes=exclusions,
        )
        == class_name
    )
    for attributes in ({"vehicle_state.parked"}, *({state} for state in RIDER_STATES)):
        box = _box(class_name, class_names, set(attributes))
        rejected = resolve_box_class(
            box,
            class_names=class_names,
            ignore_label_index=IGNORE_INDEX,
            filter_attributes=exclusions,
        )
        if (class_name, next(iter(attributes))) in exclusions:
            assert rejected is None, f"{class_name} with {attributes} stayed a target"
