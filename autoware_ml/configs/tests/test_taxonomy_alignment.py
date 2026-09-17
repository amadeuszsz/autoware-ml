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

"""The trained classes of every task are the classes of the taxonomy of its database.

The database bakes the box labels and resolves the semantic mask with its taxonomy, so a
model trained on a different class list would read labels that mean something else. These
tests hold every runnable task config against the taxonomy it is generated with.
"""

from __future__ import annotations

import glob

import hydra
import pytest
from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from autoware_ml.configs.resolvers import register_config_resolvers

TASK_CONFIGS = sorted(
    path[len("autoware_ml/configs/") : -len(".yaml")]
    for path in glob.glob("autoware_ml/configs/tasks/**/*.yaml", recursive=True)
    if path.endswith("_nuscenes.yaml") or "t4dataset" in path
)


def compose_task(config_name: str):
    """Compose one task config."""
    register_config_resolvers()
    GlobalHydra.instance().clear()
    with initialize_config_module(version_base=None, config_module="autoware_ml.configs"):
        return compose(config_name=config_name)


@pytest.mark.parametrize("config_name", TASK_CONFIGS)
def test_every_task_trains_on_the_classes_of_its_taxonomy(config_name: str) -> None:
    cfg = compose_task(config_name)
    if "dataset" not in cfg:
        pytest.skip("the task carries no class list")

    for task in ("detection3d", "segmentation3d"):
        if task not in cfg.dataset:
            continue
        class_names = OmegaConf.to_container(cfg.dataset[task].class_names, resolve=True)
        taxonomy_names = OmegaConf.to_container(
            cfg.database.taxonomy[task].class_names, resolve=True
        )
        assert class_names == taxonomy_names
        assert int(cfg.dataset[task].num_classes) == len(taxonomy_names)


@pytest.mark.parametrize("config_name", TASK_CONFIGS)
def test_every_task_instantiates_its_model_and_pipeline(config_name: str) -> None:
    cfg = compose_task(config_name)

    hydra.utils.instantiate(cfg.data_preprocessing)
    hydra.utils.instantiate(cfg.datamodule.train_dataset.transforms)
    hydra.utils.instantiate(cfg.model)
