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

"""Config level checks of the T4dataset taxonomy levels."""

import pytest
from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate

from autoware_ml.configs.resolvers import register_config_resolvers
from autoware_ml.databases.taxonomy import DatabaseTaxonomy

T4_DETECTION_CONFIG = "tasks/detection3d/ptv3/voxel012_122m_t4dataset_j6gen2"
LEVELS = ("online", "offline")


def _taxonomy(level: str) -> DatabaseTaxonomy:
    register_config_resolvers()
    GlobalHydra.instance().clear()
    with initialize_config_module(version_base=None, config_module="autoware_ml.configs"):
        cfg = compose(
            config_name=T4_DETECTION_CONFIG,
            overrides=[f"database/t4dataset/taxonomy@database.taxonomy={level}"],
        )
    return instantiate(cfg.database.taxonomy)


@pytest.mark.parametrize("level", LEVELS)
def test_every_level_is_a_strict_coarsening_of_the_vocabularies(level: str) -> None:
    # Instantiation validates that every fine label folds onto one class or null and that
    # the class keyed tables name exactly the classes of the level.
    taxonomy = _taxonomy(level)
    assert taxonomy.detection3d.class_names
    assert taxonomy.segmentation3d.class_names[: taxonomy.detection3d.num_classes] == (
        taxonomy.detection3d.class_names
    )


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("raw_name", ["static_object.bicycle_rack", "static_object.bicycle rack"])
def test_bicycle_rack_spellings_share_one_fine_label(level: str, raw_name: str) -> None:
    # Both spellings the T4 corpora use for bicycle racks must reach the same fine label.
    taxonomy = _taxonomy(level)
    assert taxonomy.detection3d.fine_name(raw_name) == "bicycle_rack"
