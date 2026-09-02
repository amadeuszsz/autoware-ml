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

"""Tests for the label vocabulary and the label taxonomies."""

from __future__ import annotations

import pytest

from autoware_ml.databases.taxonomy import DatabaseTaxonomy, LabelTaxonomy, LabelVocabulary

VOCABULARY = LabelVocabulary(
    {
        "car": "car",
        "vehicle.car": "car",
        "police_car": "emergency_vehicle",
        "truck": "truck",
        "trailer": "trailer",
        "semi_trailer": "trailer",
    }
)


def _online() -> LabelTaxonomy:
    return LabelTaxonomy(
        vocabulary=VOCABULARY,
        class_names=["car", "truck"],
        coarsening={"car": "car", "emergency_vehicle": "car", "truck": "truck", "trailer": None},
        ignore_index=-1,
    )


def _offline() -> LabelTaxonomy:
    return LabelTaxonomy(
        vocabulary=VOCABULARY,
        class_names=["car", "emergency_vehicle", "truck", "trailer"],
        coarsening={name: name for name in VOCABULARY.fine_names},
        ignore_index=-1,
    )


def test_vocabulary_resolves_fine_names_and_keeps_unknown_raw_names() -> None:
    assert VOCABULARY.fine_names == ("car", "emergency_vehicle", "trailer", "truck")
    assert VOCABULARY.fine_name("police_car") == "emergency_vehicle"
    assert VOCABULARY.fine_name("drainage") == "drainage"


def test_vocabulary_string_form_is_canonical() -> None:
    reordered = LabelVocabulary(dict(reversed(list(VOCABULARY.name_mapping.items()))))

    assert str(reordered) == str(VOCABULARY)
    assert reordered == VOCABULARY


def test_vocabulary_rejects_empty_and_null_entries() -> None:
    with pytest.raises(ValueError, match="at least one raw label name"):
        LabelVocabulary({})
    with pytest.raises(ValueError, match="Fine label names must be non-empty strings"):
        LabelVocabulary({"background": None})


def test_online_level_coarsens_fine_names_and_drops_the_rest() -> None:
    taxonomy = _online()

    assert taxonomy.class_names == ("car", "truck")
    assert taxonomy.num_classes == 2
    assert taxonomy.resolve_index("police_car") == 0
    assert taxonomy.resolve_index("semi_trailer") == -1
    assert taxonomy.resolve_index("drainage") == -1
    assert taxonomy.class_name("trailer") is None
    assert taxonomy.class_index("truck") == 1


def test_offline_level_trains_every_fine_name() -> None:
    taxonomy = _offline()

    assert taxonomy.resolve_index("police_car") == 1
    assert taxonomy.resolve_index("semi_trailer") == 3


def test_levels_differ_in_their_string_form() -> None:
    assert _online() != _offline()
    assert str(_online()) == str(_online())
    assert "coarsening=(car: car, emergency_vehicle: car" in str(_online())


def test_taxonomy_rejects_inconsistent_definitions() -> None:
    with pytest.raises(ValueError, match="at least one class name"):
        LabelTaxonomy(VOCABULARY, [], {}, -1)
    with pytest.raises(ValueError, match="unique"):
        LabelTaxonomy(VOCABULARY, ["car", "car"], {}, -1)
    with pytest.raises(ValueError, match="ignore index 1 collides"):
        LabelTaxonomy(VOCABULARY, ["car", "truck"], {}, 1)
    with pytest.raises(ValueError, match="missing \\['trailer'\\]"):
        LabelTaxonomy(
            VOCABULARY,
            ["car", "truck"],
            {"car": "car", "emergency_vehicle": "car", "truck": "truck"},
            -1,
        )
    with pytest.raises(ValueError, match="unknown \\['bus'\\]"):
        LabelTaxonomy(
            VOCABULARY,
            ["car", "truck"],
            {
                "car": "car",
                "emergency_vehicle": "car",
                "truck": "truck",
                "trailer": None,
                "bus": None,
            },
            -1,
        )
    with pytest.raises(ValueError, match="not a class of the level"):
        LabelTaxonomy(
            VOCABULARY,
            ["car", "truck"],
            {"car": "car", "emergency_vehicle": "car", "truck": "truck", "trailer": "bus"},
            -1,
        )


def test_a_class_without_fine_labels_is_a_placeholder() -> None:
    taxonomy = LabelTaxonomy(
        VOCABULARY,
        ["car", "truck", "vertical_thin"],
        {"car": "car", "emergency_vehicle": "car", "truck": "truck", "trailer": None},
        -1,
    )

    assert taxonomy.num_classes == 3
    assert taxonomy.class_index("vertical_thin") == -1


def test_database_taxonomy_compares_both_levels() -> None:
    first = DatabaseTaxonomy(detection3d=_online(), segmentation3d=_offline())
    second = DatabaseTaxonomy(detection3d=_online(), segmentation3d=_offline())
    other = DatabaseTaxonomy(detection3d=_offline(), segmentation3d=_offline())

    assert first == second
    assert first != other
    assert str(first).startswith("DatabaseTaxonomy(detection3d=LabelTaxonomy(")
