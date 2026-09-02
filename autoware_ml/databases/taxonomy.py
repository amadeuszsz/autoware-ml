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

"""Label taxonomies of a database.

A vocabulary maps the raw label names of a dataset family onto fine label names, the finest
distinction the corpus supports. A taxonomy selects the classes trained at one level of
granularity and folds every fine name onto one of them or drops it. Boxes are baked with the
fine name and the class index of the level when the record table is generated, and masks are
resolved through the same objects when they are loaded, so one definition decides the labels
of both heads.
"""

from __future__ import annotations

from typing import Mapping, Sequence


class LabelVocabulary:
    """Raw label names of a dataset family mapped onto fine label names."""

    def __init__(self, name_mapping: Mapping[str, str]) -> None:
        """
        Initialize the vocabulary.

        Args:
          name_mapping: Raw label name to fine label name. A raw label absent from the mapping
            keeps its own name, so a raw label spelled like a fine name resolves like it.
        """

        if not len(name_mapping):
            raise ValueError("A label vocabulary requires at least one raw label name.")
        for raw_name, fine_name in name_mapping.items():
            if not isinstance(raw_name, str) or not raw_name:
                raise ValueError(f"Raw label names must be non-empty strings, got {raw_name!r}.")
            if not isinstance(fine_name, str) or not fine_name:
                raise ValueError(
                    f"Fine label names must be non-empty strings, got {fine_name!r} for "
                    f"raw label {raw_name!r}."
                )
        self._name_mapping = dict(name_mapping)
        self._fine_names = tuple(sorted(set(self._name_mapping.values())))

    @property
    def fine_names(self) -> tuple[str, ...]:
        """Fine label names of the vocabulary, sorted."""
        return self._fine_names

    @property
    def name_mapping(self) -> Mapping[str, str]:
        """Raw label name to fine label name."""
        return self._name_mapping

    def fine_name(self, raw_name: str) -> str:
        """
        Fine label name of a raw label name.

        Args:
          raw_name: Raw label name of the dataset.

        Returns:
          str: The fine label name, the raw name itself when the vocabulary does not list it.
        """

        return self._name_mapping.get(raw_name, raw_name)

    def __str__(self) -> str:
        """Canonical string form, the input of the database hash."""
        entries = ", ".join(
            f"{raw_name}: {fine_name}" for raw_name, fine_name in sorted(self._name_mapping.items())
        )
        return f"{self.__class__.__name__}({entries})"

    def __eq__(self, other: object) -> bool:
        """Compare two vocabularies by their mapping."""
        return type(self) is type(other) and str(self) == str(other)

    def __hash__(self) -> int:
        """Hash the vocabulary by its string form."""
        return hash(str(self))


class LabelTaxonomy:
    """Classes trained at one level of granularity over a vocabulary."""

    def __init__(
        self,
        vocabulary: LabelVocabulary,
        class_names: Sequence[str],
        coarsening: Mapping[str, str | None],
        ignore_index: int,
    ) -> None:
        """
        Initialize the taxonomy.

        Args:
          vocabulary: Raw label names mapped onto fine label names.
          class_names: Classes of the level, in index order.
          coarsening: Fine label name to class name, null for a fine label the level drops.
            Every fine name of the vocabulary needs an entry. A class no fine label coarsens
            to is a placeholder the level trains without data.
          ignore_index: Label index of a label outside the classes of the level.
        """

        if not len(class_names):
            raise ValueError("A label taxonomy requires at least one class name.")
        if len(set(class_names)) != len(class_names):
            raise ValueError(f"Class names must be unique, got {list(class_names)}.")
        for class_name in class_names:
            if not isinstance(class_name, str) or not class_name:
                raise ValueError(f"Class names must be non-empty strings, got {class_name!r}.")
        if 0 <= ignore_index < len(class_names):
            raise ValueError(
                f"The ignore index {ignore_index} collides with the class indices "
                f"0..{len(class_names) - 1}."
            )

        fine_names = set(vocabulary.fine_names)
        coarsened_names = set(coarsening)
        if coarsened_names != fine_names:
            raise ValueError(
                "The coarsening must cover exactly the fine names of the vocabulary, missing "
                f"{sorted(fine_names - coarsened_names)}, unknown "
                f"{sorted(coarsened_names - fine_names)}."
            )
        class_set = set(class_names)
        for fine_name, class_name in coarsening.items():
            if class_name is not None and class_name not in class_set:
                raise ValueError(
                    f"Fine label {fine_name!r} coarsens to {class_name!r}, which is not a class "
                    f"of the level {list(class_names)}."
                )

        self._vocabulary = vocabulary
        self._class_names = tuple(class_names)
        self._coarsening = dict(coarsening)
        self._ignore_index = ignore_index
        self._class_indices = {name: index for index, name in enumerate(self._class_names)}

    @property
    def vocabulary(self) -> LabelVocabulary:
        """Raw label names mapped onto fine label names."""
        return self._vocabulary

    @property
    def class_names(self) -> tuple[str, ...]:
        """Classes of the level, in index order."""
        return self._class_names

    @property
    def num_classes(self) -> int:
        """Number of classes of the level."""
        return len(self._class_names)

    @property
    def ignore_index(self) -> int:
        """Label index of a label outside the classes of the level."""
        return self._ignore_index

    @property
    def coarsening(self) -> Mapping[str, str | None]:
        """Fine label name to class name, None for a dropped fine label."""
        return self._coarsening

    def fine_name(self, raw_name: str) -> str:
        """
        Fine label name of a raw label name.

        Args:
          raw_name: Raw label name of the dataset.

        Returns:
          str: The fine label name.
        """

        return self._vocabulary.fine_name(raw_name)

    def class_name(self, fine_name: str) -> str | None:
        """
        Class of the level a fine label name coarsens to.

        Args:
          fine_name: Fine label name, or any name the vocabulary does not know.

        Returns:
          str | None: The class name, None for a dropped or unknown fine label.
        """

        return self._coarsening.get(fine_name)

    def class_index(self, fine_name: str) -> int:
        """
        Label index of a fine label name at the level.

        Args:
          fine_name: Fine label name, or any name the vocabulary does not know.

        Returns:
          int: The class index, the ignore index for a dropped or unknown fine label.
        """

        class_name = self.class_name(fine_name)
        if class_name is None:
            return self._ignore_index
        return self._class_indices[class_name]

    def resolve_index(self, raw_name: str) -> int:
        """
        Label index of a raw label name at the level.

        Args:
          raw_name: Raw label name of the dataset.

        Returns:
          int: The class index, the ignore index for a label outside the level.
        """

        return self.class_index(self.fine_name(raw_name))

    def __str__(self) -> str:
        """Canonical string form, the input of the database hash."""
        coarsening = ", ".join(
            f"{fine_name}: {class_name}"
            for fine_name, class_name in sorted(self._coarsening.items())
        )
        return (
            f"{self.__class__.__name__}(class_names={list(self._class_names)}, "
            f"ignore_index={self._ignore_index}, coarsening=({coarsening}), "
            f"vocabulary={self._vocabulary})"
        )

    def __eq__(self, other: object) -> bool:
        """Compare two taxonomies by their definition."""
        return type(self) is type(other) and str(self) == str(other)

    def __hash__(self) -> int:
        """Hash the taxonomy by its string form."""
        return hash(str(self))


class DatabaseTaxonomy:
    """Taxonomies of the label spaces a database describes."""

    def __init__(self, detection3d: LabelTaxonomy, segmentation3d: LabelTaxonomy) -> None:
        """
        Initialize the database taxonomy.

        Args:
          detection3d: Taxonomy the box labels are baked with.
          segmentation3d: Taxonomy the semantic mask categories are resolved with.
        """

        self._detection3d = detection3d
        self._segmentation3d = segmentation3d

    @property
    def detection3d(self) -> LabelTaxonomy:
        """Taxonomy the box labels are baked with."""
        return self._detection3d

    @property
    def segmentation3d(self) -> LabelTaxonomy:
        """Taxonomy the semantic mask categories are resolved with."""
        return self._segmentation3d

    def __str__(self) -> str:
        """Canonical string form, the input of the database hash."""
        return (
            f"{self.__class__.__name__}(detection3d={self._detection3d}, "
            f"segmentation3d={self._segmentation3d})"
        )

    def __eq__(self, other: object) -> bool:
        """Compare two database taxonomies by their definition."""
        return type(self) is type(other) and str(self) == str(other)

    def __hash__(self) -> int:
        """Hash the database taxonomy by its string form."""
        return hash(str(self))
