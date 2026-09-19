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

"""Dataset sources a datamodule split is assembled from."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from autoware_ml.databases.base_database import BaseDatabase


@dataclass(frozen=True)
class DatasetSource:
    """One database with the supervision it contributes to a split.

    A training run mixes corpora that are annotated differently: a pseudo labelled corpus
    carries boxes and masks of its own quality, while a hand labelled one is rehearsed in for
    its masks alone. The toggles say which annotations of a source supervise the run, so a
    corpus can take part in a split without its weaker annotations reaching the loss or the
    metrics.

    A toggled off task is neutralized rather than dropped: the boxes become an empty set and
    the semantic labels become the ignore index. Every sample of the split therefore carries
    the same fields and collates with the others, which a dropped field would not.

    Attributes:
      database: Database providing the dataset records of the source.
      det3d: Whether the box annotations of the source supervise the run.
      seg3d: Whether the semantic masks of the source supervise the run.
      repeat: How many times the frames of the source appear in one epoch.
    """

    database: BaseDatabase
    det3d: bool = True
    seg3d: bool = True
    repeat: int = 1

    def __post_init__(self) -> None:
        """Validate the source declaration."""
        if not isinstance(self.database, BaseDatabase):
            raise TypeError(
                f"A dataset source needs a database, got {type(self.database).__name__}."
            )
        if self.repeat < 1:
            raise ValueError(f"A dataset source repeats at least once, got {self.repeat}.")


def coerce_sources(
    sources: Sequence[DatasetSource | Mapping[str, Any]],
) -> tuple[DatasetSource, ...]:
    """Normalize the declared sources of one split to DatasetSource instances.

    Hydra composes a source list as plain mappings unless every entry names a target, so the
    mappings are built here instead of leaving each split to guess what it received.

    Args:
        sources: Declared sources, either DatasetSource instances or mappings of its fields.

    Returns:
        tuple[DatasetSource, ...]: The sources of the split, in declaration order.
    """
    if not len(sources):
        raise ValueError("A datamodule split is served by at least one dataset source.")

    normalized: list[DatasetSource] = []
    for source in sources:
        if isinstance(source, DatasetSource):
            normalized.append(source)
        elif isinstance(source, Mapping):
            normalized.append(DatasetSource(**dict(source)))
        else:
            raise TypeError(
                f"A dataset source is a DatasetSource or a mapping, got {type(source).__name__}."
            )
    return tuple(normalized)
