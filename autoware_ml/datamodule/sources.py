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

"""Multi-source dataset specifications.

A datamodule can mix several record tables with different supervision coverage, for example a
detection and segmentation corpus combined with a pseudo labeled segmentation corpus. Each
source declares explicitly which supervision its records provide (det3d and seg3d) and how
often its frames are repeated (repeat), so nothing is inferred from the record content.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from autoware_ml.databases.record_table import RecordTable


@dataclass(frozen=True)
class DatasetSource:
    """One record table with its declared supervision coverage.

    Attributes:
        records: Record table providing the dataset records of the source.
        det3d: Whether the source's detection annotations supervise training and evaluation.
            When False, the boxes of every record are dropped, box free frames contribute no
            detection loss and neutral detection metric entries.
        seg3d: Whether the source's segmentation labels supervise training and evaluation.
            When False, the category mapping of every record is emptied so every point maps to
            the ignore index. The mask file is still loaded, so a mask path must exist for
            every frame of a segmentation pipeline.
        repeat: How many times the source's frames appear per epoch, physical repetition.
    """

    records: RecordTable
    det3d: bool = True
    seg3d: bool = True
    repeat: int = 1

    def __post_init__(self) -> None:
        """Validate the source specification."""
        if self.repeat < 1:
            raise ValueError(f"Dataset source repeat must be >= 1, got {self.repeat}.")


def coerce_sources(
    sources: Sequence[DatasetSource | Mapping[str, Any]],
) -> tuple[DatasetSource, ...]:
    """
    Normalize dataset source entries to DatasetSource instances.

    Args:
      sources: Source entries, either DatasetSource instances or mappings with the
        DatasetSource fields.

    Returns:
      tuple[DatasetSource, ...]: Normalized sources.
    """

    if not len(sources):
        raise ValueError("A datamodule requires at least one dataset source.")
    normalized = []
    for entry in sources:
        if isinstance(entry, DatasetSource):
            normalized.append(entry)
        elif isinstance(entry, Mapping):
            normalized.append(DatasetSource(**dict(entry)))
        else:
            raise TypeError(
                f"Dataset source entries must be DatasetSource or mappings, got {type(entry)!r}."
            )
    return tuple(normalized)
