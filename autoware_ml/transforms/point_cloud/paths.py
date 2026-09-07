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

"""Path resolution of dataset record files."""

from __future__ import annotations

import os


def resolve_frame_path(data_root: str, relative_path: str) -> str:
    """
    Resolve a record relative path against the data root.

    Args:
      data_root: Root directory of the dataset files.
      relative_path: Path relative to the data root, as stored in the dataset record.

    Returns:
      str: Absolute path of the file.
    """

    if os.path.isabs(relative_path):
        raise ValueError(
            f"Dataset records must store paths relative to the data root, got {relative_path}."
        )
    return os.path.join(data_root, relative_path)
