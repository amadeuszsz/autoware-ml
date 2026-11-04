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

"""Selection of the samples of a batch of model outputs."""

from __future__ import annotations

from typing import Any

import torch
from jaxtyping import Bool
from pydantic import BaseModel


def select_output_samples(
    outputs: BaseModel, mask: Bool[torch.Tensor, " batch_size"]
) -> dict[str, Any]:
    """
    Every field of some model outputs for the samples a batch mask selects.

    Args:
        outputs: Outputs whose fields lead with the batch: tensors, None, outputs with their own
            ``select_samples`` or tuples of those.
        mask: True for every sample to keep.

    Returns:
        The fields of the selected samples by name.
    """
    return {name: _select(getattr(outputs, name), mask) for name in type(outputs).model_fields}


def _select(value: Any, mask: Bool[torch.Tensor, " batch_size"]) -> Any:
    """One field of the selected samples, a missing output stays missing."""
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value[mask]
    if hasattr(value, "select_samples"):
        return value.select_samples(mask)
    return tuple(_select(item, mask) for item in value)
