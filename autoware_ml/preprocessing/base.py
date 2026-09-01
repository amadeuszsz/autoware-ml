# Copyright 2025 TIER IV, Inc.
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

"""Base classes for GPU oriented batch preprocessing pipelines.

The preprocessing pipeline runs after batch transfer and derives the model family specific
input tensors from the typed batch, for example voxel grids or frustum projections. Its
product wraps the batch together with the derived inputs, and the model binds its forward
parameters against both by name.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict

from autoware_ml.datamodule.samples.batch import Batch


class ModelInputs(BaseModel):
    """Base class for the typed inputs a preprocessing layer derives from a batch.

    Subclasses declare the tensors of one model family as fields. Field names are the
    parameter names the model forwards bind against.
    """

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)


class ProcessedBatch(BaseModel):
    """Typed batch together with the derived model inputs.

    The models resolve their forward parameters against this object: derived inputs first,
    then the flat batch properties. Everything the training, evaluation, and export paths
    consume flows through here.

    Attributes:
      batch: The collated typed batch.
      inputs: Derived model inputs, one entry per preprocessing layer.
    """

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    batch: Batch
    inputs: tuple[ModelInputs, ...] = ()

    def resolve(self, name: str) -> Any:
        """Resolve one value by name from the derived inputs or the batch.

        Args:
          name: Field name of a derived input or property name of the batch.

        Returns:
          Any: The resolved value.

        Raises:
          AttributeError: When no derived input and no batch property carries the name.
          ValueError: When the name resolves to an absent optional batch field.
        """
        for inputs in self.inputs:
            if name in type(inputs).model_fields:
                return getattr(inputs, name)
        if not hasattr(self.batch, name):
            raise AttributeError(
                f"'{name}' is neither a derived model input "
                f"({self.available_input_names()}) nor a batch property."
            )
        value = getattr(self.batch, name)
        if value is None:
            raise ValueError(
                f"'{name}' is not available on this batch. The pipeline did not produce the "
                f"task field backing it."
            )
        return value

    def has(self, name: str) -> bool:
        """Check whether a name resolves to an available value.

        Args:
          name: Field name of a derived input or property name of the batch.

        Returns:
          bool: True when resolve would succeed.
        """
        for inputs in self.inputs:
            if name in type(inputs).model_fields:
                return True
        return hasattr(self.batch, name) and getattr(self.batch, name) is not None

    def available_input_names(self) -> tuple[str, ...]:
        """List the field names of every derived input.

        Returns:
          tuple[str, ...]: Field names of the derived inputs.
        """
        names: list[str] = []
        for inputs in self.inputs:
            names.extend(type(inputs).model_fields)
        return tuple(names)


class DataPreprocessing:
    """Apply a sequence of preprocessing layers to a collated batch.

    This runtime pipeline runs after batch transfer, enabling hardware accelerated
    preprocessing such as voxelization without registering the layers as part of the neural
    network. Every layer receives the typed batch and returns one ModelInputs instance.
    """

    def __init__(self, pipeline: Sequence[Any] = ()) -> None:
        """Initialize preprocessing with optional layers.

        Args:
            pipeline: Callable layers applied sequentially. Each layer accepts
                (batch, is_training) and returns a ModelInputs instance.
        """
        self.pipeline = list(pipeline)

    def __call__(self, batch: Batch, *, is_training: bool) -> ProcessedBatch:
        """Apply the preprocessing layers after the batch is already on device.

        Args:
            batch: Collated typed batch on the target device.
            is_training: Whether the owning model is in training mode. Passed to every layer
                so mode dependent behavior never relies on implicit module state.

        Returns:
            The processed batch wrapping the batch and the derived inputs.
        """
        if not isinstance(batch, Batch):
            raise TypeError(f"DataPreprocessing expects a Batch, got {type(batch).__name__}.")
        derived = []
        for layer in self.pipeline:
            inputs = layer(batch, is_training=is_training)
            if not isinstance(inputs, ModelInputs):
                raise TypeError(
                    f"{type(layer).__name__} must return ModelInputs, got {type(inputs).__name__}."
                )
            derived.append(inputs)
        return ProcessedBatch(batch=batch, inputs=tuple(derived))
