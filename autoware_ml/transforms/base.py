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

"""Base classes and composition utilities for typed sample transforms.

Every transform maps a Sample to a new Sample. Transforms never mutate their input, they build
the output through the copy helpers of the sample models.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import numpy as np

from autoware_ml.datamodule.samples.sample import Sample


class BaseTransform(ABC):
    """Abstract base class for sample transforms.

    Class attributes to override in subclasses:
        p: Probability of applying the transform. None means the transform always runs.
        _required_fields: Sample fields that must be set before the transform runs.
    """

    p: float | None = None
    _required_fields: Sequence[str] = ()
    pre_transform: Any = None

    def __call__(self, sample: Sample, context: Any = None) -> Sample:
        """Execute the transform with probability and field validation.

        Args:
            sample: Sample passed to the transform.
            context: Optional dataset pipeline context.

        Returns:
            Transformed sample.
        """
        self._context = context
        self._validate_required_fields(sample)
        if not self._should_apply():
            return self.on_skip(sample)
        output = self.transform(sample)
        if not isinstance(output, Sample):
            raise TypeError(
                f"{self.__class__.__name__} must return a Sample, got {type(output).__name__}."
            )
        return output

    @property
    def context(self) -> Any:
        """Return the active execution context for the current transform call.

        Returns:
            Pipeline context associated with the current sample, or None when the transform is
            executed outside a dataset pipeline.
        """
        return getattr(self, "_context", None)

    def _validate_required_fields(self, sample: Sample) -> None:
        """Raise when any required sample field is not set.

        Args:
            sample: Sample validated before transform execution.

        Raises:
            ValueError: If a required field of the transform is not set on the sample.
        """
        for field_name in self._required_fields:
            if getattr(sample, field_name) is None:
                raise ValueError(
                    f"{self.__class__.__name__}: The sample field '{field_name}' must be set "
                    f"before this transform runs."
                )

    def _should_apply(self) -> bool:
        """Determine if the transform should be applied based on its probability.

        Returns:
            True if the transform should be applied, False to skip.
        """
        if self.p is None:
            return True
        if self.p <= 0.0:
            return False
        if self.p >= 1.0:
            return True
        return np.random.rand() < self.p

    def on_skip(self, sample: Sample) -> Sample:
        """Handle a sample when the transform is skipped due to probability.

        Args:
            sample: The input sample.

        Returns:
            The sample forwarded to the next transform.
        """
        return sample

    @abstractmethod
    def transform(self, sample: Sample) -> Sample:
        """Process the sample and return a new one.

        Args:
            sample: Sample with every required field set.

        Returns:
            Transformed sample.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        """Return the class name of the transform.

        Returns:
            Class name of the transform.
        """
        return f"{self.__class__.__name__}()"


class TransformsCompose:
    """Apply a sequence of transforms in order.

    The composed transform forwards one sample through every configured transform and returns
    the final result.
    """

    def __init__(self, pipeline: Sequence[BaseTransform] = ()):
        """Initialize the transform composition.

        Args:
            pipeline: Ordered transforms applied to each sample.
        """
        self.pipeline = list(pipeline)

    def __call__(self, sample: Sample, context: Any = None) -> Sample:
        """Apply each transform sequentially.

        Args:
            sample: Sample passed through the configured transforms.
            context: Optional pipeline context forwarded to each transform.

        Returns:
            Transformed sample after all pipeline stages have been applied.
        """
        if not isinstance(sample, Sample):
            raise TypeError(
                f"{self.__class__.__name__} input must be a Sample, got {type(sample).__name__}."
            )
        for transform in self.pipeline:
            sample = transform(sample, context=context)
        return sample

    def __repr__(self) -> str:
        """Return a formatted string representation of the composition.

        Returns:
            Multi-line string showing the ordered transform pipeline.
        """
        if not self.pipeline:
            return f"{self.__class__.__name__}(pipeline=[])"

        format_string = [f"{self.__class__.__name__}("]
        for i, transform in enumerate(self.pipeline):
            format_string.append(f"  ({i}): {transform}")
        format_string.append(")")
        return "\n".join(format_string)
