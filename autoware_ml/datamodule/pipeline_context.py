"""Pipeline context utilities for record driven dataset pipelines."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from autoware_ml.datamodule.samples.sample import Sample
from autoware_ml.transforms.base import TransformsCompose

logger = logging.getLogger(__name__)


def _seeded_rng() -> np.random.Generator:
    """Derive a generator from the globally seeded NumPy state.

    seed_everything(workers=True) seeds the global NumPy RNG per dataloader worker, so
    deriving from it keeps secondary sample draws reproducible.

    Returns:
        Generator seeded from the global NumPy RNG.
    """
    return np.random.default_rng(np.random.randint(2**32))


@dataclass
class PipelineContext:
    """Provide dataset access for context aware transforms.

    The context keeps orchestration state out of the samples while still allowing transforms
    such as sample mixing augmentations to request secondary examples.
    """

    dataset: Any
    index: int
    rng: np.random.Generator = field(default_factory=_seeded_rng)

    def build_seed_sample(self, index: int) -> Sample:
        """Build the untransformed seed sample of one dataset index.

        Args:
            index: Dataset index.

        Returns:
            Seed sample holding the record and the frame metadata.
        """
        return self.dataset.build_seed_sample(index)

    def sample_secondary(
        self,
        pre_transform: TransformsCompose | None = None,
    ) -> Sample:
        """Sample and optionally preprocess a secondary dataset example.

        Args:
            pre_transform: Optional pipeline applied to the sampled seed sample.

        Returns:
            Secondary sample, optionally materialized by pre_transform.
        """
        dataset_length = len(self.dataset)
        if dataset_length <= 1:
            logger.warning(
                "Dataset contains only one sample; reusing the current sample as the secondary sample."
            )
            secondary_index = self.index
        else:
            # Ensure the secondary index is different from the current index
            secondary_index = int(self.rng.integers(0, dataset_length - 1))
            if secondary_index >= self.index:
                secondary_index += 1

        sample = self.build_seed_sample(secondary_index)
        if pre_transform is None:
            return sample

        secondary_context = PipelineContext(
            dataset=self.dataset, index=secondary_index, rng=self.rng
        )
        return self.dataset.apply_transforms(sample, pre_transform, secondary_context)
