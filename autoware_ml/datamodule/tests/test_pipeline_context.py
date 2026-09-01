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

"""Tests for the pipeline context serving secondary samples to mixing transforms."""

from __future__ import annotations

from autoware_ml.datamodule.base import SourceRecords
from autoware_ml.datamodule.pipeline_context import PipelineContext
from autoware_ml.datamodule.samples.sample import Sample
from autoware_ml.datamodule.t4dataset.dataset import T4Dataset
from autoware_ml.datamodule.tests.fakes import make_source, make_stored_record, records_dataframe
from autoware_ml.testing.factories import make_point_cloud
from autoware_ml.transforms.base import BaseTransform, TransformsCompose


class _AttachPoints(BaseTransform):
    """Attach a synthetic point cloud so the pre transform pipeline is observable."""

    def transform(self, sample: Sample) -> Sample:
        return sample.model_copy(update={"points": make_point_cloud(num_points=4)})


def _dataset(num_records: int) -> T4Dataset:
    records = [make_stored_record(sample_id=f"sample-{index}") for index in range(num_records)]
    dataset = T4Dataset()
    dataset.assign_source_records(
        [SourceRecords(source=make_source(), records=records_dataframe(records), data_root="/data")]
    )
    return dataset


def test_sample_secondary_returns_a_typed_sample_of_another_index() -> None:
    context = PipelineContext(dataset=_dataset(2), index=0)

    secondary = context.sample_secondary()

    assert isinstance(secondary, Sample)
    assert secondary.meta.sample_id == "sample-1"
    assert secondary.points is None


def test_sample_secondary_applies_the_pre_transform_pipeline() -> None:
    context = PipelineContext(dataset=_dataset(3), index=1)

    secondary = context.sample_secondary(pre_transform=TransformsCompose([_AttachPoints()]))

    assert isinstance(secondary, Sample)
    assert secondary.points is not None
    assert len(secondary.points) == 4


def test_sample_secondary_reuses_the_current_index_on_a_single_record_dataset() -> None:
    context = PipelineContext(dataset=_dataset(1), index=0)

    secondary = context.sample_secondary()

    assert secondary.meta.sample_id == "sample-0"
