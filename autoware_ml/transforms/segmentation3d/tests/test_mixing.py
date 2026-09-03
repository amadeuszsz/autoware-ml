"""Tests for the segmentation mixing transforms."""

from __future__ import annotations

import numpy as np
import pytest

from autoware_ml.datamodule.pipeline_context import PipelineContext
from autoware_ml.datamodule.samples.point_cloud import PointCloud
from autoware_ml.datamodule.samples.sample import Sample
from autoware_ml.datamodule.samples.segmentation3d import SegmentationLabels
from autoware_ml.testing.factories import make_point_cloud, make_sample
from autoware_ml.transforms.base import BaseTransform, TransformsCompose
from autoware_ml.transforms.segmentation3d.mixing import FrustumMix, InstanceCopy
from autoware_ml.types.geometry import PointFeatureName


class _MixDataset:
    """Single sample dataset stub feeding typed secondary samples."""

    def __init__(self, sample: Sample) -> None:
        self._sample = sample

    def __len__(self) -> int:
        return 1

    def build_seed_sample(self, index: int) -> Sample:
        del index
        return self._sample

    def apply_transforms(
        self,
        sample: Sample,
        dataset_transforms: TransformsCompose | None,
        context: PipelineContext,
    ) -> Sample:
        if dataset_transforms is None:
            return sample
        return dataset_transforms(sample, context)


def _seg_sample(features: list[list[float]], labels: list[int]) -> Sample:
    points = PointCloud(
        features=np.array(features, dtype=np.float32),
        feature_names=(
            PointFeatureName.X,
            PointFeatureName.Y,
            PointFeatureName.Z,
            PointFeatureName.INTENSITY,
        ),
        num_current_points=len(features),
    )
    sample = make_sample(points=points)
    return sample.replace(segment=SegmentationLabels(labels=np.array(labels, dtype=np.int64)))


def _context(sample: Sample) -> PipelineContext:
    return PipelineContext(dataset=_MixDataset(sample), index=0, rng=np.random.default_rng(0))


def test_frustum_mix_combines_points_from_both_samples() -> None:
    np.random.seed(0)
    sample = _seg_sample([[5.0, 0.0, 0.0, 1.0], [4.0, 1.0, 0.0, 2.0]], [1, 2])
    mix_sample = _seg_sample([[3.0, -1.0, 0.0, 3.0], [2.0, 0.0, 0.0, 4.0]], [9, 8])

    output = FrustumMix(height=8, width=16, fov_up=10.0, fov_down=-30.0, num_areas=[2], p=1.0)(
        sample, context=_context(mix_sample)
    )

    assert output.points.features.shape[1] == 4
    assert len(output.points) == len(output.segment)
    assert output.points.num_current_points == len(output.points)
    assert set(output.segment.labels.tolist()).issubset({1, 2, 8, 9})
    assert any(label in {8, 9} for label in output.segment.labels.tolist())


def test_instance_copy_appends_requested_semantic_classes() -> None:
    sample = _seg_sample([[0.0, 0.0, 0.0, 1.0]], [1])
    mix_sample = _seg_sample([[1.0, 0.0, 0.0, 2.0], [2.0, 0.0, 0.0, 3.0]], [4, 5])

    output = InstanceCopy(instance_classes=[5], p=1.0)(sample, context=_context(mix_sample))

    assert output.points.features.shape == (2, 4)
    assert np.array_equal(output.segment.labels, np.array([1, 5], dtype=np.int64))
    assert output.points.num_current_points == 2


@pytest.mark.parametrize(
    "transform",
    [
        FrustumMix(height=8, width=16, fov_up=10.0, fov_down=-30.0, num_areas=[2], p=0.0),
        InstanceCopy(instance_classes=[5], p=0.0),
    ],
)
def test_mixing_respects_zero_probability(transform: BaseTransform) -> None:
    sample = _seg_sample([[5.0, 0.0, 0.0, 1.0]], [1])

    output = transform(sample)

    assert np.array_equal(output.points.features, sample.points.features)
    assert np.array_equal(output.segment.labels, sample.segment.labels)


def test_mixing_applies_pre_transform_to_secondary_sample() -> None:
    class _RelabelNinetyNine(BaseTransform):
        def transform(self, sample: Sample) -> Sample:
            labels = np.full_like(sample.segment.labels, 99)
            return sample.replace(segment=sample.segment.model_copy(update={"labels": labels}))

    sample = _seg_sample([[0.0, 0.0, 0.0, 1.0]], [1])
    mix_sample = _seg_sample([[1.0, 0.0, 0.0, 2.0]], [5])

    output = InstanceCopy(
        instance_classes=[99],
        pre_transform=TransformsCompose(pipeline=[_RelabelNinetyNine()]),
        p=1.0,
    )(sample, context=_context(mix_sample))

    assert np.array_equal(output.segment.labels, np.array([1, 99], dtype=np.int64))


@pytest.mark.parametrize(
    "transform",
    [
        FrustumMix(height=8, width=16, fov_up=10.0, fov_down=-30.0, num_areas=[2], p=1.0),
        InstanceCopy(instance_classes=[5], p=1.0),
    ],
)
def test_mixing_rejects_time_lag_clouds(transform: BaseTransform) -> None:
    points = make_point_cloud(num_points=4, with_time_lag=True)
    sample = make_sample(points=points)
    sample = sample.replace(segment=SegmentationLabels(labels=np.zeros(4, dtype=np.int64)))

    with pytest.raises(ValueError, match="single frame"):
        transform(sample)


def test_mixing_requires_secondary_segmentation_labels() -> None:
    sample = _seg_sample([[0.0, 0.0, 0.0, 1.0]], [1])
    mix_sample = make_sample(points=make_point_cloud(num_points=2, with_time_lag=False))

    with pytest.raises(ValueError, match="segmentation labels"):
        InstanceCopy(instance_classes=[5], p=1.0)(sample, context=_context(mix_sample))


def test_mixing_rejects_mismatched_feature_layouts() -> None:
    sample = _seg_sample([[0.0, 0.0, 0.0, 1.0]], [1])
    mix_points = make_point_cloud(num_points=2, with_time_lag=True)
    mix_sample = make_sample(points=mix_points)
    mix_sample = mix_sample.replace(segment=SegmentationLabels(labels=np.zeros(2, dtype=np.int64)))

    with pytest.raises(ValueError, match="matching point feature layouts"):
        InstanceCopy(instance_classes=[5], p=1.0)(sample, context=_context(mix_sample))
