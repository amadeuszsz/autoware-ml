"""Framework prediction-step contracts for BaseModel."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import Tensor

from autoware_ml.datamodule.samples.batch import Batch, FrameMetaBatch, PointCloudBatch
from autoware_ml.models.base import BaseModel
from autoware_ml.preprocessing.base import DataPreprocessing, ModelInputs, ProcessedBatch
from autoware_ml.types.geometry import PointFeatureName


class _ToyInputs(ModelInputs):
    """Derived inputs of the preprocessing layer under test."""

    x: Tensor
    preprocessed: Tensor


class _AddOnePreprocessing:
    """Derive ``x`` from the point cloud and mark that preprocessing ran."""

    def __call__(self, batch: Batch, *, is_training: bool) -> _ToyInputs:
        del is_training
        return _ToyInputs(
            x=batch.point_cloud.concatenated[:, 0] + 1.0,
            preprocessed=torch.tensor(True),
        )


class _ToyModel(BaseModel):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 2.0

    def compute_metrics(self, processed: ProcessedBatch, outputs: Any) -> dict[str, torch.Tensor]:
        del processed
        return {"loss": outputs.sum()}

    def predict_outputs(
        self, processed: ProcessedBatch | None, outputs: Any
    ) -> dict[str, torch.Tensor]:
        return {
            "prediction": outputs,
            "preprocessed": processed.resolve("preprocessed"),
        }


class _MissingLossModel(_ToyModel):
    def compute_metrics(self, processed: ProcessedBatch, outputs: Any) -> dict[str, torch.Tensor]:
        del processed, outputs
        return {"accuracy": torch.tensor(1.0)}


def _make_batch() -> Batch:
    meta = FrameMetaBatch(
        sample_ids=("sample-0",),
        scene_tokens=None,
        timestamps=(0.0,),
        ego2globals=None,
        prev_exists=None,
    )
    points = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    return Batch(
        meta=meta,
        point_cloud=PointCloudBatch(
            features=(points,),
            feature_names=(
                PointFeatureName.X,
                PointFeatureName.Y,
                PointFeatureName.Z,
                PointFeatureName.INTENSITY,
            ),
            num_current_points=(1,),
        ),
    )


def test_on_after_batch_transfer_applies_preprocessing_pipeline() -> None:
    model = _ToyModel()
    model.set_data_preprocessing(DataPreprocessing([_AddOnePreprocessing()]))

    processed = model.on_after_batch_transfer(_make_batch(), dataloader_idx=0)

    assert torch.equal(processed.resolve("x"), torch.tensor([2.0]))
    assert processed.resolve("preprocessed").item() is True


def test_predict_step_runs_forward_and_formats_predictions() -> None:
    model = _ToyModel()
    model.set_data_preprocessing(DataPreprocessing([_AddOnePreprocessing()]))

    processed = model.on_after_batch_transfer(_make_batch(), dataloader_idx=0)
    predictions = model.predict_step(processed, batch_idx=0)

    assert torch.equal(predictions["prediction"], torch.tensor([4.0]))
    assert predictions["preprocessed"].item() is True


def test_bind_forward_inputs_rejects_missing_required_parameter() -> None:
    model = _ToyModel()
    processed = ProcessedBatch(batch=_make_batch(), inputs=())

    with pytest.raises(ValueError, match="'x'"):
        model.bind_forward_inputs(processed)


def test_shared_step_requires_loss_metric() -> None:
    model = _MissingLossModel()
    model.set_data_preprocessing(DataPreprocessing([_AddOnePreprocessing()]))
    processed = model.on_after_batch_transfer(_make_batch(), dataloader_idx=0)

    with pytest.raises(ValueError, match="'loss' key"):
        model.training_step(processed, batch_idx=0)
