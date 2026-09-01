"""Framework export-contract tests for BaseModel."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import Tensor

from autoware_ml.datamodule.samples.batch import Batch, FrameMetaBatch
from autoware_ml.models.base import BaseModel
from autoware_ml.preprocessing.base import ModelInputs, ProcessedBatch


class _XYInputs(ModelInputs):
    """Derived inputs carrying two named tensors plus one the forward never reads."""

    x: Tensor
    y: Tensor
    unused: Tensor


class _StructuredExportModel(BaseModel):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"sum": x + y, "diff": x - y}

    def compute_metrics(self, processed: ProcessedBatch, outputs: Any) -> dict[str, torch.Tensor]:
        del processed
        return {"loss": outputs["sum"].sum()}

    def predict_outputs(
        self, processed: ProcessedBatch | None, outputs: Any
    ) -> dict[str, torch.Tensor]:
        del processed
        return outputs

    def get_export_output_names(self) -> list[str]:
        return ["sum", "diff"]


class _UnnamedStructuredModel(_StructuredExportModel):
    def get_export_output_names(self) -> list[str] | None:
        return None


def _make_processed() -> ProcessedBatch:
    meta = FrameMetaBatch(
        sample_ids=("sample-0",),
        scene_tokens=None,
        timestamps=(0.0,),
        ego2globals=None,
        prev_exists=None,
    )
    inputs = _XYInputs(
        x=torch.tensor([2.0]),
        y=torch.tensor([0.5]),
        unused=torch.tensor([99.0]),
    )
    return ProcessedBatch(batch=Batch(meta=meta), inputs=(inputs,))


def test_build_export_spec_uses_forward_signature_inputs() -> None:
    model = _StructuredExportModel()

    spec = model.build_export_spec(_make_processed())
    outputs = spec.module(*spec.args)

    assert spec.input_param_names == ["x", "y"]
    assert spec.output_names == ["sum", "diff"]
    assert len(spec.args) == 2
    assert isinstance(outputs, tuple)
    assert torch.equal(outputs[0], torch.tensor([2.5]))
    assert torch.equal(outputs[1], torch.tensor([1.5]))


def test_build_export_spec_rejects_unresolvable_forward_parameter() -> None:
    class _MissingInputModel(_StructuredExportModel):
        def forward(self, x: torch.Tensor, missing: torch.Tensor) -> dict[str, torch.Tensor]:
            return {"sum": x + missing, "diff": x - missing}

    model = _MissingInputModel()

    with pytest.raises(ValueError, match="missing"):
        model.build_export_spec(_make_processed())


def test_structured_export_outputs_require_names() -> None:
    model = _UnnamedStructuredModel()

    with pytest.raises(ValueError, match="explicit export output names"):
        model.prepare_export_outputs({"sum": torch.tensor([1.0])})
