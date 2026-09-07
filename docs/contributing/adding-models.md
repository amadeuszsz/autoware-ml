---
icon: lucide/plus-circle
---

# Adding Models

This guide walks you through adding a new model to Autoware-ML. You'll implement a model class, provide the data through the shared datamodule, and wire everything together with a config.

## The BaseModel Interface

New models should inherit from `BaseModel`. The minimal contract is two abstract methods:

```python
from autoware_ml.models.base import BaseModel
from autoware_ml.preprocessing.base import ProcessedBatch

class MyModel(BaseModel):
    def forward(self, **kwargs: Any) -> torch.Tensor | Sequence[torch.Tensor]:
        ...

    def compute_metrics(
        self, processed: ProcessedBatch, outputs: Any
    ) -> dict[str, torch.Tensor]:
        ...
```

The base class handles training/validation/test/predict steps, optimizer
configuration, metric logging, prediction output conversion, runtime
preprocessing, and deployment export integration. Every `forward()` parameter
is bound by name: the base class resolves it against the derived model inputs
first and the flat properties of the typed `Batch` second, and a required
parameter that resolves to nothing raises immediately.

!!! note "Extending `BaseModel`"
    Specialized models should still use `BaseModel`. When the default
    signature-based path is not enough, prefer overriding hooks such as
    `set_data_preprocessing()`, `predict_outputs()`, `get_log_batch_size()`,
    or `build_export_spec()` instead of introducing a standalone
    `LightningModule`. Output decoding (for example, voxel-to-point scatter
    for segmentation) belongs inside the model, typically in `forward()`,
    `compute_metrics()`, and `predict_outputs()` - not in a separate
    framework pipeline.

## Step 1: Implement the Model

Create a new file in `autoware_ml/models/`:

```python title="autoware_ml/models/my_task/my_model.py"
from typing import Any

import torch
import torch.nn as nn

from autoware_ml.models.base import BaseModel
from autoware_ml.preprocessing.base import ProcessedBatch


class MyModel(BaseModel):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        num_classes: int,
        **kwargs: Any,  # Pass optimizer, scheduler, metrics to BaseModel
    ):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.num_classes = num_classes
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, fused_img: torch.Tensor) -> torch.Tensor:
        features = self.encoder(fused_img)
        logits = self.decoder(features)
        return logits

    def compute_metrics(
        self,
        processed: ProcessedBatch,
        outputs: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        gt_labels = processed.resolve("gt_calibration_status")
        loss = self.loss_fn(outputs, gt_labels)

        preds = torch.argmax(outputs, dim=1)
        accuracy = (preds == gt_labels).float().mean()

        return {
            "loss": loss,
            "accuracy": accuracy,
        }
```

### Key Points

1. **`forward()` parameters are bound by name** - Every parameter resolves against the fields of the derived `ModelInputs` first and the flat properties of the typed `Batch` second (for example `points`, `gt_boxes`, `segment`, `fused_img`). A required parameter that resolves to nothing raises immediately.

2. **`compute_metrics()` receives the processed batch and outputs** - The first argument is the `ProcessedBatch` wrapping the typed batch and the derived model inputs, and the second is `outputs` from `forward()`. Read targets with `processed.resolve(name)` or through `processed.batch`.

3. **Return `'loss'`** - The metrics dict must include a `'loss'` key for backpropagation.

4. **Optimizer and scheduler** - Passed as callables to `BaseModel.__init__()`. Need to be marked as `_partial_: true` in YAML configs.

5. **Task metrics** - Models that report task metrics implement `build_eval_output(processed, outputs)`, mapping the raw forward outputs to the flat dict the attached metric suites read. The suites are passed through the `metrics` argument.

6. **Use hooks when needed** - If your model needs custom batch unpacking,
   prediction formatting, or an explicit deployment wrapper, override the
   appropriate `BaseModel` hook instead of bypassing the shared training and
   deployment flow.

## Step 2: Provide the Data

Models do not own datamodules. The shared `DataModule` in
`autoware_ml/datamodule/base.py` generates the record table of every configured
database when it is missing, splits the records by the scenario lists of the
database, and serves typed samples through the transform pipelines. What you
add depends on the data:

- **New corpus of a known format** - Reuse `T4Dataset` or `NuscenesDataset`
  and add a scenario group and a database config under
  `autoware_ml/configs/database/`. No code is needed.
- **New corpus format** - Subclass `BaseDatabase` and implement
  `generate_records()` emitting `DatasetRecord` objects (see
  [database design](../databases/design.md)). Then subclass `Dataset` and
  implement `build_meta()` returning the frame metadata of the family. The
  dataset seeds a typed `Sample` from one record, file loading and sample
  materialization happen in transforms.

### Data Flow

```text
build_seed_sample() -> transforms -> Batch.collate() -> BaseModel.on_after_batch_transfer() -> forward() -> compute_metrics()/predict_outputs()
```

1. `build_seed_sample()`: seed a typed `Sample` with the dataset record and the frame metadata
2. `transforms`: load files from the record and fill the task fields, per sample on CPU
3. `Batch.collate()`: collate the typed samples into the typed `Batch`
4. `BaseModel.on_after_batch_transfer()`: runtime preprocessing derives the model inputs on the target device
5. `forward()`: model inference/training forward pass, parameters bound by name
6. `compute_metrics()` / `predict_outputs()`: model owns any output shaping
   (e.g., voxel-to-point scatter for segmentation) directly inside these
   methods

## Step 3: Create Config

Keep `__init__.py` files empty and point every `_target_` at the concrete
implementation module. Create a task config:

```yaml title="configs/tasks/my_task/my_model/base.yaml"
# @package _global_
defaults:
  - /defaults/default_runtime
  - _self_

model:
  _target_: autoware_ml.models.my_task.my_model.MyModel
  num_classes: 10

  encoder:
    _target_: autoware_ml.models.common.backbones.resnet.ResNet18
    in_channels: 3

  decoder:
    _target_: torch.nn.Linear
    in_features: 512
    out_features: ${model.num_classes}

  optimizer:
    _target_: torch.optim.AdamW
    _partial_: true
    lr: 0.001
    weight_decay: 0.01

  scheduler:
    _target_: torch.optim.lr_scheduler.CosineAnnealingLR
    _partial_: true
    T_max: ${trainer.max_epochs}

trainer:
  max_epochs: 50

data_preprocessing:
  _target_: autoware_ml.preprocessing.base.DataPreprocessing
  pipeline: []
```

Create a dataset-specific config binding the database group and the datamodule
sources:

```yaml title="configs/tasks/my_task/my_model/my_variant_t4dataset_j6gen2.yaml"
# @package _global_
defaults:
  - /tasks/my_task/my_model/base
  - /datasets/t4dataset/detection3d
  - /datasets/t4dataset/lidar
  - /database@database: t4dataset/t4dataset_j6gen2_base
  - _self_

batch_size: 8
num_workers: 8

dataset: ${t4dataset}

datamodule:
  dataset:
    _target_: autoware_ml.datamodule.t4dataset.dataset.T4Dataset
    _partial_: true
  train_sources:
    - database: ${database}
      det3d: true
      seg3d: false
      repeat: 1
  val_sources: ${datamodule.train_sources}
  test_sources: ${datamodule.train_sources}
  train_dataloader_cfg:
    batch_size: ${batch_size}
    num_workers: ${num_workers}
    shuffle: true
  val_dataloader_cfg:
    batch_size: ${batch_size}
    num_workers: ${num_workers}
```

!!! note
    Some parameters are inherited from the default runtime config, including the
    `DataModule` target and the scenario splitter. Take a look at
    `configs/defaults/default_runtime.yaml` and
    `configs/defaults/modules/datamodule.yaml` for more details.

Runtime preprocessing lives at the top level of the composed config and is
attached to the model by the entrypoints.

## Step 4: Add Transforms (Optional)

Transforms map a typed `Sample` to a new `Sample` and never mutate their
input. If your task needs custom transforms:

```python title="autoware_ml/transforms/my_transforms/my_transform.py"
import numpy as np

from autoware_ml.datamodule.samples.sample import Sample
from autoware_ml.transforms.base import BaseTransform


class MyAugmentation(BaseTransform):
    _required_fields = ["points"]

    def __init__(self, *, p: float | None = None, intensity: float = 0.1):
        # BaseTransform handles the application probability through `p`.
        self.p = p
        self.intensity = intensity

    def transform(self, sample: Sample) -> Sample:
        noise = (self.intensity * np.random.randn(len(sample.points), 3)).astype(np.float32)
        points = sample.points.with_coord(sample.points.coord + noise)
        return sample.replace(points=points)
```

Build the output through `Sample.replace()`, which validates the derived
sample, or through the copy helpers of the sample models. Operations that
filter or reorder points must go through `Sample.filter_points()` or
`Sample.reorder_points()` so aligned fields such as segmentation labels stay
consistent. `_required_fields` lists the sample fields validated before the
transform runs.

Add to config:

```yaml
datamodule:
  train_transforms:
    _target_: autoware_ml.transforms.base.TransformsCompose
    pipeline:
      - _target_: autoware_ml.transforms.my_transforms.my_transform.MyAugmentation
        p: 0.5
        intensity: 0.1
```

## Step 5: Add Runtime Data Preprocessing (Optional)

Runtime preprocessing runs on the target device after batch transfer and
before the forward pass. Every layer receives the typed `Batch` and returns
one `ModelInputs` instance whose field names are the parameter names the
model forward binds against. The pipeline wraps the batch together with the
derived inputs into a `ProcessedBatch`.

If your task needs custom preprocessing:

```python title="autoware_ml/preprocessing/my_preprocessing/my_preprocessing.py"
from torch import Tensor

from autoware_ml.datamodule.samples.batch import Batch
from autoware_ml.preprocessing.base import ModelInputs


class MyInputs(ModelInputs):
    input_tensor: Tensor


class MyPreprocessingLayer:
    def __init__(self, scale: float = 1.0):
        self.scale = scale

    def __call__(self, batch: Batch, *, is_training: bool) -> MyInputs:
        return MyInputs(input_tensor=batch.point_cloud.concatenated * self.scale)
```

Add to config:

```yaml
data_preprocessing:
  _target_: autoware_ml.preprocessing.base.DataPreprocessing
  pipeline:
    - _target_: autoware_ml.preprocessing.my_preprocessing.my_preprocessing.MyPreprocessingLayer
      scale: 1.0
```

!!! warning
    Preprocessing layers must be callable objects that accept `(batch, is_training)` and return a `ModelInputs` instance.

Output-side shaping (logits -> probabilities, decoder scatter, voxel-to-point mapping, etc.) belongs
**inside the model** - in `forward()`, `compute_metrics()`, or `predict_outputs()`.

## Step 6: Add Tests

Tests are co-located per namespace in `<package>/tests/`, for example
`autoware_ml/models/tests/` for models and
`autoware_ml/transforms/point_cloud/tests/` for point cloud transforms. Add
tests for a component once it is fully implemented, next to the code they
cover.

## Step 7: Train and Deploy

### Config Naming Convention

Task configs should follow:

```text
<task>/<model>/<variant>_<dataset>
```

Use these rules when creating `<variant>`:

- include only future-distinguishing choices such as backbone, modality, voxel size, or range
- do not encode properties that are inherent to the model family
- normalize voxel sizes as `voxel020`, `voxel005`
- encode ranges as human-readable suffixes such as `50m`, `90m`, `102m`, `121m`
- keep dataset names explicit and stable, for example `nuscenes` and `t4dataset_j6gen2`

Examples:

```text
segmentation3d/ptv3/voxel005_51m_nuscenes
segmentation3d/ptv3/voxel012_122m_t4dataset_j6gen2
my_task/my_model/my_variant_my_dataset
```

```bash
# Train
autoware-ml train --config-name my_task/my_model/my_variant_t4dataset_j6gen2

# Deploy
autoware-ml deploy \
    --config-name my_task/my_model/my_variant_t4dataset_j6gen2 \
    --weights mlruns/my_task/my_model/my_variant_t4dataset_j6gen2/<run_id>/artifacts/checkpoints/last.ckpt
```

## Common Patterns

### Multiple Inputs

```python
def forward(self, img: tuple[torch.Tensor, ...], points: tuple[torch.Tensor, ...]) -> torch.Tensor:
    img_features = self.image_encoder(img)
    lidar_features = self.lidar_encoder(points)
    fused = torch.cat([img_features, lidar_features], dim=1)
    return self.head(fused)
```

Every parameter must resolve by name, here against the `img` and `points`
properties of the typed `Batch`. Derived `ModelInputs` fields such as
`voxels` resolve the same way.

### Multiple Outputs

```python
def forward(self, points: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    features = self.backbone(points)
    boxes = self.box_head(features)
    scores = self.score_head(features)
    return boxes, scores

def compute_metrics(
    self,
    processed: ProcessedBatch,
    outputs: tuple[torch.Tensor, torch.Tensor],
):
    boxes, scores = outputs
    gt_boxes = processed.resolve("gt_boxes")
    gt_labels = processed.resolve("gt_labels")
    box_loss = self.box_loss(boxes, gt_boxes)
    score_loss = self.score_loss(scores, gt_labels)
    return {"loss": box_loss + score_loss, "box_loss": box_loss, "score_loss": score_loss}
```
