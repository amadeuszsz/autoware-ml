---
icon: lucide/settings
---

# Configuration

Autoware-ML uses [Hydra](https://hydra.cc/) for configuration management. This gives you hierarchical YAML configs with powerful runtime overrides without code changes.

## Config Structure

All configs live in `autoware_ml/configs/`:

```text
configs/
├── records/           # Record table definitions, one per corpus
├── datasets/          # Shared dataset parameters referenced by task configs
├── defaults/          # Base settings and module defaults
├── generators/        # Dataset record generation configs
└── tasks/             # Task-specific configs
```

## Hydra Syntax

### `# @package _global_`

Always include this directive at the top of task configs to merge contents at the root level:

```yaml
# @package _global_
defaults:
  - /defaults/default_runtime
  - _self_
```

### `_target_`

Specifies the Python class or function to instantiate:

```yaml
model:
  _target_: autoware_ml.models.my_task.MyModel
  num_classes: 10
```

Nested `_target_` keys are recursively instantiated by default.

### `_partial_`

Use `_partial_: true` when you want Hydra to create a `functools.partial` instead of calling the function immediately:

```yaml
optimizer:
  _target_: torch.optim.AdamW
  _partial_: true
  lr: 0.001
  weight_decay: 0.01
```

This creates `functools.partial(AdamW, lr=0.001, weight_decay=0.01)` which can later be called with additional arguments (like `params`).

### `_recursive_`

Controls whether nested `_target_` keys are instantiated (default: `true`). Set to `false` to receive raw configs.

## Top-Level Config Keys

A complete task config includes these sections:

### `datamodule`

Controls data sources and split-specific transforms. The datamodule reads dataset records from
the configured record tables, keeps the split each table declares, and serves typed samples
through the transform pipelines:

```yaml
datamodule:
  _target_: autoware_ml.datamodule.base.DataModule
  dataset:
    _target_: autoware_ml.datamodule.t4dataset.dataset.T4Dataset
    _partial_: true
  sources:
    - records: ${records}
      det3d: true
      seg3d: false
      repeat: 1

  train_dataloader_cfg:
    batch_size: 8
    num_workers: 4
    shuffle: true

  train_transforms:
    _target_: autoware_ml.transforms.base.TransformsCompose
    pipeline:
      - _target_: autoware_ml.transforms.point_cloud.crop.PointsRangeFilter
        point_cloud_range: ${point_cloud_range}
```

The section is built from these blocks:

- `dataset` - partial factory of the dataset family (`T4Dataset` or `NuscenesDataset`). The datamodule calls it once per split with the transform pipeline of that split.
- `sources` - record tables served by the datamodule. Every source declares its supervision coverage (`det3d`, `seg3d`) and how often its frames appear per epoch (`repeat`), so one datamodule can mix corpora with different labels.
- `train/val/test/predict_transforms` - per-split transform pipelines, applied per sample on CPU.
- `train/val/test/predict_dataloader_cfg` - per-split dataloader settings.
- `train_frame_sampling` - optional repeat factor sampling settings for the training split.

The `records` value of a source comes from the `configs/records/` group and is bound through a
defaults entry in the task config:

```yaml
defaults:
  - /records@records: t4dataset/t4dataset_j6gen2_base
```

A record table is a parquet file generated outside this repository, by t4dataset-generator for
T4dataset. It carries its own splits and database names, so a corpus config is just a table path,
the data root its paths resolve against, and the databases to keep:

```yaml
_target_: autoware_ml.databases.record_table.RecordTable
path: /workspace/records/t4dataset.parquet
data_root: ${data_root_path}/t4dataset/
databases:
  - db_j6gen2_v1
```

Tables live at a fixed `/workspace/records`, mounted with `--records-path` or
`AUTOWARE_ML_RECORDS_PATH`, so the data mount can stay read only and every user can map their
own records directory.

Collation is not configurable. `Batch.collate` turns the transformed samples into the typed
`Batch` the models consume, and model family specific layouts are derived later by the runtime
preprocessing on the target device.

For custom components, point `_target_` at the concrete implementation module,
for example `autoware_ml.transforms.point_cloud.crop.PointsRangeFilter` or
`autoware_ml.models.common.backbones.my_backbone.MyBackbone`.

### `data_preprocessing`

Defines runtime preprocessing that runs after batch transfer and before
`forward()`. This section is instantiated by the entrypoint and attached to the
model through `BaseModel.set_data_preprocessing(...)`:

```yaml
data_preprocessing:
  _target_: autoware_ml.preprocessing.base.DataPreprocessing
  pipeline:
    - _target_: autoware_ml.preprocessing.my_preprocessing.my_preprocessing.MyPreprocessingLayer
      param: value
```

Output-side shaping (logits -> probabilities, voxel-to-point scatter, etc.)
is **not** a configurable framework pipeline - it lives inside the model's
own `forward()`, `compute_metrics()`, and `predict_outputs()` methods.

### `model`

Defines model architecture and optimization:

```yaml
model:
  _target_: autoware_ml.models.my_task.MyModel

  backbone:
    _target_: autoware_ml.models.common.backbones.my_backbone.MyBackbone
    in_channels: 3

  optimizer:
    _target_: torch.optim.AdamW
    _partial_: true
    lr: 0.001
    weight_decay: 0.01

  scheduler:
    _target_: torch.optim.lr_scheduler.CosineAnnealingLR
    _partial_: true
    T_max: ${trainer.max_epochs}
```

### `trainer`

PyTorch Lightning Trainer settings:

```yaml
trainer:
  _target_: lightning.Trainer
  max_epochs: 30
  accelerator: auto
  devices: auto
  precision: 16-mixed
  gradient_clip_val: 10.0
  gradient_clip_algorithm: norm
  accumulate_grad_batches: 1
  check_val_every_n_epoch: 1
  log_every_n_steps: 10
```

### `callbacks`

Lightning callbacks for checkpointing, early stopping, etc.:

```yaml
callbacks:
  model_checkpoint:
    _target_: lightning.pytorch.callbacks.ModelCheckpoint
    dirpath: ${hydra:run.dir}/checkpoints
    filename: "epoch={epoch}-step={step}"
    save_top_k: 3
    monitor: val/loss
    mode: min
    save_last: true

  early_stopping:
    _target_: lightning.pytorch.callbacks.EarlyStopping
    monitor: val/loss
    patience: 10
    mode: min
```

### `logger`

MLflow experiment tracking:

```yaml
logger:
  _target_: lightning.pytorch.loggers.MLFlowLogger
  tracking_uri: sqlite:///mlruns/mlflow.db
```

Autoware-ML populates `experiment_name`, `run_name`, `run_id`, and default tags automatically at runtime.

### `deploy`

ONNX and TensorRT export settings:

```yaml
deploy:
  onnx:
    opset_version: 21
    input_names: [input]
    output_names: [output]
    dynamic_shapes:
      fused_img: { 2: height, 3: width }
    modify_graph: null  # Optional graph modifier

  tensorrt:
    workspace_size: 8589934592  # 8 GiB
    input_shapes:
      input:
        min_shape: [1, 5, 1080, 1920]
        opt_shape: [1, 5, 1440, 2560]
        max_shape: [1, 5, 2160, 3840]
```

## Config Inheritance

Configs inherit using the `defaults` key:

```yaml
# @package _global_
defaults:
  - /tasks/my_task/my_model/base  # Inherit base config
  - _self_                        # Apply this file's overrides

# Override specific values
batch_size: 16

datamodule:
  train_dataloader_cfg:
    batch_size: ${batch_size}
```

## Variable Interpolation

Reference other config values with `${...}`:

```yaml
point_cloud_range: [-122.4, -122.4, -3.0, 122.4, 122.4, 5.0]

model:
  point_cloud_range: ${point_cloud_range}
```

Hydra resolvers:

```yaml
output_dir: ${hydra:run.dir}          # Hydra's output directory
experiment: ${hydra:job.config_name}  # Config name
```

## Runtime Overrides

Override any value from the command line:

```bash
# Override existing parameter (no + prefix)
autoware-ml train --config-name <task>/<model>/<config> \
    trainer.max_epochs=100

# Nested override
autoware-ml train --config-name <task>/<model>/<config> \
    model.optimizer.lr=0.0005

# Add new parameter (use + prefix)
autoware-ml train --config-name <task>/<model>/<config> \
    +callbacks.my_callback._target_=lightning.pytorch.callbacks.MyCallback
```

!!! warning "Override vs Add"
    Use `+` prefix only when adding a **new** parameter that doesn't exist in the config. For overriding existing parameters, use the path directly without `+`.

## Creating Custom Configs

Create a new YAML file inheriting from a base config:

```yaml title="configs/tasks/my_task/my_model/my_experiment.yaml"
# @package _global_
defaults:
  - /tasks/my_task/my_model/base
  - _self_

trainer:
  max_epochs: 50

model:
  optimizer:
    lr: 0.0001
```

Run with your config:

```bash
autoware-ml train --config-name my_task/my_model/my_experiment
```

## Debugging Configs

Print the resolved config without running:

```bash
autoware-ml train --config-name <task>/<model>/<config> --cfg job
```

Print a specific section:

```bash
autoware-ml train --config-name <task>/<model>/<config> \
    --cfg job --package model
```

## Multi-Run (Sweeps)

Run parameter sweeps with `--multirun`:

```bash
autoware-ml train --config-name <task>/<model>/<config> \
    --multirun \
    model.optimizer.lr=0.001,0.0005,0.0001
```

For intelligent hyperparameter search, see [Optuna](optuna.md).

## Learn More

- [Hydra Documentation](https://hydra.cc/docs/intro/)
- [OmegaConf Reference](https://omegaconf.readthedocs.io/)
