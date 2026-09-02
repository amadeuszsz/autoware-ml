---
icon: lucide/database
---

# T4Dataset

This module implements the database layer for the **T4** annotation format, built on top of
the abstract base classes in the [database module](design.md).

## Summary

| Property     | Value                                                       |
| ------------ | ----------------------------------------------------------- |
| Format       | JSON (T4 annotation tables via `t4-devkit`)                 |
| Annotations  | 3D bounding boxes and point wise semantic masks             |
| Modality     | Multiple LiDAR and cameras                                  |
| Dependencies | `t4-devkit`, `polars`, `numpy`                              |
| Input        | Scenario list yaml files and T4 annotation directories      |
| Output       | Record table saved as Parquet via Polars                    |

## Module relationships

| Module                   | Role                                                                                         |
| ------------------------ | -------------------------------------------------------------------------------------------- |
| `t4scenarios.py`         | `T4Scenarios` extends `Scenarios`: reads the scenario list of every dataset per split        |
| `t4records_generator.py` | `T4RecordsGenerator` reads one scenario through `t4-devkit` and builds one record per sample |
| `t4database.py`          | `T4Database` extends `BaseDatabase`: one worker generates the records of one scenario        |

## Scenario lists

A dataset is described by one yaml file named after it below the scenario root, with the
scenario entries of every split:

```yaml
train:
  - <scenario_id>/<version>/<location>/<vehicle_type>/<status>
val:
  - <scenario_id>/<version>
test: []
```

The short and the annotated entry forms may be mixed. The location and the vehicle type of
an annotated entry are written into every record of the scenario. The scenario root is the
task directory of the perception-devops checkout, so the scenario lists live outside this
repository and the database config only names the datasets and their parameters.

## What the generator decides

- which samples are kept: every sample, every n-th sample, or only the masked samples of a
  dataset labelled at a lower rate than it was recorded
- how many preceding lidar frames each sample carries, with the transform from the sample's
  sensor frame into every sweep composed in double precision
- the calibration of every lidar sensor of the scene, the segmentation category table, and
  every camera image of the sample with its intrinsics, distortion and poses
- the fine label and the class index of every box, baked through the taxonomy and the box
  pipelines of the database, and optionally the recounted number of lidar points inside
  every box

Every path in a record is relative to the database root. A missing file, a point cloud
whose size does not match the declared feature count, or a mask whose length does not match
the point count stops the run.

## Output table schema

`T4Database.process_scenario_records()` produces `DatasetRecord` objects and persists them as
a Polars `DataFrame` written to Parquet, named after the database hash. For the complete
table layout and nested struct definitions, see [Dataset Schema](schemas.md).

## Implementation

| Path                                                     | Description                                       |
| -------------------------------------------------------- | ------------------------------------------------- |
| `autoware_ml/databases/t4dataset/t4scenarios.py`         | T4 scenario list parsing and split construction   |
| `autoware_ml/databases/t4dataset/t4records_generator.py` | T4 annotation reading and per sample extraction   |
| `autoware_ml/databases/t4dataset/t4database.py`          | T4 database orchestration with parallel workers   |
| `autoware_ml/configs/database/t4dataset/`                | Database, scenario group and pipeline configs     |

## Acknowledgment

T4Dataset is based on the nuScenes dataset schema.

<!-- cspell:ignore Bankiti Liong Krishnan Baldan Beijbom Vora -->
- Repository: <https://github.com/nutonomy/nuscenes-devkit>
- License: Apache 2.0
- Paper: Caesar, H., Bankiti, V., Lang, A. H., Vora, S., Liong, V. E., Xu, Q., Krishnan, A., Pan, Y., Baldan, G., and Beijbom, O. "nuScenes: A Multimodal Dataset for Autonomous Driving." CVPR, 2020.
