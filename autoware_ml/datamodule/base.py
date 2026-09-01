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

"""Base datamodule abstractions for Autoware-ML.

The datamodule reads dataset records from the configured record tables, selects the split
each table declares, and serves typed samples through the transform pipelines. Batches are
collated into the typed Batch, the single interface the models consume.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import logging
from typing import Any

import lightning as L
import numpy as np
import polars as pl
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from autoware_ml.databases.schemas.dataset_schemas import (
    DatasetRecord,
)
from autoware_ml.datamodule.samples.calibration import CalibrationSample
from autoware_ml.utils.calibration import CalibrationData
from autoware_ml.datamodule.pipeline_context import PipelineContext
from autoware_ml.datamodule.samplers import (
    DistributedWeightedRandomSampler,
    FrameSamplingConfig,
    coerce_frame_sampling,
    compute_frame_sampling_weights,
)
from autoware_ml.datamodule.samples.batch import Batch
from autoware_ml.datamodule.samples.sample import Sample
from autoware_ml.datamodule.samples.meta import FrameMeta
from autoware_ml.datamodule.sources import DatasetSource, coerce_sources
from autoware_ml.transforms.base import TransformsCompose

logger = logging.getLogger(__name__)

_STAGE_SPLITS: Mapping[str, Sequence[str]] = {
    "fit": ("train", "val"),
    "validate": ("val",),
    "test": ("test",),
    "predict": ("predict",),
}

_RECORD_SPLITS: Mapping[str, str] = {
    "train": "train",
    "val": "val",
    "test": "test",
    # The predict split serves the test frames without ground truth requirements
    "predict": "test",
}


@dataclass
class DataLoaderConfig:
    """Store configuration values for one dataloader.

    Attributes:
        batch_size: Number of samples per batch.
        num_workers: Number of worker processes used by the dataloader.
        pin_memory: Whether to pin host memory before device transfer.
        persistent_workers: Whether worker processes stay alive across epochs.
        shuffle: Whether the dataloader shuffles samples.
        drop_last: Whether to drop the final incomplete batch.
    """

    batch_size: int = 1
    num_workers: int = 1
    pin_memory: bool = False
    persistent_workers: bool = False
    shuffle: bool = False
    drop_last: bool = False

    def to_dataloader_kwargs(self) -> dict[str, Any]:
        """Convert to keyword arguments accepted by DataLoader.

        Returns:
            Dictionary of DataLoader constructor keyword arguments.
        """
        return {
            "batch_size": self.batch_size,
            "shuffle": self.shuffle,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers and self.num_workers > 0,
            "drop_last": self.drop_last,
        }


@dataclass(frozen=True)
class SourceRecords:
    """Records of one dataset source for one split.

    Attributes:
        source: The dataset source the records belong to.
        records: Dataset records of the split as a polars dataframe.
        data_root: Root directory the record paths resolve against.
    """

    source: DatasetSource
    records: pl.DataFrame
    data_root: str


class Dataset(TorchDataset, ABC):
    """Family agnostic dataset over dataset records.

    The dataset serves one split assembled from one or more sources. Every index maps to one
    record row of one source, repeated sources contribute their rows several times. A sample
    starts as the seeded record plus frame metadata, and the transform pipeline fills the task
    fields. Subclasses only derive the family specific frame metadata.
    """

    def __init__(
        self,
        dataset_transforms: TransformsCompose | None = None,
        calibration_cameras: Sequence[str] | None = None,
    ) -> None:
        """Initialize the dataset.

        Args:
            dataset_transforms: Transform pipeline applied per sample.
            calibration_cameras: Camera channels served by the calibration status task. When
                set, every record expands into one sample per listed camera and the samples
                are seeded with the calibration state of that camera.
        """
        self.dataset_transforms = dataset_transforms
        self.calibration_cameras = (
            tuple(calibration_cameras) if calibration_cameras is not None else None
        )
        self._source_records: tuple[SourceRecords, ...] = ()
        self._index_map: tuple[tuple[int, int, str | None], ...] = ()

    def assign_source_records(self, source_records: Sequence[SourceRecords]) -> None:
        """Assign the split records of every source and build the index map.

        Args:
            source_records: Records of every source of the split.
        """
        camera_names: tuple[str | None, ...] = (None,)
        if self.calibration_cameras is not None:
            camera_names = self.calibration_cameras
        index_map = []
        for source_index, entry in enumerate(source_records):
            for _ in range(entry.source.repeat):
                index_map.extend(
                    (source_index, row_index, camera_name)
                    for row_index in range(len(entry.records))
                    for camera_name in camera_names
                )
        self._source_records = tuple(source_records)
        self._index_map = tuple(index_map)
        if not len(self._index_map):
            raise ValueError(f"{self.__class__.__name__} received no records.")

    def __len__(self) -> int:
        """
        Get the number of samples of the split, including repeated sources.

        Returns:
          int: Number of samples.
        """

        return len(self._index_map)

    def load_record(self, index: int) -> tuple[DatasetRecord, SourceRecords]:
        """Load the dataset record behind one dataset index.

        The supervision toggles of the source are applied here: a source without det3d
        supervision loses its boxes and a source without seg3d supervision loses its category
        mapping, so unmatched segmentation labels map to the ignore index downstream.

        Args:
            index: Dataset index.

        Returns:
            Tuple of the dataset record and the source records entry it came from.
        """
        source_index, row_index, _ = self._index_map[index]
        entry = self._source_records[source_index]
        row = entry.records.row(row_index, named=True)
        record = DatasetRecord.load_from_dictionary(row)
        update: dict[str, Any] = {}
        if not entry.source.det3d:
            update["boxes_3d"] = []
        if not entry.source.seg3d:
            update["category_mapping"] = None
        if update:
            record = record.model_copy(update=update)
        return record, entry

    def build_seed_sample(self, index: int) -> Sample:
        """Build the untransformed seed sample of one dataset index.

        Args:
            index: Dataset index.

        Returns:
            Seed sample holding the record and the frame metadata.
        """
        record, entry = self.load_record(index)
        calibration = None
        camera_name = self._index_map[index][2]
        if camera_name is not None:
            calibration = self._build_calibration(record, camera_name)
        return Sample(
            record=record,
            data_root=entry.data_root,
            meta=self.build_meta(record),
            calibration=calibration,
        )

    @staticmethod
    def _build_calibration(record: DatasetRecord, camera_name: str) -> CalibrationSample:
        """Build the calibration state of one camera of a record.

        The lidar to camera transformation is composed through both ego poses to account for
        the ego motion between the lidar and camera capture timestamps.

        Args:
            record: Dataset record of the sample.
            camera_name: Camera channel the calibration belongs to.

        Returns:
            Calibration state of the camera.
        """
        if record.camera_frames is None:
            raise ValueError(
                f"Record {record.sample_id} carries no camera frames but the calibration task "
                f"requires camera {camera_name}."
            )
        camera_frames = [
            frame
            for frame in record.camera_frames
            if frame.camera_sensor_channel_name == camera_name
        ]
        if len(camera_frames) != 1:
            raise ValueError(
                f"Record {record.sample_id} must carry exactly one camera frame for channel "
                f"{camera_name}, got {len(camera_frames)}."
            )
        camera_frame = camera_frames[0]
        lidar_frame = record.lidar_frames[0]

        # Lidar frame -> ego at lidar time -> global -> ego at camera time -> camera frame
        lidar_to_camera = (
            np.linalg.inv(camera_frame.camera_sensor_to_ego_pose_matrix)
            @ np.linalg.inv(camera_frame.camera_frame_ego_pose_to_global_matrix)
            @ lidar_frame.lidar_frame_ego_pose_to_global_matrix
            @ lidar_frame.lidar_sensor_to_ego_pose_matrix
        )
        calibration_data = CalibrationData(
            camera_matrix=camera_frame.camera_intrinsic_matrix_fp32,
            distortion_coefficients=np.asarray(
                camera_frame.camera_distortion_coefficients, dtype=np.float32
            ),
            lidar_to_camera_transformation=lidar_to_camera.astype(np.float32),
            distortion_model=camera_frame.camera_distortion_model,
        )
        return CalibrationSample(data=calibration_data, camera_name=camera_name)

    def __getitem__(self, index: int) -> Sample:
        """Load and transform one sample.

        Args:
            index: Dataset index.

        Returns:
            Transformed sample.
        """
        sample = self.build_seed_sample(index)
        context = PipelineContext(dataset=self, index=index)
        return self.apply_transforms(sample, self.dataset_transforms, context)

    def apply_transforms(
        self,
        sample: Sample,
        dataset_transforms: TransformsCompose | None,
        context: PipelineContext,
    ) -> Sample:
        """Apply a transform pipeline to a sample.

        Args:
            sample: Seed sample.
            dataset_transforms: Transform pipeline applied to the sample.
            context: Pipeline context associated with the sample.

        Returns:
            Transformed sample.
        """
        if dataset_transforms is None:
            return sample
        return dataset_transforms(sample, context=context)

    def iter_records(self) -> Sequence[tuple[DatasetRecord, DatasetSource]]:
        """Iterate the records of the split in index order, with their sources.

        Returns:
            Sequence of record and source pairs, one per dataset index.
        """
        return [
            (
                self.load_record(index)[0],
                self._source_records[self._index_map[index][0]].source,
            )
            for index in range(len(self))
        ]

    @abstractmethod
    def build_meta(self, record: DatasetRecord) -> FrameMeta:
        """Build the family specific frame metadata of one record.

        Args:
            record: Dataset record of the sample.

        Returns:
            Frame metadata of the sample.
        """
        raise NotImplementedError("Dataset must implement build_meta")


class DataModule(L.LightningDataModule):
    """Lightning DataModule serving typed batches from the configured record tables.

    Setup reads the split each source declares in its record table and assigns the records
    to the split datasets, and the dataloaders collate typed samples into the typed Batch.
    Record tables are generated outside this repository, so nothing is built here.
    """

    def __init__(
        self,
        dataset: Any,
        sources: Sequence[DatasetSource | Mapping[str, Any]],
        train_transforms: TransformsCompose | None = None,
        val_transforms: TransformsCompose | None = None,
        test_transforms: TransformsCompose | None = None,
        predict_transforms: TransformsCompose | None = None,
        train_dataloader_cfg: DataLoaderConfig | Mapping[str, Any] | None = None,
        val_dataloader_cfg: DataLoaderConfig | Mapping[str, Any] | None = None,
        test_dataloader_cfg: DataLoaderConfig | Mapping[str, Any] | None = None,
        predict_dataloader_cfg: DataLoaderConfig | Mapping[str, Any] | None = None,
        train_frame_sampling: FrameSamplingConfig | Mapping[str, Any] | None = None,
    ):
        """Initialize DataModule.

        Args:
            dataset: Partial dataset factory of the dataset family. It is called once per
                split with the split transform pipeline as dataset_transforms.
            sources: Dataset sources served by this datamodule. All sources must belong to
                the dataset family of the factory.
            train_transforms: Transform pipeline applied to training samples.
            val_transforms: Transform pipeline applied to validation samples.
            test_transforms: Transform pipeline applied to test samples.
            predict_transforms: Transform pipeline applied to predict samples.
            train_dataloader_cfg: Configuration for the training dataloader.
            val_dataloader_cfg: Configuration for the validation dataloader.
            test_dataloader_cfg: Configuration for the test dataloader.
            predict_dataloader_cfg: Configuration for the predict dataloader.
            train_frame_sampling: Repeat factor sampling settings for the training split, or
                None for uniform sampling.
        """
        super().__init__()

        self.dataset_factory = dataset
        self.sources = coerce_sources(sources)
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.test_transforms = test_transforms
        self.predict_transforms = predict_transforms
        self.train_dataloader_cfg = self._coerce_dataloader_cfg(train_dataloader_cfg)
        self.val_dataloader_cfg = self._coerce_dataloader_cfg(val_dataloader_cfg)
        self.test_dataloader_cfg = self._coerce_dataloader_cfg(test_dataloader_cfg)
        self.predict_dataloader_cfg = self._coerce_dataloader_cfg(predict_dataloader_cfg)
        self.train_frame_sampling = coerce_frame_sampling(train_frame_sampling)

        self.train_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None
        self.test_dataset: Dataset | None = None
        self.predict_dataset: Dataset | None = None
        self._split_source_records: dict[str, list[SourceRecords]] = {}

    @staticmethod
    def _coerce_dataloader_cfg(
        cfg: DataLoaderConfig | Mapping[str, Any] | None,
    ) -> DataLoaderConfig:
        """Normalize dataloader config values to DataLoaderConfig.

        Hydra composition can pass split dataloader settings as a plain dict or DictConfig.
        Normalize those mapping inputs at the datamodule boundary so downstream code can rely
        on the dataclass API.

        Args:
            cfg: Optional dataloader config object or mapping.

        Returns:
            Normalized DataLoaderConfig instance.

        Raises:
            TypeError: If the provided value cannot be converted.
        """
        if cfg is None:
            return DataLoaderConfig()
        if isinstance(cfg, DataLoaderConfig):
            return cfg
        if isinstance(cfg, Mapping):
            return DataLoaderConfig(**dict(cfg))
        raise TypeError(
            "Expected dataloader config to be a DataLoaderConfig, mapping, or None, "
            f"got {type(cfg)!r}."
        )

    def _split_records(self, split: str) -> list[SourceRecords]:
        """Collect the records of every source for one split.

        Args:
            split: Dataset split name.

        Returns:
            Records of every source of the split.
        """
        if split in self._split_source_records:
            return self._split_source_records[split]

        record_split = _RECORD_SPLITS[split]
        source_records = [
            SourceRecords(
                source=source,
                records=source.records.load(record_split),
                data_root=source.records.data_root,
            )
            for source in self.sources
        ]
        self._split_source_records[split] = source_records
        return source_records

    def setup(self, stage: str | None = None) -> None:
        """Create the datasets of every split of the stage.

        Args:
            stage: Current stage, fit, validate, test, or predict, or None to prepare all
                splits.
        """
        splits = ("train", "val", "test", "predict") if stage is None else _STAGE_SPLITS[stage]
        for split in splits:
            if getattr(self, f"{split}_dataset") is not None:
                continue
            dataset = self.dataset_factory(dataset_transforms=getattr(self, f"{split}_transforms"))
            if not isinstance(dataset, Dataset):
                raise TypeError(
                    f"The dataset factory must build a Dataset, got {type(dataset).__name__}."
                )
            dataset.assign_source_records(self._split_records(split))
            setattr(self, f"{split}_dataset", dataset)

    @staticmethod
    def collate_fn(samples: list[Sample]) -> Batch:
        """Collate typed samples into the typed batch.

        Args:
            samples: Samples produced by the transform pipelines.

        Returns:
            Collated batch.
        """
        return Batch.collate(samples)

    def _create_dataloader(self, split: str) -> DataLoader:
        """Create a dataloader for the given split.

        Args:
            split: Dataset split name.

        Returns:
            Configured DataLoader for the split.
        """
        dataset = getattr(self, f"{split}_dataset")
        cfg: DataLoaderConfig = getattr(self, f"{split}_dataloader_cfg")
        kwargs = cfg.to_dataloader_kwargs()
        if split == "train" and self.train_frame_sampling is not None:
            weights = compute_frame_sampling_weights(
                dataset.iter_records(), self.train_frame_sampling
            )
            kwargs["shuffle"] = False
            return DataLoader(
                dataset=dataset,
                collate_fn=self.collate_fn,
                sampler=DistributedWeightedRandomSampler(dataset, weights),
                **kwargs,
            )
        return DataLoader(dataset=dataset, collate_fn=self.collate_fn, **kwargs)

    def train_dataloader(self) -> DataLoader:
        """Create the training dataloader.

        Returns:
            Training dataloader.
        """
        return self._create_dataloader("train")

    def val_dataloader(self) -> DataLoader:
        """Create the validation dataloader.

        Returns:
            Validation dataloader.
        """
        return self._create_dataloader("val")

    def test_dataloader(self) -> DataLoader:
        """Create the test dataloader.

        Returns:
            Test dataloader.
        """
        return self._create_dataloader("test")

    def predict_dataloader(self) -> DataLoader:
        """Create the prediction dataloader.

        Returns:
            Prediction dataloader.
        """
        return self._create_dataloader("predict")
