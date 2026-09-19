from abc import abstractmethod
from pathlib import Path
from typing import Sequence


import polars as pl
from torch.utils.data import Dataset

from autoware_ml.dataclasses.batch.sample_batch import ModelGTBatch, ModelGTSample
from autoware_ml.transforms.base import TransformsCompose


class BaseDataset(Dataset):
    """Multi-task dataset interface that can be shared by multiple databases."""

    def __init__(
        self,
        database_root_path: str,
        max_num_3d_gt_bboxes: int,
        dataset_records_dataframe: pl.DataFrame | None,
        transforms: TransformsCompose | None,
        det3d_supervised: bool = True,
        seg3d_supervised: bool = True,
    ) -> None:
        """
        Initialize the multi-task dataset interface.
        Args:
          database_root_path: Root directory of the dataset.
          max_num_3d_gt_bboxes: Maximum number of 3D ground truth bounding boxes in the dataset.
              This is allowed to be 0 if the dataset does not contain any 3D ground truth
              bounding boxes or it does not need to run 3D detection tasks.
          dataset_records_dataframe: Polars DataFrame of dataset records to be used in the
              multi-task dataset. Accept None if the dataset records
              are not available at initialization.
          transforms: Global transforms to be applied to the dataset records.
          det3d_supervised: Whether the box annotations of this corpus supervise the run. When
              False every sample carries an empty box set, so the frames still collate with
              the supervised ones while contributing no detection target.
          seg3d_supervised: Whether the semantic masks of this corpus supervise the run. When
              False every point takes the ignore index, so the frames still collate with the
              supervised ones while contributing no segmentation target.
        """
        super().__init__()
        self.database_root_path = Path(database_root_path)
        self.max_num_3d_gt_bboxes = max_num_3d_gt_bboxes
        self.transforms = transforms
        self.dataset_records_dataframe = dataset_records_dataframe
        self.det3d_supervised = det3d_supervised
        self.seg3d_supervised = seg3d_supervised

    def __len__(self) -> int:
        """Return the number of dataset records.

        Returns:
          int: Number of dataset records.
        """
        if self.dataset_records_dataframe is None:
            raise ValueError("Dataset records dataframe is not available.")
        return len(self.dataset_records_dataframe)

    def __getitem__(self, index: int) -> ModelGTSample:
        """Load and transform one dataset sample.

        Args:
            index: Sample index.

        Returns:
            Transformed ModelGTSample instance.
        """
        model_gt_sample = self.get_data_sample(index)
        return self.apply_transforms(model_gt_sample)

    def assign_dataset_records(self, dataset_records_dataframe: pl.DataFrame) -> None:
        """Assign the dataset records dataframe.

        Args:
            dataset_records_dataframe: Polars DataFrame of dataset records.
        """
        self.dataset_records_dataframe = dataset_records_dataframe

    @abstractmethod
    def get_data_sample(self, index: int) -> ModelGTSample:
        """Return raw metadata for a given dataset index.

        Args:
            index: Index of the sample.

        Returns:
            ModelGTSample instance consumed by the transform pipeline.
        """
        raise NotImplementedError("Dataset must implement get_data_sample")

    def apply_transforms(
        self,
        model_gt_sample: ModelGTSample,
    ) -> ModelGTSample:
        """Apply a specific transform pipeline to a metadata sample.

        Args:
            model_gt_sample: ModelGTSample instance.

        Returns:
            Transformed ModelGTSample instance.
        """
        if self.transforms is None:
            return model_gt_sample
        return self.transforms(model_gt_sample)

    def collate_fn(self, batch: Sequence[ModelGTSample]) -> ModelGTBatch:
        """
        Collate a batch of ModelGTSample into a ModelGTBatch.
        Args:
          batch: List of ModelGTSample instances to be collated.
        Returns:
          ModelGTBatch: Collated multi-task GT batch.
        """
        return ModelGTBatch.collate_gt_samples(
            gt_samples=batch, max_num_3d_gt_bboxes=self.max_num_3d_gt_bboxes
        )


class ConcatDataset(Dataset):
    """Serve one split assembled from several corpora.

    Every index maps to one record of one source dataset, and a source declaring a repeat
    contributes its records that many times, which is how a small rehearsal corpus keeps a
    share of the epoch next to a large one. The sources keep their own root directory,
    records and supervision, so mixing corpora needs nothing from the datasets themselves.
    """

    def __init__(self, datasets: Sequence[BaseDataset], repeats: Sequence[int]) -> None:
        """
        Initialize the concatenated dataset.

        Args:
          datasets: Dataset of every source of the split, in declaration order.
          repeats: How many times each source contributes its records to one epoch.
        """
        super().__init__()
        if len(datasets) != len(repeats):
            raise ValueError(
                f"Got {len(datasets)} source datasets and {len(repeats)} repeats, one repeat "
                "belongs to every source."
            )
        if not len(datasets):
            raise ValueError("A concatenated dataset serves at least one source.")
        self.datasets = tuple(datasets)
        self.index_map = tuple(
            (source_index, record_index)
            for source_index, (dataset, repeat) in enumerate(zip(datasets, repeats, strict=True))
            for _ in range(repeat)
            for record_index in range(len(dataset))
        )
        self.max_num_3d_gt_bboxes = max(dataset.max_num_3d_gt_bboxes for dataset in self.datasets)

    def __len__(self) -> int:
        """Return the number of samples of the split, repeated sources included.

        Returns:
          int: Number of samples.
        """
        return len(self.index_map)

    def __getitem__(self, index: int) -> ModelGTSample:
        """Load and transform the sample behind one index of the split.

        Args:
            index: Sample index.

        Returns:
            Transformed ModelGTSample instance.
        """
        source_index, record_index = self.index_map[index]
        return self.datasets[source_index][record_index]

    def collate_fn(self, batch: Sequence[ModelGTSample]) -> ModelGTBatch:
        """
        Collate a batch of ModelGTSample into a ModelGTBatch.

        A batch mixes the sources of the split, so it pads the boxes to the largest budget
        any source declares.

        Args:
          batch: List of ModelGTSample instances to be collated.

        Returns:
          ModelGTBatch: Collated multi-task GT batch.
        """
        return ModelGTBatch.collate_gt_samples(
            gt_samples=batch, max_num_3d_gt_bboxes=self.max_num_3d_gt_bboxes
        )
