import numpy as np
import polars as pl
import torch

from autoware_ml.databases.schemas.category_mapping import CategoryMappingDataModel
from autoware_ml.databases.schemas.dataset_schemas import DatasetTableSchema
from autoware_ml.databases.schemas.lidar_frames import LidarFrameDataModel
from autoware_ml.databases.taxonomy import LabelTaxonomy
from autoware_ml.datamodule.base_dataset_task import BaseDatasetTask
from autoware_ml.dataclasses.batch.sample_batch import ModelGTSample
from autoware_ml.dataclasses.batch.segmentation3d import Segmentation3DGTSample


class T4Segmentation3DTask(BaseDatasetTask):
    """
    Dataset task for 3D segmentation in the T4 dataset.
    This class defines how to process the dataset records for 3D segmentation in the T4 dataset and retrieve the necessary information for training and evaluation.
    """

    def __init__(
        self,
        database_root_path: str,
        dataset_records_dataframe: pl.DataFrame | None,
        taxonomy: LabelTaxonomy,
        semantic_mask_dtype: str = "uint8",
    ) -> None:
        """
        Initialize the T4Segmentation3DTask class.
        Args:
          database_root_path: Root directory of the dataset.
          dataset_records_dataframe: Polars DataFrame of dataset records to be processed for 3D segmentation in the T4 dataset.
          taxonomy: Taxonomy the categories of the semantic mask are resolved with.
          semantic_mask_dtype: Data type the raw category index of every point is stored with.
        """
        super().__init__(
            database_root_path=database_root_path,
            dataset_records_dataframe=dataset_records_dataframe,
        )
        self.taxonomy = taxonomy
        self.semantic_mask_dtype = np.dtype(semantic_mask_dtype)

    def pre_filter_dataset_records(
        self, dataset_records_dataframe: pl.DataFrame | None
    ) -> pl.DataFrame | None:
        """
        Pre-filter the dataset records dataframe for 3D segmentation in the T4 dataset.
        This method filters the dataset records dataframe to only include columns related to the
        semantic mask of the points.

        Args:
          dataset_records_dataframe: Polars DataFrame of dataset records to be filtered for 3D segmentation in the T4 dataset.
        Returns:
          Polars DataFrame of filtered dataset records for 3D segmentation in the T4 dataset
        """
        if dataset_records_dataframe is None:
            return None

        return dataset_records_dataframe.select(
            [
                DatasetTableSchema.LIDAR_FRAMES.name,
                DatasetTableSchema.CATEGORY_MAPPING.name,
            ]
        )

    def __str__(self) -> str:
        """
        String representation of the dataset type.

        Returns:
          str: String representation of the dataset type.
        """
        return "T4Segmentation3DTask"

    def get_data_sample(self, idx: int) -> ModelGTSample:
        """
        Process the dataset records dataframe for 3D segmentation in the T4 dataset. The semantic
        mask of the current lidar frame holds one raw category index per point, the category
        mapping of the record names every raw index, and the taxonomy resolves every name to its
        class index.

        Args:
          idx: Index of the specific record to be processed.

        Returns:
          ModelGTSample: Multi-task data row holding the semantic labels of the current frame.
        """
        if self.dataset_records_dataframe is None:
            raise ValueError("Dataset records dataframe is not available.")

        lidar_frame = LidarFrameDataModel.load_from_dictionary(
            self.dataset_records_dataframe.item(idx, DatasetTableSchema.LIDAR_FRAMES.name)[0]
        )
        relative_path = lidar_frame.lidarseg_pointcloud_semantic_mask_relative_path
        if relative_path is None:
            raise ValueError(
                f"The lidar frame {lidar_frame.lidar_frame_id} of record {idx} carries no "
                "semantic mask path, so 3D segmentation cannot run on it."
            )

        raw_labels = np.fromfile(
            str(self.database_root_path / relative_path), dtype=self.semantic_mask_dtype
        ).astype(np.int64)
        category_mapping = CategoryMappingDataModel.load_from_dictionary(
            self.dataset_records_dataframe.item(idx, DatasetTableSchema.CATEGORY_MAPPING.name)
        )

        return ModelGTSample(
            lidar_point_cloud_samples=None,
            image_samples=None,
            point_cloud_data=None,
            camera_image_data=None,
            detection3d_gt_bboxes_3d=None,
            segmentation3d_gt_sample=Segmentation3DGTSample(
                gt_semantic_mask=torch.from_numpy(
                    self.resolve_class_indices(raw_labels, category_mapping)
                ),
                ignore_index=self.taxonomy.ignore_index,
            ),
        )

    def resolve_class_indices(
        self,
        raw_labels: np.ndarray,
        category_mapping: CategoryMappingDataModel,
    ) -> np.ndarray:
        """
        Resolve the raw category index of every point to the class index of the taxonomy.

        Args:
          raw_labels: Raw category index of every point of the current frame.
          category_mapping: Category mapping of the record, naming every raw category index.

        Returns:
          np.ndarray: Class index of every point of the current frame.
        """
        lookup = np.full(
            max(category_mapping.category_indices, default=-1) + 1,
            self.taxonomy.ignore_index,
            dtype=np.int64,
        )
        named = np.zeros(lookup.shape[0], dtype=np.bool_)
        for category_name, category_index in zip(
            category_mapping.category_names, category_mapping.category_indices
        ):
            lookup[category_index] = self.taxonomy.resolve_index(category_name)
            named[category_index] = True

        in_range = (raw_labels >= 0) & (raw_labels < lookup.shape[0])
        known = in_range.copy()
        known[in_range] = named[raw_labels[in_range]]
        if not known.all():
            raise ValueError(
                "The semantic mask carries the raw category indices "
                f"{sorted(set(raw_labels[~known].tolist()))}, which the category mapping of the "
                "record does not name."
            )
        return lookup[raw_labels]
