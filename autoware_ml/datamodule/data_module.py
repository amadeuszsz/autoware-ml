import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import lightning as L
from torch.utils.data import DataLoader

from autoware_ml.databases.base_database import BaseDatabase
from autoware_ml.datamodule.base import DataLoaderConfig
from autoware_ml.datamodule.base_dataset import (
    BaseDataset,
    ConcatDataset,
)
from autoware_ml.datamodule.sources import DatasetSource, coerce_sources
from autoware_ml.datamodule.splitters.splitter_interface import SplitterInterface
from autoware_ml.types.dataset import SplitType

logger = logging.getLogger(__name__)


class DataModule(L.LightningDataModule):
    """Base LightningDataModule for multi-task learning that can be shared by multiple datasets.

    Every split declares the dataset sources it is served from. A split naming one source is
    the ordinary single corpus run, while a split naming several mixes them: each source keeps
    its own database, root directory and supervision, and its repeat decides the share of the
    epoch it takes. Preparation generates the record table of every distinct database, setup
    splits each table by the scenario lists of its own database, and every source gets a
    dataset of its own built from the split dataset factory.
    """

    def __init__(
        self,
        splitter: SplitterInterface,
        train_sources: Sequence[DatasetSource | Mapping[str, Any]] | None,
        validation_sources: Sequence[DatasetSource | Mapping[str, Any]] | None,
        test_sources: Sequence[DatasetSource | Mapping[str, Any]] | None,
        predict_sources: Sequence[DatasetSource | Mapping[str, Any]] | None,
        train_dataset: Callable[..., BaseDataset] | None,
        validation_dataset: Callable[..., BaseDataset] | None,
        test_dataset: Callable[..., BaseDataset] | None,
        predict_dataset: Callable[..., BaseDataset] | None,
        train_dataloader: DataLoaderConfig | None,
        validation_dataloader: DataLoaderConfig | None,
        test_dataloader: DataLoaderConfig | None,
        predict_dataloader: DataLoaderConfig | None,
    ) -> None:
        """
        Initialize the datamodule.

        Args:
          splitter: Splitter assigning the records of a database to its splits.
          train_sources: Dataset sources served by the training split, None when unused.
          validation_sources: Dataset sources served by the validation split, None when unused.
          test_sources: Dataset sources served by the test split, None when unused.
          predict_sources: Dataset sources served by the predict split, None when unused.
          train_dataset: Dataset factory of the training split, called once per source.
          validation_dataset: Dataset factory of the validation split.
          test_dataset: Dataset factory of the test split.
          predict_dataset: Dataset factory of the predict split.
          train_dataloader: Dataloader settings of the training split.
          validation_dataloader: Dataloader settings of the validation split.
          test_dataloader: Dataloader settings of the test split.
          predict_dataloader: Dataloader settings of the predict split.
        """
        super().__init__()

        self.splitter = splitter
        self.sources: dict[SplitType, tuple[DatasetSource, ...]] = {
            split: coerce_sources(sources)
            for split, sources in (
                (SplitType.TRAIN, train_sources),
                (SplitType.VAL, validation_sources),
                (SplitType.TEST, test_sources),
                (SplitType.PREDICT, predict_sources),
            )
            if sources is not None
        }
        self.dataset_factories: dict[SplitType, Callable[..., BaseDataset]] = {
            split: factory
            for split, factory in (
                (SplitType.TRAIN, train_dataset),
                (SplitType.VAL, validation_dataset),
                (SplitType.TEST, test_dataset),
                (SplitType.PREDICT, predict_dataset),
            )
            if factory is not None
        }
        self._validate_shared_taxonomy()
        self.datasets: dict[SplitType, ConcatDataset] = {}
        self.train_dataloader_config = train_dataloader
        self.validation_dataloader_config = validation_dataloader
        self.test_dataloader_config = test_dataloader
        self.predict_dataloader_config = predict_dataloader

    def _validate_shared_taxonomy(self) -> None:
        """
        Reject sources whose taxonomies disagree.

        The class index of a label is what the model learns, so corpora mixed into one run
        have to name their classes the same way. A disagreement would silently train two
        meanings into one output channel.
        """
        taxonomies = {
            str(source.database.taxonomy): source.database.version
            for sources in self.sources.values()
            for source in sources
        }
        if len(taxonomies) > 1:
            raise ValueError(
                "Every dataset source of a datamodule shares one taxonomy, got different ones "
                f"in the databases {sorted(taxonomies.values())}."
            )

    def databases(self) -> Sequence[BaseDatabase]:
        """
        Every distinct database of every split, in declaration order.

        Returns:
          Sequence[BaseDatabase]: Databases the datamodule reads, one per database hash.
        """
        databases: dict[str, BaseDatabase] = {}
        for sources in self.sources.values():
            for source in sources:
                databases.setdefault(source.database.database_hash, source.database)
        return tuple(databases.values())

    def setup(self, stage: str | None = None) -> None:
        """Build the dataset of every split the stage needs.

        Each database of a split is read and split once, and every source of the split gets a
        dataset of its own carrying that database's records, root directory and supervision.
        The datasets of a split are then concatenated in declaration order.

        Args:
            stage: Current stage ('fit', 'validate', 'test', 'predict') or
                ``None`` to prepare all splits.
        """
        stage_to_splits = {
            None: (SplitType.TRAIN, SplitType.VAL, SplitType.TEST, SplitType.PREDICT),
            "fit": (SplitType.TRAIN, SplitType.VAL),
            "validate": (SplitType.VAL,),
            "test": (SplitType.TEST,),
            "predict": (SplitType.PREDICT,),
        }

        split_records: dict[str, Mapping[SplitType, Any]] = {}
        for split in stage_to_splits[stage]:
            if split not in self.sources or split not in self.dataset_factories:
                logger.info(f"No dataset source declared for split {split}, skipping it.")
                continue

            datasets = []
            for source in self.sources[split]:
                database = source.database
                if database.database_hash not in split_records:
                    logger.info(f"Splitting the records of database {database.version}...")
                    split_records[database.database_hash] = self.splitter.split_by_polars_dataframe(
                        dataset_records_dataframe=database.load_polars_scenario_dataframe(),
                        scenarios=database.scenarios,
                    )
                records = split_records[database.database_hash][split]
                logger.info(
                    f"Serving {len(records)} records of database {database.version} to split "
                    f"{split}, det3d {source.det3d}, seg3d {source.seg3d}, "
                    f"repeated {source.repeat} times."
                )
                datasets.append(
                    self.dataset_factories[split](
                        database_root_path=str(database.root_path),
                        dataset_records_dataframe=records,
                        det3d_supervised=source.det3d,
                        seg3d_supervised=source.seg3d,
                    )
                )

            self.datasets[split] = ConcatDataset(
                datasets=datasets, repeats=[source.repeat for source in self.sources[split]]
            )
            logger.info(f"Split {split} serves {len(self.datasets[split])} samples.")

    def prepare_data(self) -> None:
        """
        Prepare the data for the multi-task learning.
        This method is called only in a single process from a main node.
        """
        logger.info("Preparing data for multi-task learning...")
        # Process the scenario records and create caches for every database that has none yet.
        for database in self.databases():
            database.process_scenario_records()
        logger.info("Finished preparing data for multi-task learning.")

    def build_dataloader(self, split: SplitType, config: DataLoaderConfig | None) -> DataLoader:
        """
        Build the dataloader of one split.

        Args:
          split: Split the dataloader serves.
          config: Dataloader settings of the split.

        Returns:
          DataLoader: Dataloader over the concatenated sources of the split.
        """
        if split not in self.datasets:
            raise ValueError(
                f"Split {split} has no dataset, declare its sources and its dataset factory "
                "before asking for its dataloader."
            )
        if config is None:
            raise ValueError(f"Split {split} has no dataloader settings.")

        dataset = self.datasets[split]
        return DataLoader(
            dataset=dataset,
            batch_size=config.batch_size,
            shuffle=config.shuffle,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            drop_last=config.drop_last,
            persistent_workers=config.persistent_workers,
            collate_fn=dataset.collate_fn,
        )

    def train_dataloader(self) -> DataLoader:
        """Create the dataloader of the training split."""
        return self.build_dataloader(SplitType.TRAIN, self.train_dataloader_config)

    def val_dataloader(self) -> DataLoader:
        """Create the dataloader of the validation split."""
        return self.build_dataloader(SplitType.VAL, self.validation_dataloader_config)

    def test_dataloader(self) -> DataLoader:
        """Create the dataloader of the test split."""
        return self.build_dataloader(SplitType.TEST, self.test_dataloader_config)

    def predict_dataloader(self) -> DataLoader:
        """Create the dataloader of the predict split."""
        return self.build_dataloader(SplitType.PREDICT, self.predict_dataloader_config)
