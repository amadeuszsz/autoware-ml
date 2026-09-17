from typing import Mapping, Sequence

from autoware_ml.databases.box3d_pipelines.box3d_pipeline import Box3DPipeline
from autoware_ml.databases.schemas.box3d_schemas import Box3DDataModel
from autoware_ml.databases.taxonomy import LabelTaxonomy


class Box3DLabelRemapper(Box3DPipeline):
    """
    Pipeline to remap the label names and indices of the 3D bounding boxes to another label name.
    It reads the name the previous pipeline left, so several entries compose: the first turns the
    raw dataset name into the fine name of the vocabulary, a later one folds the labels a merger
    left behind. A name the remapper maps to null drops its box, and a name the taxonomy does not
    name takes the ignore label index.
    """

    def __init__(
        self,
        taxonomy: LabelTaxonomy,
        label_remapper: Mapping[str, str | None] | None = None,
    ):
        """
        Initialize Box3DLabelRemapper.

        Args:
          taxonomy: Taxonomy the boxes are baked with, giving the class index of a label name.
          label_remapper: Mapping to remap label names, null for a name to drop. Defaults to the
            vocabulary of the taxonomy, which turns raw dataset names into fine names.
        """
        super().__init__()
        self.taxonomy = taxonomy
        self.label_remapper = dict(
            taxonomy.vocabulary.class_renaming if label_remapper is None else label_remapper
        )

    def __str__(self) -> str:
        """
        String representation of the pipeline, used for logging.

        Returns:
          str: String representation of the pipeline.
        """
        return f"{self.__class__.__name__}(label_remapper={self.label_remapper})"

    def __call__(self, boxes3d_data_model: Sequence[Box3DDataModel]) -> Sequence[Box3DDataModel]:
        """
        Remap the label names of the 3D bounding boxes to another label name.
        """
        new_boxes3d_data_model = []
        for box3d_data_model in boxes3d_data_model:
            new_box3d_label_name = self.label_remapper.get(
                box3d_data_model.box3d_label_name, box3d_data_model.box3d_label_name
            )
            if new_box3d_label_name is None:
                continue

            new_boxes3d_data_model.append(
                box3d_data_model.create_new_data_model(
                    box3d_label_name=new_box3d_label_name,
                    box3d_label_index=self.taxonomy.class_index(new_box3d_label_name),
                )
            )

        return new_boxes3d_data_model
