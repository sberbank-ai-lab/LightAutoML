from typing import Optional, TypeVar

from log_calls import record_history

from lightautoml.validation.base import TrainValidIterator
from .base import SelectionPipeline, ImportanceEstimator
from ..features.base import FeaturesPipeline
from ...dataset.base import LAMLDataset
from ...ml_algo.base import MLAlgo

ImportanceEstimatedAlgo = TypeVar('ImportanceEstimatedAlgo', bound=ImportanceEstimator)


@record_history(enabled=False)
class ModelBasedImportanceEstimator(ImportanceEstimator):
    """
    Base class for performing feature selection according using feature importance.

    """

    def fit(self, train_valid: Optional[TrainValidIterator] = None,
            ml_algo: Optional[ImportanceEstimatedAlgo] = None,
            preds: Optional[LAMLDataset] = None):
        """
        Find the importances of features.

        Args:
            train_valid: dataset iterator.
            ml_algo: ML algorithm used for importance estimation.
            preds: predicted target values.

        """
        assert ml_algo is not None, 'ModelBasedImportanceEstimator: raw importances are None and no MLAlgo to calculate them.'
        self.raw_importances = ml_algo.get_features_score()


@record_history(enabled=False)
class ImportanceCutoffSelector(SelectionPipeline):
    """
    Selector based on importance treshold.
    Data passed to .fit should be ok to fit ml_algo or preprocessing pipeline should be defined.

    """

    def __init__(self, feature_pipeline: Optional[FeaturesPipeline],
                 ml_algo: MLAlgo,
                 imp_estimator: ImportanceEstimator,
                 fit_on_holdout: bool = True,
                 cutoff: float = 0.0):
        """
        Args:
            feature_pipeline: composition of feature transforms.
            ml_algo: Tuple (MlAlgo, ParamsTuner).
            imp_estimator: feature importance estimator.
            fit_on_holdout: if use the holdout iterator.
            cutoff: threshold to cut-off features.

        """
        super().__init__(feature_pipeline, ml_algo, imp_estimator, fit_on_holdout)
        self.cutoff = cutoff

    def perform_selection(self, train_valid: Optional[TrainValidIterator] = None):
        """
        Select features.

        Args:
            train_valid: ignored.

        """
        imp = self.imp_estimator.get_features_score()
        self.map_raw_feature_importances(imp)
        selected = self.mapped_importances.index.values[self.mapped_importances.values > self.cutoff]
        self._selected_features = list(selected)
