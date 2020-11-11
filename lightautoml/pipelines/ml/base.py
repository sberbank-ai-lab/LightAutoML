from typing import Sequence, Optional, Tuple, Union

from log_calls import record_history

from lightautoml.validation.base import TrainValidIterator
from ..features.base import FeaturesPipeline, EmptyFeaturePipeline
from ..selection.base import SelectionPipeline, EmptySelector
from ...dataset.base import LAMLDataset

from ...dataset.utils import concatenate
from ...ml_algo.base import MLAlgo
from ...ml_algo.tuning.base import ParamsTuner, DefaultTuner
from ...ml_algo.utils import tune_and_fit_predict


@record_history(enabled=False)
class MLPipeline:
    """
    ML Pipeline contains 2 selection part (pre and post), preprocessing part and multiple ML parts.
    """

    def __init__(self, ml_algos: Sequence[Union[MLAlgo, Tuple[MLAlgo, ParamsTuner]]],
                 force_calc: Union[bool, Sequence[bool]] = True,
                 pre_selection: Optional[SelectionPipeline] = None,
                 features_pipeline: Optional[FeaturesPipeline] = None,
                 post_selection: Optional[SelectionPipeline] = None
                 ):

        """

        Args:
            ml_algos:  Sequence of MLAlgo's or Pair - (MlAlgo, ParamsTuner)
            force_calc: flag if single fold of ml_algo should be calculated anyway
            pre_selection: initial feature selection. If ``None`` there is no initial selection.
            features_pipeline: composition of feature transforms.
            post_selection: post feature selection. If ``None`` there is no post selection.

        """
        if pre_selection is None:
            pre_selection = EmptySelector()

        self.pre_selection = pre_selection

        if features_pipeline is None:
            features_pipeline = EmptyFeaturePipeline()

        self.features_pipeline = features_pipeline

        if post_selection is None:
            post_selection = EmptySelector()

        self.post_selection = post_selection

        self._ml_algos = []
        self.params_tuners = []

        for n, mt_pair in enumerate(ml_algos):

            try:
                # case when model and tuner are defined
                mod, tuner = mt_pair
            except (TypeError, ValueError):
                # case when only model is definded
                mod, tuner = mt_pair, DefaultTuner()

            mod.set_prefix('Mod_{0}'.format(n))

            self._ml_algos.append(mod)
            self.params_tuners.append(tuner)

        self.force_calc = [force_calc] * len(self._ml_algos) if type(force_calc) is bool else force_calc
        # TODO: Do we need this assert?
        assert any(self.force_calc), 'At least single algo in pipe should be forced to calc'

    def fit_predict(self, train_valid: TrainValidIterator) -> LAMLDataset:
        """
        Fit on train/valid iterator and transform on validation part.

        Args:
            train_valid: dataset iterator.

        Returns:
            dataset with predictions of all models.

        """
        self.ml_algos = []
        # train and apply pre selection
        train_valid = train_valid.apply_selector(self.pre_selection)

        # apply features pipeline
        train_valid = train_valid.apply_feature_pipeline(self.features_pipeline)

        # train and apply post selection
        train_valid = train_valid.apply_selector(self.post_selection)

        predictions = []

        for ml_algo, param_tuner, force_calc in zip(self._ml_algos, self.params_tuners, self.force_calc):
            ml_algo, preds = tune_and_fit_predict(ml_algo, param_tuner, train_valid, force_calc)
            if ml_algo is not None:
                self.ml_algos.append(ml_algo)

                predictions.append(preds)

        predictions = concatenate(predictions)

        return predictions

    def predict(self, dataset: LAMLDataset) -> LAMLDataset:
        """
        Predict on new dataset.

        Args:
            dataset: dataset used for prediction.

        Returns:
            dataset with predictions of all trained modles.

        """
        dataset = self.pre_selection.select(dataset)
        dataset = self.features_pipeline.transform(dataset)
        dataset = self.post_selection.select(dataset)

        predictions = []

        for model in self.ml_algos:
            pred = model.predict(dataset)
            predictions.append(pred)

        predictions = concatenate(predictions)

        return predictions

    def upd_model_names(self, prefix: str):
        """
        Update prefix pipeline models names.

        Args:
            prefix: new prefix name.

        """
        for n, mod in enumerate(self._ml_algos):
            mod.set_prefix(prefix)

    def prune_algos(self, idx: Sequence[int]):

        self.ml_algos = [x for (n, x) in enumerate(self.ml_algos) if n not in idx]
