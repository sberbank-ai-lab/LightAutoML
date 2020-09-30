from typing import Tuple, Sequence, List, cast, Optional, Callable

import numpy as np
from log_calls import record_history
from scipy.optimize import minimize_scalar

from ..dataset.base import LAMLDataset
from ..dataset.np_pd_dataset import NumpyDataset
from ..dataset.roles import NumericRole
from ..pipelines.ml.base import MLPipeline


@record_history()
class Blender:
    """
    Basic class for blending
    Blender learns how to make blend on Sequence of prediction datasets and prune pipes, that are not used in final blend
    """

    _outp_dim = None

    @property
    def outp_dim(self) -> int:
        return self._outp_dim

    def fit_predict(self, predictions: Sequence[LAMLDataset], pipes: Sequence[MLPipeline]
                    ) -> Tuple[LAMLDataset, Sequence[MLPipeline]]:
        if len(pipes) == 1 and len(pipes[0].ml_algos) == 1:
            return predictions[0], pipes

        return self._fit_predict(predictions, pipes)

    def _fit_predict(self, predictions: Sequence[LAMLDataset], pipes: Sequence[MLPipeline]
                     ) -> Tuple[LAMLDataset, Sequence[MLPipeline]]:
        raise NotImplementedError

    def predict(self, predictions: Sequence[LAMLDataset]) -> LAMLDataset:

        if len(predictions) == 1:
            return predictions[0]

        return self._predict(predictions)

    def _predict(self, predictions: Sequence[LAMLDataset]) -> LAMLDataset:

        raise NotImplementedError

    def split_models(self, predictions: Sequence[LAMLDataset]) -> Tuple[Sequence[LAMLDataset], List[int], List[int]]:
        """
        Split predictions by single model

        Args:
            predictions:

        Returns:

        """
        splitted_preds = []
        model_idx = []
        pipe_idx = []

        for n, preds in enumerate(predictions):

            features = preds.features
            n_models = len(features) // self.outp_dim

            for k in range(n_models):
                curr_pred = preds[:, features[k * self.outp_dim: (k + 1) * self.outp_dim]]
                splitted_preds.append(curr_pred)
                model_idx.append(k)
                pipe_idx.append(n)

        return splitted_preds, model_idx, pipe_idx

    def _set_metadata(self, predictions: Sequence[LAMLDataset], pipes: Sequence[MLPipeline]):

        pred0 = predictions[0]
        pipe0 = pipes[0]

        self._outp_dim = pred0.shape[1] // len(pipe0.ml_algos)
        self._outp_prob = pred0.task.name in ['binary', 'multiclass']


@record_history()
class BestModelSelector(Blender):
    """
    Select best single model from level
    Drops pipes that are not used in calc best model
    """

    def _fit_predict(self, predictions: Sequence[LAMLDataset], pipes: Sequence[MLPipeline]
                     ) -> Tuple[LAMLDataset, Sequence[MLPipeline]]:
        """

        Args:
            predictions:
            pipes:

        Returns:

        """
        metric_func = pipes[0].ml_algos[0].score
        self._set_metadata(predictions, pipes)
        splitted_preds, model_idx, pipe_idx = self.split_models(predictions)

        best_pred = None
        best_pipe_idx = 0
        best_model_idx = 0
        best_score = -np.inf

        for pred, mod, pipe in zip(splitted_preds, model_idx, pipe_idx):

            score = metric_func(pred)

            if score > best_score:
                best_pipe_idx = pipe
                best_model_idx = mod
                best_score = score
                best_pred = pred

        best_pipe = pipes[best_pipe_idx]
        best_pipe.ml_algos = best_pipe.ml_algos[best_model_idx]

        return best_pred, [best_pipe]

    def _predict(self, predictions: Sequence[LAMLDataset]) -> LAMLDataset:
        """

        Args:
            predictions:

        Returns:

        """
        return predictions[0]


@record_history()
class MeanBlender(Blender):
    """
    Simple average level predictions
    """

    def _get_mean_pred(self, splitted_preds: Sequence[NumpyDataset]) -> NumpyDataset:
        """


        Args:
            splitted_preds:

        Returns:

        """
        outp = splitted_preds[0].empty()

        pred = np.nanmean([x.data for x in splitted_preds], axis=0)

        outp.set_data(pred, ['MeanBlend_{0}'.format(x) for x in range(pred.shape[0])],
                      NumericRole(np.float32, prob=self._outp_prob))

        return outp

    def _fit_predict(self, predictions: Sequence[NumpyDataset], pipes: Sequence[MLPipeline]
                     ) -> Tuple[NumpyDataset, Sequence[MLPipeline]]:
        """

        Args:
            predictions:
            pipes:

        Returns:

        """
        self._set_metadata(predictions, pipes)
        splitted_preds, _, __ = cast(List[NumpyDataset], self.split_models(predictions))

        outp = self._get_mean_pred(splitted_preds)

        return outp, pipes

    def _predict(self, predictions: Sequence[LAMLDataset]) -> LAMLDataset:
        """


        Args:
            predictions:

        Returns:

        """
        splitted_preds, _, __ = cast(List[NumpyDataset], self.split_models(predictions))
        outp = self._get_mean_pred(splitted_preds)

        return outp


@record_history()
class WeightedBlender(Blender):
    """
    Estimate weight to blend
    Weight sum eq. 1
    """

    def __init__(self, max_iters: int = 5, max_inner_iters: int = 7, max_nonzero_coef: float = 0.05):

        self.max_iters = max_iters
        self.max_inner_iters = max_inner_iters
        self.max_nonzero_coef = max_nonzero_coef
        self.wts = None

    def _get_weighted_pred(self, splitted_preds: Sequence[NumpyDataset], wts: Optional[np.ndarray]) -> NumpyDataset:
        length = len(splitted_preds)
        if wts is None:
            wts = np.ones(length, dtype=np.float32) / length

        weighted_pred = np.nansum([x.data * w for (x, w) in zip(splitted_preds, wts)], axis=0).astype(np.float32)
        not_nulls = np.sum([np.logical_not(np.isnan(x.data).any(axis=1)) for x in splitted_preds], axis=0).astype(
            np.float32) / length
        not_nulls = not_nulls[:, np.newaxis]

        weighted_pred /= not_nulls
        weighted_pred = np.where(not_nulls == 0, np.nan, weighted_pred)

        outp = splitted_preds[0].empty()
        outp.set_data(weighted_pred, ['WeightedBlend_{0}'.format(x) for x in range(weighted_pred.shape[1])],
                      NumericRole(np.float32, prob=self._outp_prob))

        return outp

    @staticmethod
    def _get_candidate(wts: np.ndarray, idx: int, value: float):

        candidate = wts.copy()
        sl = np.arange(wts.shape[0]) != idx
        s = candidate[sl].sum()
        candidate[sl] = candidate[sl] / s * (1 - value)
        candidate[idx] = value

        return candidate

    def _get_scorer(self, score_func: Callable, splitted_preds: Sequence[NumpyDataset], idx: int, wts: np.ndarray) -> Callable:

        def scorer(x):
            candidate = self._get_candidate(wts, idx, x)

            pred = self._get_weighted_pred(splitted_preds, candidate)
            score = score_func(pred)

            return -score

        return scorer

    def _optimize(self, score_func: Callable, splitted_preds: Sequence[NumpyDataset]) -> np.ndarray:

        length = len(splitted_preds)
        candidate = np.ones(length, dtype=np.float32) / length
        best_pred = self._get_weighted_pred(splitted_preds, candidate)

        best_score = score_func(best_pred)
        print('Blending: Optimization starts with equal weights and score {0}'.format(best_score))
        score = best_score
        for _ in range(self.max_iters):
            flg_no_upd = True
            for i in range(len(splitted_preds)):
                obj = self._get_scorer(score_func, splitted_preds, i, candidate)
                opt_res = minimize_scalar(obj, method='Bounded', bounds=(0, 1),
                                          options={'disp': False, 'maxiter': self.max_inner_iters})
                w = opt_res.x
                score = -opt_res.fun
                if score > best_score:
                    flg_no_upd = False
                    best_score = score
                    if w < self.max_nonzero_coef:
                        w = 0
                    candidate = self._get_candidate(candidate, i, w)

            print('Blending, iter {0}: score = {1}, weights = {2}'.format(_, score, candidate))

            if flg_no_upd:
                print('No score update. Terminated')
                break

        return candidate

    @staticmethod
    def _prune_pipe(pipes: Sequence[MLPipeline], wts: np.ndarray,
                    pipe_idx: np.ndarray) -> Tuple[Sequence[MLPipeline], np.ndarray]:
        """


        Args:
            pipes:
            wts:
            pipe_idx:

        Returns:

        """
        new_pipes = []

        for i in range(max(pipe_idx) + 1):
            pipe = pipes[i]
            weights = wts[np.array(pipe_idx) == i]

            pipe.ml_algos = [x for (x, w) in zip(pipe.ml_algos, weights) if w > 0]

            new_pipes.append(pipe)

        new_pipes = [x for x in new_pipes if len(x.ml_algos) > 0]
        wts = wts[wts > 0]
        return new_pipes, wts

    def _fit_predict(self, predictions: Sequence[NumpyDataset], pipes: Sequence[MLPipeline]
                     ) -> Tuple[NumpyDataset, Sequence[MLPipeline]]:
        """

        Args:
            predictions:
            pipes:

        Returns:

        """
        self._set_metadata(predictions, pipes)
        splitted_preds, _, pipe_idx = cast(List[NumpyDataset], self.split_models(predictions))
        score_func = pipes[0].ml_algos[0].score
        wts = self._optimize(score_func, splitted_preds)
        splitted_preds = [x for (x, w) in zip(splitted_preds, wts) if w > 0]
        pipes, self.wts = self._prune_pipe(pipes, wts, pipe_idx)

        outp = self._get_weighted_pred(splitted_preds, self.wts)

        return outp, pipes

    def _predict(self, predictions: Sequence[LAMLDataset]) -> LAMLDataset:
        """


        Args:
            predictions:

        Returns:

        """
        splitted_preds, _, __ = cast(List[NumpyDataset], self.split_models(predictions))
        outp = self._get_weighted_pred(splitted_preds, self.wts)

        return outp
