import math
import numpy as np
from pandas import concat, Series
from autogluon.core.metrics.softclass_metrics import soft_log_loss


class CustomMetric:
    def __init__(self, metric, is_higher_better, needs_pred_proba):
        self.metric = metric
        self.is_higher_better = is_higher_better
        self.needs_pred_proba = needs_pred_proba

    @staticmethod
    def get_final_error(error, weight):
        return error

    def is_max_optimal(self):
        return self.is_higher_better

    def evaluate(self, approxes, target, weight):
        raise NotImplementedError


class SoftclassCustomMetric(CustomMetric):
    from catboost import MultiRegressionCustomMetric
    def __init__(self, metric, is_higher_better, needs_pred_proba):  # metric is ignored
        super().__init__(metric, is_higher_better, needs_pred_proba)
        self.softlogloss = self.SoftLogLossMetric()  # the metric object to pass to CatBoostRegressor

    def evaluate(self, approxes, target, weight):
        return self.softlogloss.evaluate(approxes, target, weight)

    class SoftLogLossMetric(MultiRegressionCustomMetric):
        def get_final_error(self, error, weight):
            return error

        def is_max_optimal(self):
            return True

        def evaluate(self, approxes, target, weight):
            assert len(target) == len(approxes)
            assert len(target[0]) == len(approxes[0])
            weight_sum = len(target)
            approxes = np.array(approxes)
            approxes = np.exp(approxes)
            approxes = np.multiply(approxes, 1/np.sum(approxes, axis=1)[:, np.newaxis])
            error_sum = soft_log_loss(np.array(target), np.array(approxes))
            return error_sum, weight_sum


class SoftclassObjective(object):
    from catboost import MultiRegressionCustomObjective
    def __init__(self):
        self.softlogloss = self.SoftLogLossObjective()  # the objective object to pass to CatBoostRegressor

    class SoftLogLossObjective(MultiRegressionCustomObjective):
        def calc_ders_multi(self, approxes, targets, weight):
            exp_approx = [math.exp(val) for val in approxes]
            exp_sum = sum(exp_approx)
            exp_approx = [val / exp_sum for val in exp_approx]
            grad = [(targets[j] - exp_approx[j])*weight for j in range(len(targets))]
            hess = [[(exp_approx[j] * exp_approx[j2] - (j==j2)*exp_approx[j]) * weight
                    for j in range(len(targets))] for j2 in range(len(targets))]
            return (grad, hess)


def spunge_augment(X,
                   num_augmented_samples=10000,
                   frac_perturb=0.1,
                   continuous_feature_noise=0.1,
                   **kwargs):
    num_feature_perturb = max(1, int(frac_perturb*len(X.columns)))
    X_aug = concat([X.iloc[[0]].copy()]*num_augmented_samples)
    X_aug.reset_index(drop=True, inplace=True)
    continuous_types = ['float', 'int']
    continuous_featnames = X.select_dtypes(continuous_types).columns

    for i in range(num_augmented_samples): # hot-deck sample some features per datapoint
        og_ind = i % len(X)
        augdata_i = X.iloc[og_ind].copy()
        num_feature_perturb_i = np.random.choice(range(1,num_feature_perturb+1))  # randomly sample number of features to perturb
        cols_toperturb = np.random.choice(list(X.columns), size=num_feature_perturb_i, replace=False)
        for feature in cols_toperturb:
            feature_data = X[feature]
            augdata_i[feature] = feature_data.sample(n=1).values[0]
        X_aug.iloc[i] = augdata_i

    for feature in X.columns:
        if feature in continuous_featnames:
            feature_data = X[feature]
            aug_data = X_aug[feature]
            noise = np.random.normal(scale=np.nanstd(feature_data)*continuous_feature_noise, size=num_augmented_samples)
            mask = np.random.binomial(n=1, p=frac_perturb, size=num_augmented_samples)
            aug_data = aug_data + noise*mask
            X_aug[feature] = Series(aug_data, index=X_aug.index)

    return concat((X, X_aug))
