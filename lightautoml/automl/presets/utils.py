import logging

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


def calc_one_feat_imp(iters, feat, model, data, norm_score, target, metric, silent):
    initial_col = data[feat].copy()
    data[feat] = np.random.permutation(data[feat].values)

    preds = model.predict(data)
    preds.target = data[target].values
    new_score = metric(preds)

    if not silent:
        logger.info3(
            "{}/{} Calculated score for {}: {:.7f}".format(
                iters[0], iters[1], feat, norm_score - new_score
            )
        )
    data[feat] = initial_col
    return feat, norm_score - new_score


def calc_feats_permutation_imps(model, used_feats, data, target, metric, silent=False):
    n_used_feats = len(used_feats)
    if not silent:
        logger.info3("LightAutoML used {} feats".format(n_used_feats))
    data = data.reset_index(drop=True)
    preds = model.predict(data)
    preds.target = data[target].values
    norm_score = metric(preds)
    feat_imp = []
    for it, f in enumerate(used_feats):
        feat_imp.append(
            calc_one_feat_imp(
                (it + 1, n_used_feats),
                f,
                model,
                data,
                norm_score,
                target,
                metric,
                silent,
            )
        )
    feat_imp = pd.DataFrame(feat_imp, columns=["Feature", "Importance"])
    feat_imp = feat_imp.sort_values("Importance", ascending=False).reset_index(
        drop=True
    )
    return feat_imp
