import pandas as pd
import pytest

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from lightautoml.automl.presets.whitebox_presets import WhiteBoxPreset


@pytest.mark.integtest
def tests_wb_preset(binary_task):
    # load and prepare data
    data = pd.read_csv("../../examples/data/jobs_train.csv")
    train, test = train_test_split(data.drop("enrollee_id", axis=1), test_size=0.2, stratify=data["target"])

    # run automl
    automl = WhiteBoxPreset(binary_task)
    _ = automl.fit_predict(train.reset_index(drop=True), roles={"target": "target"})
    test_prediction = automl.predict(test).data[:, 0]

    # calculate scores
    print(f"ROCAUC score: {roc_auc_score(test['target'].values, test_prediction)}")
