import pytest
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task


@pytest.mark.integtest
def test_default_tabular():
  # load and prepare data
  data = pd.read_csv("./data/sampled_app_train.csv")
  train_data, test_data = train_test_split(data, test_size=0.2, stratify=data["TARGET"], random_state=42)

  # run automl
  automl = TabularAutoML(task=Task("binary"))
  oof_predictions = automl.fit_predict(train_data, roles={"target": "TARGET", "drop": ["SK_ID_CURR"]})
  te_pred = automl.predict(test_data)

  # calculate scores
  print(f"Score for out-of-fold predictions: {roc_auc_score(train_data['TARGET'].values, oof_predictions.data[:, 0])}")
  print(f"Score for hold-out: {roc_auc_score(test_data['TARGET'].values, te_pred.data[:, 0])}")
