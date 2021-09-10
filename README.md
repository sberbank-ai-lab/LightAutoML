# LightAutoML - automatic model creation framework

[![Slack](https://lightautoml-slack.herokuapp.com/badge.svg)](https://lightautoml-slack.herokuapp.com)
[![Telegram](https://img.shields.io/badge/chat-on%20Telegram-2ba2d9.svg)](https://t.me/lightautoml)
![PyPI - Downloads](https://img.shields.io/pypi/dm/lightautoml?color=green&label=PyPI%20downloads&logo=pypi&logoColor=orange&style=plastic)
![Read the Docs](https://img.shields.io/readthedocs/lightautoml?style=plastic)

LightAutoML (LAMA) - project from Sberbank AI Lab AutoML group is the framework for automatic classification and regression model creation.

Current available tasks to solve:
- binary classification
- multiclass classification
- regression

Currently we work with datasets, where **each row is an object with its specific features and target**. Multitable datasets and sequences are now under contruction :)

**Note**: for automatic creation of interpretable models we use [`AutoWoE`](https://github.com/sberbank-ai-lab/AutoMLWhitebox) library made by our group as well.

# Quick tour

Let's solve the popular Kaggle Titanic competition below. There are two main ways to solve machine learning problems using LightAutoML:
* Use ready preset for tabular data:

    ```python
    import pandas as pd
    from sklearn.metrics import f1_score

    from lightautoml.automl.presets.tabular_presets import TabularAutoML
    from lightautoml.tasks import Task

    df_train = pd.read_csv('../input/titanic/train.csv')
    df_test = pd.read_csv('../input/titanic/test.csv')

    automl = TabularAutoML(
        task = Task(
            name = 'binary',
            metric = lambda y_true, y_pred: f1_score(y_true, (y_pred > 0.5)*1))
    )
    oof_pred = automl.fit_predict(
        df_train,
        roles = {'target': 'Survived', 'drop': ['PassengerId']}
    )
    test_pred = automl.predict(df_test)

    pd.DataFrame({
        'PassengerId':df_test.PassengerId,
        'Survived': (test_pred.data[:, 0] > 0.5)*1
    }).to_csv('submit.csv', index = False)
    ```

* Build your own custom pipeline
    ```python
    import pandas as pd
    from sklearn.metrics import f1_score

    from lightautoml.automl.presets.tabular_presets import TabularAutoML
    from lightautoml.tasks import Task

    df_train = pd.read_csv('../input/titanic/train.csv')
    df_test = pd.read_csv('../input/titanic/test.csv')

    # define that machine learning problem is binary classification
    task = Task("binary")

    reader = PandasToPandasReader(task, cv=N_FOLDS, random_state=RANDOM_STATE)

    # create a feature selector
    model0 = BoostLGBM(
        default_params={'learning_rate': 0.05, 'num_leaves': 64, 'seed': 42, 'num_threads': N_THREADS}
    )
    pipe0 = LGBSimpleFeatures()
    mbie = ModelBasedImportanceEstimator()
    selector = ImportanceCutoffSelector(pipe0, model0, mbie, cutoff=0)

    # build first level pipeline for AutoML
    pipe = LGBSimpleFeatures()
    params_tuner1 = OptunaTuner(n_trials=20, timeout=30) # stop after 20 iterations or after 30 seconds
    model1 = BoostLGBM(
        default_params={'learning_rate': 0.05, 'num_leaves': 128, 'seed': 1, 'num_threads': N_THREADS}
    )
    model2 = BoostLGBM(
        default_params={'learning_rate': 0.025, 'num_leaves': 64, 'seed': 2, 'num_threads': N_THREADS}
    )
    pipeline_lvl1 = MLPipeline([
        (model1, params_tuner1),
        model2
    ], pre_selection=selector, features_pipeline=pipe, post_selection=None)

    # build second level pipeline for AutoML
    pipe1 = LGBSimpleFeatures()
    model = BoostLGBM(
        default_params={'learning_rate': 0.05, 'num_leaves': 64, 'max_bin': 1024, 'seed': 3, 'num_threads': N_THREADS},
        freeze_defaults=True
    )
    pipeline_lvl2 = MLPipeline([model], pre_selection=None, features_pipeline=pipe1, post_selection=None)

    # build AutoML pipeline
    automl = AutoML(reader, [
        [pipeline_lvl1],
        [pipeline_lvl2],
    ], skip_conn=False)

    # train AutoML and get predictions
    oof_pred = automl.fit_predict(df_train, roles = {'target': 'Survived', 'drop': ['PassengerId']})
    test_pred = automl.predict(df_test)

    pd.DataFrame({
        'PassengerId':df_test.PassengerId,
        'Survived': (test_pred.data[:, 0] > 0.5)*1
    }).to_csv('submit.csv', index = False)
    ```
LighAutoML framework has a lot of ready-to-use parts and extensive customization options, to learn more check out the [resources](#Resources) section.

# Resources
* Documentation of LightAutoML documentation is available [here](https://lightautoml.readthedocs.io/).

* Kaggle kernel examples of LightAutoML usage:
    - [Tabular Playground Series April 2021 competition solution](https://www.kaggle.com/alexryzhkov/n3-tps-april-21-lightautoml-starter)
    - [Titanic competition solution (80% accuracy)](https://www.kaggle.com/alexryzhkov/lightautoml-titanic-love)
    - [Titanic **12-code-lines** competition solution (78% accuracy)](https://www.kaggle.com/alexryzhkov/lightautoml-extreme-short-titanic-solution)
    - [House prices competition solution](https://www.kaggle.com/alexryzhkov/lightautoml-houseprices-love)
    - [Natural Language Processing with Disaster Tweets solution](https://www.kaggle.com/alexryzhkov/lightautoml-starter-nlp)
    - [Tabular Playground Series March 2021 competition solution](https://www.kaggle.com/alexryzhkov/lightautoml-starter-for-tabulardatamarch)
    - [Tabular Playground Series February 2021 competition solution](https://www.kaggle.com/alexryzhkov/lightautoml-tabulardata-love)
    - [Interpretable WhiteBox solution](https://www.kaggle.com/simakov/lama-whitebox-preset-example)
    - [Custom ML pipeline elements inside existing ones](https://www.kaggle.com/simakov/lama-custom-automl-pipeline-example)

* To find out how to work with LightAutoML, we have several tutorials and examples [here](examples/). Some of them you can run in Google Colab:

    - `Tutorial_1. Create your own pipeline.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sberbank-ai-lab/LightAutoML/blob/master/Tutorial_1.%20Create%20your%20own%20pipeline.ipynb) - shows how to create your own pipeline from specified blocks: pipelines for feature generation and feature selection, ML algorithms, hyperparameter optimization etc.
    - `Tutorial_2. AutoML pipeline preset.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sberbank-ai-lab/LightAutoML/blob/master/Tutorial_2.%20AutoML%20pipeline%20preset.ipynb) - shows how to use LightAutoML presets (both standalone and time utilized variants) for solving ML tasks on tabular data. Using presets you can solve binary classification, multiclass classification and regression tasks, changing the first argument in Task.
    - `Tutorial_3. Multiclass task.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sberbank-ai-lab/LightAutoML/blob/master/Tutorial_3.%20Multiclass%20task.ipynb) - shows how to build ML pipeline for multiclass ML task by hand
    - `Tutorial_4. SQL data source for pipeline preset.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sberbank-ai-lab/LightAutoML/blob/master/Tutorial_4.%20SQL%20data%20source%20for%20pipeline%20preset.ipynb) - shows how to use LightAutoML presets (both standalone and time utilized variants) for solving ML tasks on tabular data from SQL data base instead of CSV.

    Each tutorial has the step to enable Profiler and completes with Profiler run, which generates distribution for each function call time and shows it in interactive HTML report: the report show full time of run on its top and interactive tree of calls with percent of total time spent by the specific subtree.

    **Important 1**: for production you have no need to use profiler (which increase work time and memory consomption), so please do not turn it on - it is in off state by default

    **Important 2**: to take a look at this report after the run, please comment last line of demo with report deletion command.

* Video guides
    - (Russian) [LightAutoML webinar for Sberloga community](https://www.youtube.com/watch?v=ci8uqgWFJGg) ([Alexander Ryzhkov](https://kaggle.com/alexryzhkov), [Dmitry Simakov](https://kaggle.com/simakov))
    - (Russian) [LightAutoML hands-on tutorial in Kaggle Kernels](https://www.youtube.com/watch?v=TYu1UG-E9e8) ([Alexander Ryzhkov](https://kaggle.com/alexryzhkov))
    - (English) [Automated Machine Learning with LightAutoML: theory and practice](https://www.youtube.com/watch?v=4pbO673B9Oo) ([Alexander Ryzhkov](https://kaggle.com/alexryzhkov))
    - (English) [LightAutoML framework general overview, benchmarks and advantages for business](https://vimeo.com/485383651) ([Alexander Ryzhkov](https://kaggle.com/alexryzhkov))
    - (English) [LightAutoML practical guide - ML pipeline presets overview](https://vimeo.com/487166940) ([Dmitry Simakov](https://kaggle.com/simakov))

* Articles about LightAutoML
    - (English) [LightAutoML vs Titanic: 80% accuracy in several lines of code (Medium)](https://alexmryzhkov.medium.com/lightautoml-preset-usage-tutorial-2cce7da6f936)
    - (English) [Hands-On Python Guide to LightAutoML – An Automatic ML Model Creation Framework (Analytic Indian Mag)](https://analyticsindiamag.com/hands-on-python-guide-to-lama-an-automatic-ml-model-creation-framework/?fbclid=IwAR0f0cVgQWaLI60m1IHMD6VZfmKce0ZXxw-O8VRTdRALsKtty8a-ouJex7g)

# Installation
### Installation via pip from PyPI
To install LAMA framework on your machine:
```bash
pip install -U lightautoml
```
### Installation from sources with virtual environment creation
If you want to create a specific virtual environment for LAMA, you need to install  `python3-venv` system package and run the following command, which creates `lama_venv` virtual env with LAMA inside:
```bash
bash build_package.sh
```
To check this variant of installation and run all the demo scripts, use the command below:
```bash
bash test_package.sh
```
To install optional support for generating reports in pdf format run following commands:
```bash
# MacOS
brew install cairo pango gdk-pixbuf libffi

# Debian / Ubuntu
sudo apt-get install build-essential libcairo2 libpango-1.0-0 libpangocairo-1.0-0 libgdk-pixbuf2.0-0 libffi-dev shared-mime-info

# Fedora
sudo yum install redhat-rpm-config libffi-devel cairo pango gdk-pixbuf2

# Windows
# follow this tutorial https://weasyprint.readthedocs.io/en/stable/install.html#windows

poetry install -E pdf
```

# Contributing to LightAutoML
If you are interested in contributing to LightAutoML, please read the [Contributing Guide](.github/CONTRIBUTING.md) to get started.

Authors: [Alexander Ryzhkov](https://kaggle.com/alexryzhkov), [Anton Vakhrushev](https://kaggle.com/btbpanda), [Dmitry Simakov](https://kaggle.com/simakov), Vasilii Bunakov, Rinchin Damdinov, Pavel Shvets, Alexander Kirilin.
# Questions / Issues / Suggestions
Seek prompt advice at [Slack community](https://lightautoml-slack.herokuapp.com) or [Telegram group](https://t.me/lightautoml).

Open bug reports and feature requests on GitHub [issues](https://github.com/sberbank-ai-lab/LightAutoML/issues).

# Reference Papers
Anton Vakhrushev, Alexander Ryzhkov, Dmitry Simakov, Rinchin Damdinov, Maxim Savchenko, Alexander Tuzhilin ["LightAutoML: AutoML Solution for a Large Financial Services Ecosystem"](https://arxiv.org/pdf/2109.01528.pdf). arXiv:2109.01528, 2021.

# Licence
This project is licensed under the Apache License, Version 2.0. See [LICENSE](https://github.com/dev-rinchin/LightAutoML/blob/master/LICENSE) file for more details.
