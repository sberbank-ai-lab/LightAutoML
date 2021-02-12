# LightAutoML (LAMA) - automatic model creation framework

[![Slack](https://lightautoml-slack.herokuapp.com/badge.svg)](https://lightautoml-slack.herokuapp.com)
![PyPI - Downloads](https://img.shields.io/pypi/dm/lightautoml?color=green&label=PyPI%20downloads&logo=pypi&logoColor=orange&style=plastic)
![Read the Docs](https://img.shields.io/readthedocs/lightautoml?style=plastic)

LightAutoML (LAMA) project from Sberbank AI Lab AutoML group is the framework for automatic classification and regression model creation.

Current available tasks to solve:
- binary classification
- multiclass classification
- regression

Currently we work with datasets, where **each row is an object with its specific features and target**. Multitable datasets and sequences are now under contruction :)

**Note**: for automatic creation of interpretable models we use [`AutoWoE`](https://github.com/sberbank-ai-lab/AutoMLWhitebox) library made by our group as well.

**Authors**: Ryzhkov Alexander, Vakhrushev Anton, Simakov Dmitry, Bunakov Vasilii, Damdinov Rinchin, Shvets Pavel, Kirilin Alexander

**LAMA video guides**:
- [LAMA framework general overview, benchmarks and advantages for business](https://vimeo.com/485383651) (Ryzhkov Alexander)
- [LAMA practical guide - ML pipeline presets overview](https://vimeo.com/487166940) (Simakov Dmitry)

See the [Documentation of LightAutoML](https://lightautoml.readthedocs.io/).

*******
# Installation
### Installation via pip from PyPI
To install LAMA framework on your machine:
```bash 
pip install lightautoml
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
*******
# Docs generation
To generate documentation for LAMA framework, you can use command below (it uses virtual env created on installation step from sources):
```bash 
bash build_docs.sh
```

Builded official documentation for LightAutoML is available [`here`](https://lightautoml.readthedocs.io/en/latest/).
*******
# Usage examples

To find out how to work with LightAutoML, we have several tutorials:
1. `Tutorial_1. Create your own pipeline.ipynb` - shows how to create your own pipeline from specified blocks: pipelines for feature generation and feature selection, ML algorithms, hyperparameter optimization etc.
2. `Tutorial_2. AutoML pipeline preset.ipynb` - shows how to use LightAutoML presets (both standalone and time utilized variants) for solving ML tasks on tabular data. Using presets you can solve binary classification, multiclass classification and regression tasks, changing the first argument in Task.
3. `Tutorial_3. Multiclass task.ipynb` - shows how to build ML pipeline for multiclass ML task by hand

Each tutorial has the step to enable Profiler and completes with Profiler run, which generates distribution for each function call time and shows it in interactive HTML report: the report show full time of run on its top and interactive tree of calls with percent of total time spent by the specific subtree.

**Important 1**: for production you have no need to use profiler (which increase work time and memory consomption), so please do not turn it on - it is in off state by default

**Important 2**: to take a look at this report after the run, please comment last line of demo with report deletion command. 

For more examples, in `tests` folder you can find different scenarios of LAMA usage:
1. `demo0.py` - building ML pipeline from blocks and fit + predict the pipeline itself.
2. `demo1.py` - several ML pipelines creation (using importances based cutoff feature selector) to build 2 level stacking using AutoML class
3. `demo2.py` - several ML pipelines creation (using iteartive feature selection algorithm) to build 2 level stacking using AutoML class
4. `demo3.py` - several ML pipelines creation (using combination of cutoff and iterative FS algos) to build 2 level stacking using AutoML class
5. `demo4.py` - creation of classification and regression tasks for AutoML with loss and evaluation metric setup
6. `demo5.py` - 2 level stacking using AutoML class with different algos on first level including LGBM, Linear and LinearL1
7. `demo6.py` - AutoML with nested CV usage
8. `demo7.py` - AutoML preset usage for tabular datasets (predefined structure of AutoML pipeline and simple interface for users without building from blocks)
9. `demo8.py` - creation pipelines from blocks to build AutoML, solving multiclass classification task
10. `demo9.py` - AutoML time utilization preset usage for tabular datasets (predefined structure of AutoML pipeline and simple interface for users without building from blocks)
11. `demo10.py` - creation pipelines from blocks (including CatBoost) to build AutoML , solving multiclass classification task
12. `demo11.py` - AutoML NLP preset usage for tabular datasets with text columns
13. `demo12.py` - AutoML tabular preset usage with custom validation scheme and multiprocessed inference


******
# Contributing to LightAutoML

If you are interested in contributing to LightAutoML, please read the [Contributing Guide](CONTRIBUTING.md) to get started.


*******
# Questions / Issues / Suggestions 

Write a message to us:
- Alexander Ryzhkov (_email_: AMRyzhkov@sberbank.ru, _telegram_: @RyzhkovAlex)
- Anton Vakhrushev (_email_: AGVakhrushev@sberbank.ru)

