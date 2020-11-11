# LightAutoML (LAMA) - automatic model creation framework

[![Slack](https://lightautoml-slack.herokuapp.com/badge.svg)](https://lightautoml-slack.herokuapp.com)

LightAutoML (LAMA) project from Sberbank AI Lab - framework for automatic classification and regression model creation.

Current available tasks to solve:
- binary classification
- multiclass classification
- regression

Currently we work with datasets, where **each row is an object with its specific features and target**. Multitable datasets and sequences are now under contruction :)

**Authors**: Alexander Ryzhkov, Anton Vakhrushev

*******
# Installation
### Installation to the system python
To install LAMA framework on your machine:
```bash 
pip install -U poetry pip
poetry lock
poetry install
```
### Installation to the virtual environment
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
To generate documentation for LAMA framework, you can use command below:
```bash 
bash build_docs.sh
```
*******
# Usage examples

In `examples` folder you can find different demos of LAMA usage:
1. `demo0.py` - building ML pipeline from blocks and fit + predict the pipeline itself.
2. `demo1.py` - several ML pipelines creation (using importances based cutoff feature selector) to build 2 level stacking using AutoML class
3. `demo2.py` - several ML pipelines creation (using iteartive feature selection algorithm) to build 2 level stacking using AutoML class
4. `demo3.py` - several ML pipelines creation (using combination of cutoff and iterative FS algos) to build 2 level stacking using AutoML class
5. `demo4.py` - creation of classification and regression tasks for AutoML with loss and evaluation metric setup
6. `demo5.py` - 2 level stacking using AutoML class with different algos on first level including LGBM, Linear and LinearL1
7. `demo6.py` - **currently under construction** (AutoML with nested CV to recieve better results on small datasets)
8. `demo7.py` - AutoML preset usage for tabular datasets (predefined structure of AutoML pipeline and simple interface for users without building from blocks)
9. `demo8.py` - creation pipelines from blocks to build AutoML, solving multiclass classification task

Each example completes with Profiler run, which generates distribution for each function call time and shows it in interactive HTML report: the report show full time of run on its top and interactive tree of calls with percent of total time spent by the specific subtree. **Important**: to take a look at this report after the run, please comment last line of demo with report deletion command. 

*******
# Questions / Issues / Suggestions 

Write a message to us:
- Alexander Ryzhkov (_email_: AMRyzhkov@sberbank.ru, _tel._: +7-999-979-21-31, _telegram_: @RyzhkovAlex)
- Anton Vakhrushev (_email_: AGVakhrushev@sberbank.ru, _tel._: +7-916-263-62-14, _telegram_: using tel. number)





