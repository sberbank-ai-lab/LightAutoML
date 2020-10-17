#!/bin/bash

python3 -m venv automl_venv
source ./automl_venv/bin/activate

pip install -U poetry pip

poetry lock
poetry install