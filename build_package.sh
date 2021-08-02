#!/usr/bin/env bash

python3.8 -m venv lama_venv
source ./lama_venv/bin/activate

pip install -U pip
pip install -U poetry

poetry lock
poetry install -E all
poetry build
