#!/bin/bash

python3 -m venv lama_venv
source ./lama_venv/bin/activate

pip install -U poetry pip

poetry lock
poetry install --no-dev
poetry build
