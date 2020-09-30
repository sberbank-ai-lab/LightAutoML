#!/bin/bash

python3 -m venv lama_venv
source ./lama_venv/bin/activate

pip install -U poetry pip

# TODO: добавить сборку whl и выкладывания в Nexus
poetry lock
poetry install