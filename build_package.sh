#!/bin/bash

python3 -m venv venv
source ./venv/bin/activate

pip install --index-url http://mirror.sigma.sbrf.ru/pypi/simple \
    --trusted-host mirror.sigma.sbrf.ru -U poetry pip

# TODO: добавить сборку whl и выкладывания в Nexus
poetry lock
poetry install