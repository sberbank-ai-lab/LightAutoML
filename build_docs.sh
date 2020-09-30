#!/bin/bash

set -e
PACKAGE_NAME=LightAutoML_LAMA
source ./lama_venv/bin/activate

#Build docs
cd docs
mkdir -p _static
make clean html
cd ..