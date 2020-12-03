#!/bin/bash

set -e
PACKAGE_NAME=LightAutoML_LAMA
source ./lama_venv/bin/activate

# Run demos
cd tests
pytest demo*
cd ..
