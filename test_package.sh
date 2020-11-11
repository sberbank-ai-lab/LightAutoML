#!/bin/bash

set -e
PACKAGE_NAME=LightAutoML_LAMA
source ./lama_venv/bin/activate

# Run demos
cd examples

for file in `ls *.py | sort -V`
do
  echo "===== Start  ${file} ====="
  python "$file"
done

cd ..
