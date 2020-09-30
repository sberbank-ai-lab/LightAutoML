#!/bin/bash

set -e
PACKAGE_NAME=LightAutoML_LAMA
source ./venv/bin/activate

#Build docs
cd docs
mkdir -p _static
make clean html
cd ..


echo "===== Start check_docs.py ====="
python check_docs.py
echo "===== Start demo.py ====="
python demo.py
echo "===== Start demo1.py ====="
python demo1.py
echo "===== Start demo2.py ====="
python demo2.py
echo "===== Start demo3.py ====="
python demo3.py
echo "===== Start demo4.py ====="
python demo4.py
echo "===== Start demo5.py ====="
python demo5.py
echo "===== Start demo6.py ====="
python demo6.py
echo "===== Start demo7.py ====="
python demo7.py
echo "===== Start demo8.py ====="
python demo8.py