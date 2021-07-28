#!/usr/bin/env bash

set -xe

PYTHON_EXE=python3
ALL=false
POETRY_INSTALL_ARGS=""
INSTALL=false
BUILD_DIST=false
BUILD_DOCS=false
NO_DEV_DEPS=false

function usage()
{
    echo "OVERVIEW: LightAutoML(LAMA)"
    echo ""
    echo "USAGE: ./build.sh [options]"
    echo ""
    echo "OPTIONS:"
    echo "    -p|--python <PATH>    Path to python interpretator(default: 'python3')"
    echo "    -i|--install    Install library 'LAMA' without nlp/cv/dev dependecies, use -e flag for additional dependencies"
    echo "    -a|--all    Make all actions: install, build dist and docs (other option will be ignored)"
    echo "    -e|--extra <value>    Additioanl libs := [dev, nlp, cv]"
    echo "    -b|--dist    Build the source and wheels archives"
    echo "    -d|--docs    Build and check docs"
    echo "    -h|--help    Print help information"
    echo ""
}

EXTRA_FLAGS=("dev nlp cv")

params="$(getopt \
    -o p:i:abdh \
    -l python:,all,build,docs,help,install \
)"

while [ "$1" != "" ]
do
    case $1 in
        -p|--python)
            PYTHON_EXE=(${2-})
            shift 2
            ;;
        -i|--install)
            INSTALL=true
            shift
            ;;
        -e|--extra)
            EXTRA_FLAG=(${2-})

            if [[ ! " ${EXTRA_FLAGS[@]} " =~ " ${EXTRA_FLAG} " ]]
                then
                echo "ERROR: Wrong extra deps values '$EXTRA_FLAG', only '$EXTRA_FLAGS'"
                exit 1
            fi

            if [[ $EXTRA_FLAG == "dev" ]]
                then
                NO_DEV_DEPS=true
            else
                POETRY_INSTALL_ARGS+=" -E $EXTRA_FLAG "
            fi

            shift 2
            ;;
        -a|--all)
            ALL=true
            echo "WARNING: other options will be ignored (except: '-p|--python')"
            shift
            ;;
        -b|--dist)
            # Will override $default_excludes
            BUILD_DIST=true
            shift
            ;;
        -d|--docs)
            BUILD_DOCS=true
            shift
            ;;
        -h|--help)
            usage
            exit
            ;;
        *)
            echo "ERROR: unknown parameter '$1'"
            echo "Check help for using"
            exit
            ;;
    esac
done


if [[ ($NO_DEV_DEPS = false) && ($BUILD_DOCS = false) ]]
    then
    POETRY_INSTALL_ARGS+=" --no-dev "
elif [[ ($NO_DEV_DEPS = false) && ($BUILD_DOCS = true) ]]
    then
    echo "WARNING: Can't build docs without dev-deps. Dev-deps will be installed."
fi

if [[ $ALL = true ]]
    then
    INSTALL=true
    BUILD_DIST=true
    BUILD_DOCS=true
    POETRY_INSTALL_ARGS=" -E nlp -E cv "
fi

# echo "--- TEST ---"
# echo "PYTHON_EXE: '$PYTHON_EXE'"
# echo "INSTALL: '$INSTALL'"
# echo "BUILD_DIST: '$BUILD_DIST'"
# echo "BUILD_DOCS: '$BUILD_DOCS'"
# echo "POETRY_INSTALL_ARGS: '$POETRY_INSTALL_ARGS'"
# echo "EXTRA_FLAGS: '$EXTRA_FLAGS'"

if [[ $INSTALL = true ]]
    then
    $PYTHON_EXE -m venv lama_venv

    source ./lama_venv/bin/activate

    pip install -U pip
    pip install -U poetry

    poetry lock
    poetry install $POETRY_INSTALL_ARGS

    deactivate
fi


if [[ $BUILD_DOCS = true ]]
    then
    source ./lama_venv/bin/activate

    # Build docs
    cd docs
    mkdir _static
    make clean html
    cd ..

    echo "===== Start check_docs.py ====="
    python check_docs.py
fi

if [[ $BUILD_DIST = true ]]
    then
    source ./lama_venv/bin/activate

    rm -rf dist
    poetry build
fi
