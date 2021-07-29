#!/usr/bin/env bash

set -xe

PYTHON_EXE=python3
MAKE_ALL=false
POETRY_INSTALL_ARGS=""
INSTALL=false
BUILD_DIST=false
BUILD_DOCS=false
DEV_DEPS=false
FULL_DEPS=false

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
    echo "    -e|--extra <value>    Additioanl dependecies := [dev, nlp, cv, full]"
    echo "    -b|--dist    Build the source and wheels archives"
    echo "    -d|--docs    Build and check docs"
    echo "    -h|--help    Print help information"
    echo ""
}

EXTRA_FLAGS=("full" "dev" "cv" "nlp")  # (full deps, dev deps, [extra1 deps], [extra1 deps], ...)

containsElement () {
  local e match="$1"
  shift
  for e; do [[ "$e" == "$match" ]] && return 0; done
  return 1
}

function getFullDeps() {
    arraylength=${#EXTRA_FLAGS[@]}

    POETRY_INSTALL_ARGS=""

    # Ignore 'full' and 'dev'
    for (( i=2; i<${arraylength}; i++ )); do
        POETRY_INSTALL_ARGS+=" -E ${EXTRA_FLAGS[$i]} "
    done
}

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
            extra_flag=(${2-})

            containsElement $extra_flag "${EXTRA_FLAGS[@]}"

            if [[ $? == 1 ]]
                then
                echo "ERROR: Wrong extra deps values '$extra_flag', only '${EXTRA_FLAGS[@]}'"
                exit 1
            fi

            if [[ $extra_flag == "dev" ]]
                then
                DEV_DEPS=true
            elif [[ $extra_flag == "full" ]]
                then
                FULL_DEPS=true
            else
                POETRY_INSTALL_ARGS+=" -E $extra_flag "
            fi

            shift 2
            ;;
        -a|--all)
            MAKE_ALL=true
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

# Process docs building
if [[ ($DEV_DEPS = false) && ($BUILD_DOCS = false) ]]
    then
    POETRY_INSTALL_ARGS+=" --no-dev "
elif [[ ($DEV_DEPS = false) && ($BUILD_DOCS = true) ]]
    then
    echo "WARNING: Can't build docs without dev-deps. Dev-deps will be installed."
fi

if [[ $FULL_DEPS = true ]]
    then
    getFullDeps
fi

if [[ $MAKE_ALL = true ]]
    then
    INSTALL=true
    BUILD_DIST=true
    BUILD_DOCS=true
    getFullDeps
fi

# echo "--- TEST ---"
# echo "PYTHON_EXE: '$PYTHON_EXE'"
# echo "INSTALL: '$INSTALL'"
# echo "BUILD_DIST: '$BUILD_DIST'"
# echo "BUILD_DOCS: '$BUILD_DOCS'"
# echo "POETRY_INSTALL_ARGS: '$POETRY_INSTALL_ARGS'"
# echo "FULL_DEPS: $FULL_DEPS"
# echo "EXTRA_FLAGS: '${EXTRA_FLAGS[@]}'"

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
