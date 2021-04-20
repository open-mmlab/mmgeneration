#!/usr/bin/env bash

set -x

CONFIG=$1
CKPT=$2
PY_ARGS=${@:3}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -u tools/evaluation.py ${CONFIG} ${CKPT} --launcher="none" ${PY_ARGS}
