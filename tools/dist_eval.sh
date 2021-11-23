#!/usr/bin/env bash

CONFIG=$1
CKPT=$2
GPUS=$3
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    tools/evaluation.py ${CONFIG} ${CKPT} --launcher pytorch ${@:4}
