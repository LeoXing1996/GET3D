#!/bin/sh
JOB_NAME=$1
# CONFIG=$2
WORK_DIR=$2
GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-16}
PY_ARGS=${@:5}
SRUN_ARGS=${SRUN_ARGS:-""}

DATA_ROOT=render_shapenet_data/shapenet-data
IMG_ROOT=${DATA_ROOT}/img
CAM_ROOT=${DATA_ROOT}/camera
CLASS_ID=03790512 # motorbike

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
    srun -p mm_lol \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u train_3d.py ${CONFIG} \
    --outdir=${WORK_DIR} \
    --data=${IMG_ROOT}/${CLASS_ID} \
    --camera_path=${CAM_ROOT} \
    --gpus=${GPUS} \
    --batch=32 \
    --gamma=80 \
    --data_camera_mode shapenet_motorbike \
    --dmtet_scale 1.0 \
    --use_shapenet_split 1 \
    --one_3d_generator 0 \
    --fp32 0 \
    --slurm ${PY_ARGS}
