PARTITION=$1
CHECKPOINT_DIR=$2

echo 'configs/styleganv2/stylegan2_c2_ffhq_256_b4x8_800k.py' &
# GPUS=1 for the test with data parallel
GPUS=1 bash tools/slurm_eval.sh ${PARTITION} test-benchmark \
    configs/styleganv2/stylegan2_c2_ffhq_256_b4x8_800k.py \
    ${CHECKPOINT_DIR}/stylegan2/stylegan2_c2_ffhq_256_b4x8_20210407_160709-7890ae1f.pth \
    --online &
