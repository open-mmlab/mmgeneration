PARTITION=$1
CHECKPOINT_DIR=$2

echo 'configs/styleganv2/stylegan2_c2_ffhq_256_b4x8_800k.py' &
# GPUS=1 for the test with data parallel
GPUS=1 bash tools/slurm_eval.sh ${PARTITION} test-benchmark \
    configs/styleganv2/stylegan2_c2_ffhq_256_b4x8_800k.py \
    ${CHECKPOINT_DIR}/stylegan2/stylegan2_c2_ffhq_256_b4x8_20210407_160709-7890ae1f.pth \
    --online &
y
echo 'configs/styleganv1/styleganv1_ffhq_256_g8_25Mimg.py' &
# GPUS=1 for the test with data parallel
GPUS=1 bash tools/slurm_eval.sh ${PARTITION} test-benchmark \
    configs/styleganv1/styleganv1_ffhq_256_g8_25Mimg.py \
    ${CHECKPOINT_DIR}/styleganv1/styleganv1_ffhq_1024_g8_25Mimg_20210407_161627-850a7234.pth \
    --online &
