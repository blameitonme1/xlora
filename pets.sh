export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
CUDA_VISIBLE_DEVICES=1 python main.py \
    --model-name-or-path google/vit-base-patch16-224-in21k \
    --dataset-name pets \
    --mode xlora \
    --n_frequency 3000 \
    --num_epochs 10 \
    --n_trial 1 \
    --head_lr 1e-3 \
    --weight_decay 8e-4 \
    --fft_lr 5e-3 \
    --share_entry 