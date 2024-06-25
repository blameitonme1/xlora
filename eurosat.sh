export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
CUDA_VISIBLE_DEVICES=0 python main.py \
    --model-name-or-path google/vit-base-patch16-224-in21k \
    --dataset-name eurosat \
    --mode xlora \
    --n_frequency 3000 \
    --num_epochs 10 \
    --n_trial 1 \
    --head_lr 8e-4 \
    --weight_decay 3e-4 \
    --fft_lr 2e-2 \
    --mhsa_dim 16 \
    --ffn_dim 4 \
    --xlora_mode 3 