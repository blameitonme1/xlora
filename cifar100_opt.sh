export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
CUDA_VISIBLE_DEVICES=3 python main_optuna.py \
    --model-name-or-path google/vit-base-patch16-224-in21k \
    --dataset-name cifar100 \
    --mode xlora \
    --n_frequency 3000 \
    --num_epochs 10 