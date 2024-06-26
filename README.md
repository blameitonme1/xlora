# Non-linear Low-rank Adaptation
Improve low-rank adaptation with non-linear property to better approximate the $\Delta W$, namely xLoRA.

# Environment setup
```bash
conda create -n xlora python=3.12
pip install evaluate
pip install peft
pip install transformer
```

# Train xlora on several vision tasks

There're different modes in xLoRA to allow flexible deployment of the adaptation:
```
xlora mode:
        1. vanilla xlora on mhsa q, v
        2. vanilla xlora on mhsa q, k, v
        3. vanilla xlora on ffn
        4. vanilla xlora on both mhsa q, v and ffn
        5. vanilla xlora on both mhsa q, k, v and ffn
```

Take a dataset like eurosat, just run
```bash
./eurosat.sh
```
In eurosat.sh, change the arguement to whichever mode or dimension of the adapter you like:
```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --model-name-or-path google/vit-base-patch16-224-in21k \
    --dataset-name eurosat \
    --mode xlora \
    --num_epochs 10 \
    --n_trial 1 \
    --head_lr 8e-4 \
    --weight_decay 3e-4 \
    --fft_lr learning_rate_you_want \
    --mhsa_dim dim_you_want_to_adapt_MHSA \
    --ffn_dim dim_you_want_to_adapt_FFN \
    --xlora_mode mode_you_want_when_deploying_adapters 
```