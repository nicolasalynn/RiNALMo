#!/bin/bash
# Run TIS prediction training on Lambda Labs GPU
# Usage: bash lambda_train.sh

set -e

cd ~/RiNALMo
export PYTHONPATH=$HOME/RiNALMo:$PYTHONPATH

python train_tis_prediction.py \
    --data_dir ./data/tis_prediction \
    --pretrained_rinalmo_weights ./weights/rinalmo_giga_pretrained.pt \
    --output_dir ./outputs/tis \
    --max_seq_len 1022 \
    --target_block_size 400 \
    --batch_size 16 \
    --max_epochs 20 \
    --lr 1e-4 \
    --seed 42 \
    --num_workers 4 \
    --pin_memory \
    --tversky_alpha 0.3 \
    --tversky_beta 0.7 \
    --atg_lambda 1.0 \
    --wandb \
    --wandb_project rinalmo-tis \
    --wandb_experiment_name tis-giga-tversky-posmine-v4 \
    --checkpoint_every_epoch \
    --gradient_clip_val 1.0 \
    --log_every_n_steps 50

# --- Standalone dilated-conv architecture (no pretrained LM) ---
# python train_tis_prediction.py \
#     --architecture dilated_conv \
#     --data_dir ./data/tis_prediction \
#     --output_dir ./outputs/tis_dilated \
#     --max_seq_len 1022 \
#     --target_block_size 400 \
#     --batch_size 32 \
#     --max_epochs 50 \
#     --lr 3e-4 \
#     --seed 42 \
#     --num_workers 4 \
#     --pin_memory \
#     --conv_channels 256 \
#     --conv_kernel_size 9 \
#     --conv_dilations 1 2 4 8 16 32 64 128 \
#     --conv_dropout 0.1 \
#     --num_transformer_layers 2 \
#     --transformer_heads 8 \
#     --wandb \
#     --wandb_project rinalmo-tis \
#     --wandb_experiment_name tis-dilated-conv-v1 \
#     --checkpoint_every_epoch \
#     --gradient_clip_val 1.0 \
#     --log_every_n_steps 50
