#!/bin/bash
# Evaluate a TIS prediction checkpoint on the test set
# Usage: bash scripts/eval_tis.sh <path-to-checkpoint>
#
# Example:
#   bash scripts/eval_tis.sh ./outputs/tis_dilated/tis_pred-epoch_ckpt-epoch=19-step=XXXX.ckpt

set -e

CKPT=${1:?Usage: bash scripts/eval_tis.sh <path-to-checkpoint>}

cd ~/RiNALMo
export PYTHONPATH=$HOME/RiNALMo:$PYTHONPATH

python eval_tis_prediction.py \
    --checkpoint "$CKPT" \
    --test_data_dir ./data/tis_prediction \
    --max_seq_len 1022 \
    --target_block_size 400 \
    --batch_size 16 \
    --num_workers 4 \
    --pin_memory \
    --seed 42
