#!/bin/bash
# Lambda Labs setup script for RiNALMo TIS prediction training
# Usage: ssh into your Lambda instance, then:
#   bash lambda_setup.sh

set -e

echo "=== Setting up RiNALMo TIS training ==="

# 1. Clone repo and install
cd ~
git clone https://github.com/lbcb-sci/RiNALMo.git
cd RiNALMo

pip install -e .
pip install flash-attn==2.3.2 --no-build-isolation
pip install pytorch-lightning wandb torchmetrics pandas scikit-learn

# 2. Download pretrained weights
mkdir -p weights
cd weights
wget -q https://zenodo.org/records/10725749/files/rinalmo_giga_pretrained.pt
cd ..

# 3. Create data directory (you'll scp your data here)
mkdir -p data/tis_prediction
echo ">>> Data directory created at ~/RiNALMo/data/tis_prediction/"
echo ">>> SCP your data from your local machine:"
echo ">>>   scp data/tis_prediction/*.csv <lambda-user>@<lambda-ip>:~/RiNALMo/data/tis_prediction/"

# 4. wandb login
echo ""
echo "=== Now run: wandb login ==="
echo "=== Then paste your API key from https://wandb.ai/authorize ==="
