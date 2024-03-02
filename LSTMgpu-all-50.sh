#!/bin/bash
#SBATCH --gpus=1
#SBATCH -p gpu_4090
module load cuda/12.1 anaconda/2021.05
source activate torch_env
python LSTMgpu-all-50.py