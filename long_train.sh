#!/bin/bash
#SBATCH --output=logs/%j.log
#SBATCH --constraint=v100-32g
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --account=krk@v100
#SBATCH --qos=qos_gpu-t4
#SBATCH --time=60:00:00

module load pytorch-gpu/py3/2.3.0

python main_dsb.py $1 ADNI_AD_CN
#python main_dsb.py $1 ADNI_hypo