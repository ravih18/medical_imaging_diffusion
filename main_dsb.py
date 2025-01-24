import toml
from pathlib import Path
import argparse

from utils.config import *
import torch
from diffusion.dsb import IPF

parser = argparse.ArgumentParser(description='Training DSB model')
parser.add_argument(
    'model_number',
    help='Folder containing the toml config',
)
args = parser.parse_args()

caps_dir = Path("/lustre/fswork/projects/rech/krk/commun/datasets/IXI/caps_IXI")
expe_dir = Path(f"experiments/dsb{args.model_number}")

print(f"Working dir : {expe_dir}")

transfer = True
dsb = IPF(caps_dir, expe_dir, transfer)
print(dsb.device)

dsb.train()