from pathlib import Path
import argparse

from image_datasets.capsSlicesIXI import CapsSlicesIXI
from torchvision import transforms
from torch.utils.data import DataLoader

from utils.config import get_configs_from_toml

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from model.unet import Unet
from torchinfo import summary

from diffusion.diffusion import DiffusionModel
from diffusion.sampler import save_sample


parser = argparse.ArgumentParser(description='Training diffusion model')
parser.add_argument(
    'model_number',
    help='Folder containing the toml config',
)
args = parser.parse_args()

experiment_dir = Path(
    f"/gpfswork/rech/krk/usy14zi/diffusion_models/medical_imaging_diffusion/experiments/model_{args.model_number}"
)
caps_dir = Path("/lustre/fswork/projects/rech/krk/commun/datasets/IXI/caps_IXI")

print(f"Working dir : {experiment_dir}")

### PARAMETERS
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 10

data_config, unet_config, diffusion_config = get_configs_from_toml(experiment_dir / "config.toml")

### DATASET
train_tsv = caps_dir / "IXI_train.tsv"
val_tsv = caps_dir / "IXI_validation.tsv"

transform=transforms.Compose([
    transforms.Pad([0, 18], fill=-1),
    transforms.Resize((unet_config.dim, unet_config.dim)),
])

train_set = CapsSlicesIXI(caps_dir, train_tsv, transform)
val_set = CapsSlicesIXI(caps_dir, val_tsv, transform)

print(f"Train set size: {len(train_set)}.")
print(f"Validation set size: {len(val_set)}.")

train_loader = DataLoader(
    train_set,
    **data_config.model_dump()
)
val_loader = DataLoader(
    val_set,
    **data_config.model_dump()
)

### DENOISER UNET
model = Unet(**unet_config.model_dump())
model.to(DEVICE)
summary(model)

### DIFFUSION MODEL
ddpm = DiffusionModel(model, diffusion_config, DEVICE, experiment_dir)

### TRAIN
optimizer = Adam(model.parameters(), lr=5e-4)
scheduler = ExponentialLR(optimizer, gamma=0.9)
ddpm.train(EPOCHS, optimizer, train_loader, val_loader, scheduler=scheduler)

### Generate images and save them
save_sample(ddpm, unet_config.dim, 10, unet_config.channels, experiment_dir)
