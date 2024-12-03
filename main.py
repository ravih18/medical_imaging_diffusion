from pathlib import Path
import toml

from image_datasets.capsSlicesIXI import CapsSlicesIXI
from image_datasets.transforms import ClipTensor
from torchvision import transforms
from torch.utils.data import DataLoader

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from model.unet import Unet
from torchinfo import summary

from utils.config import *

from diffusion.diffusion import DiffusionModel

experiment_directory = Path("/gpfswork/rech/krk/usy14zi/diffusion_models/medical_imaging_diffusion/experiments/model_1")

### PARAMETERS
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 500

config_toml = experiment_directory / "config.toml"
with open(config_toml, 'r') as f:
    config = toml.load(f)

data_config = DataConfig.parse_obj(config["Data"])
unet_config = UnetConfig.parse_obj(config["Unet"])
diffusion_config = DiffusionConfig.parse_obj(config["Diffusion"])

### DATASET
caps_dir = Path("/lustre/fswork/projects/rech/krk/commun/datasets/IXI/caps_IXI")
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
ddpm = DiffusionModel(model, diffusion_config, DEVICE, experiment_directory)

### TRAIN
optimizer = Adam(model.parameters(), lr=5e-4)
scheduler = ExponentialLR(optimizer, gamma=0.9)
ddpm.train(EPOCHS, optimizer, train_loader, val_loader, scheduler=scheduler)
