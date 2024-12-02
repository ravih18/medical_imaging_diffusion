from pathlib import Path
from image_datasets.capsSlicesIXI import CapsSlicesIXI
from image_datasets.transforms import ClipTensor
from torchvision import transforms
from torch.utils.data import DataLoader

import torch
from torch.optim import Adam
from model.unet import Unet
from torchinfo import summary

from diffusion.time_scheduler import quadratic_beta_schedule
from diffusion.diffusion import DiffusionModel

from datetime import datetime

### PARAMETERS
caps_dir = Path("/lustre/fswork/projects/rech/krk/commun/datasets/IXI/caps_IXI")
train_tsv = caps_dir / "IXI_train.tsv"
val_tsv = caps_dir / "IXI_validation.tsv"

BATCH_SIZE = 64
TIMESTEPS = 2000
EPOCHS = 300

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

### DATASET
transform=transforms.Compose([
    transforms.Pad([0, 18], fill=0),
    ClipTensor(),
    transforms.Resize((64, 64)),
    transforms.Lambda(lambda t: 2 * (t-t.min())/(t.max()-t.min()) - 1),    # Image range between [-1, 1]
])
train_set = CapsSlicesIXI(caps_dir, train_tsv, transform)
val_set = CapsSlicesIXI(caps_dir, val_tsv, transform)

channels, image_length, image_width  = train_set[0]['T1'].shape
print(f"Train set size: {len(train_set)}.")
print(f"Validation set size: {len(val_set)}.")
print(f"Image of size {image_length}x{image_width}, with {channels} channel(s).")
image_size = image_length

train_loader = DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=10,
    prefetch_factor=2,
    drop_last=True,
)
val_loader = DataLoader(
    val_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=8,
    drop_last=True,
)

### DENOISER UNET
model = Unet(
    dim=image_size,
    channels=channels,
    dim_mults=(1, 2, 2,),
    n_block_klass = 2,
)
model.to(DEVICE)
summary(model)

### DIFFUSION MODEL
betas = quadratic_beta_schedule(TIMESTEPS, beta_start=0.00001, beta_end=0.01)
ddpm = DiffusionModel(model, TIMESTEPS, betas, DEVICE, loss_type='huber')

### TRAIN
optimizer = Adam(model.parameters(), lr=3e-4)
ddpm.train(EPOCHS, optimizer, train_loader, val_loader)

# Save Unet
date = datetime.now()
MODEL_PATH = f"results/unet_e{EPOCHS}_T{TIMESTEPS}_d{date.month}{date.day}_h{date.hour}{date.minute}.pt"
torch.save(model.state_dict(), MODEL_PATH)
print(f"Unet saved at {MODEL_PATH}")
