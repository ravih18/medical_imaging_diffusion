from pathlib import Path
import numpy as np
import pandas as pd

import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from diffusion.dsb import DiffusionSchrodingerBridge
from image_datasets.capsSlicesADNI import get_ADNI_datasets
from image_datasets.capsSlicesADNI import get_dataset_val_hypo
from evaluation import compute_metrics
from evaluation.visualisation import *
from evaluation.mask import get_mni_mask
from utils.config import *

#task = "ADNI_AD_CN"
task = "ADNI_hypo"
nums = range(48, 51)

n_ipf = 6
num_iter = 10000
fb = 'b'

for num in nums:
    print(num)
    
    if 15<=num<25 or 31<=num<35 or 40<=num<50:
        img_size = 32
    elif 6<=num<15 or 25<=num<31 or 53<=num<56 or num==51:
        img_size = 128
    else:
        img_size = 64

    print('image size:', img_size)
    dataset_val = get_dataset_val_hypo(img_size=img_size)
    val_dl = DataLoader(
        dataset_val,
        batch_size=64,
    )

    expe_dir = Path(f"experiments/{task}/dsb{num}")

    dsb_param = dsb_config_from_toml(expe_dir / "config.toml")
    datasets = get_ADNI_datasets(task, img_size=img_size)

    dsb = DiffusionSchrodingerBridge(
        experiment_directory = expe_dir,
        dsb_params = dsb_param,
        datasets = datasets,
        transfer = True,
        evaluation = True,
    )

    tsv_1, tsv_2 = compute_metrics(dsb, val_dl, n_ipf, num_iter, img_size, fb)

    df_m = pd.read_csv(tsv_1, sep="\t")

    df_ssim = df_m[df_m['metric']=='SSIM']
    print('SSIM to input', df_ssim[df_ssim['image_X']=='input'].value.mean())
    print('SSIM to gt', df_ssim[df_ssim['image_X']=='ground_truth'].value.mean())

    ## MAKE PLOTS
    dsb.load_checkpoints(n_ipf, num_iter, fb=fb)
    batch = next(iter(val_dl))
    image = batch['image'][10].unsqueeze(dim=0)
    slice_id = batch['slice_id'][10].item()

    mni_mask, mask_transform = get_mni_mask(img_size)
    mask_mni_slice = mask_transform(mni_mask[:,:,:,slice_id]).squeeze()

    renorm = transforms.Compose([
        transforms.Lambda(lambda t: 0.5*(t+1))
    ])

    samples = dsb.sample_batch(image, fb)

    plot_dir = expe_dir / "evaluation"
    plot_dir.mkdir(parents=True, exist_ok=True)
    X = renorm(image.squeeze().cpu()) * mask_mni_slice
    Y = renorm(samples[0][-1].squeeze().cpu()) * mask_mni_slice
    Gt = renorm(batch['label'][10].squeeze().cpu()) * mask_mni_slice
    make_diff_plot(X, Y, Gt, plot_file = plot_dir / f"diff_plot_{fb}.pdf")

    plot_dir = expe_dir / "evaluation"
    plot_dir.mkdir(parents=True, exist_ok=True)
    make_traj_plot(
        renorm(image.squeeze().cpu()),
        renorm(samples[0]),
        mask = mask_mni_slice,
        plot_file = plot_dir / f"trajectory_{fb}.pdf",
    )
