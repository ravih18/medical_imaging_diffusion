from pathlib import Path
import argparse
from utils.config import *
from diffusion.dsb import IPF     

parser = argparse.ArgumentParser(description='Training DSB model')
parser.add_argument(
    'model_number',
    help='Folder containing the toml config',
)
parser.add_argument(
    'task',
    choices=['IXI', 'ADNI_T1_PET', 'ADNI_AD_CN', 'ADNI_hypo', 'BRATS'],
    help='Training task for DSB',
)
args = parser.parse_args()

expe_dir = Path(f"experiments/{args.task}/dsb{args.model_number}")
print(f"Working dir : {expe_dir}")
dsb_param = dsb_config_from_toml(expe_dir / "config.toml")

img_size = 32

print(f"Working with images of size {img_size}")

if args.task == 'IXI':
    from image_datasets.capsSlicesIXI import get_IXI_datasets
    datasets = get_IXI_datasets()
elif args.task == 'ADNI_T1_PET':
    from image_datasets.capsSlicesADNI import get_ADNI_datasets
    datasets = get_ADNI_datasets(args.task, img_size = img_size)
elif args.task == 'ADNI_AD_CN':
    from image_datasets.capsSlicesADNI import get_ADNI_datasets
    datasets = get_ADNI_datasets(args.task, img_size = img_size)
elif args.task == 'ADNI_hypo':
    from image_datasets.capsSlicesADNI import get_ADNI_hypo_datasets
    datasets = get_ADNI_hypo_datasets(img_size = img_size, pathology='AD', percentage=30)
elif args.task == 'BRATS':
    pass

dsb = IPF(
    #caps_directory = caps_dir,
    experiment_directory = expe_dir,
    dsb_params = dsb_param,
    datasets = datasets,
    transfer = True
)
print(dsb.device)

dsb.train()
