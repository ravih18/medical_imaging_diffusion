import toml
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
    choices=['IXI', 'ADNI_T1_PET', 'ADNI_AD_CN', 'BRATS'],
    help='Training task for DSB',
)
args = parser.parse_args()

expe_dir = Path(f"experiments/{task}/dsb{args.model_number}")
print(f"Working dir : {expe_dir}")
dsb_param = dsb_config_from_toml(experiment_directory / "config.toml")

if args.task == 'IXI':
    from image_dataset.CapsSlicesIXI import get_IXI_datasets
    datasets = get_IXI_datasets()
elif args.task == 'ADNI_T1_PET':
    from image_dataset.CapsSlicesADNI import get_ADNI_datasets
    datasets = get_ADNI_datasets(args.task)
elif args.task == 'ADNI_AD_CN':
    from image_dataset.CapsSlicesADNI import get_ADNI_datasets
    datasets = get_ADNI_datasets(args.task)
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