import toml
from pathlib import Path
import argparse
from utils.config import *
from diffusion.dsbm import DSBM_IMF     

parser = argparse.ArgumentParser(description='Training DSBM-IMF model')
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

expe_dir = Path(f"experiments_dsbm/{args.task}/dsbm{args.model_number}")
print(f"Working dir : {expe_dir}")
dsbm_params = dsbm_config_from_toml(expe_dir / "config.toml")

if args.task == 'IXI':
    from image_datasets.capsSlicesIXI import get_IXI_datasets
    datasets = get_IXI_datasets()
elif args.task == 'ADNI_T1_PET':
    from image_datasets.capsSlicesADNI import get_ADNI_datasets
    datasets = get_ADNI_datasets(args.task)
elif args.task == 'ADNI_AD_CN':
    from image_datasets.capsSlicesADNI import get_ADNI_datasets
    datasets = get_ADNI_datasets(args.task)
elif args.task == 'BRATS':
    pass

dsbm = DSBM_IMF(
    #caps_directory = caps_dir,
    experiment_directory = expe_dir,
    dsbm_paramss = dsbm_params,
    datasets = datasets,
    transfer = True
)
print(dsbm.device)

dsbm.train()
