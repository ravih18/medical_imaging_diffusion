from pathlib import Path
import argparse
from utils.config import *
from diffusion.dsb import IPF  

parser = argparse.ArgumentParser(description='Resume DSB model')
parser.add_argument(
    'model_number',
    help='Folder containing the toml config',
)
parser.add_argument(
    'task',
    choices=['IXI', 'ADNI_T1_PET', 'ADNI_AD_CN', 'ADNI_hypo', 'BRATS'],
    help='Training task for DSB',
)
parser.add_argument(
    'fb',
    choices=['f', 'b'],
    help='forward or backward',
)
parser.add_argument(
    'checkpoint_ipf',
    help='IPF iteration',
)
args = parser.parse_args()

expe_dir = Path(f"experiments/{args.task}/dsb{args.model_number}")
print(f"Working dir : {expe_dir}")
dsb_param = dsb_config_from_toml(expe_dir / "config.toml")

img_size = 64

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

dsb.checkpoint_it = int(args.checkpoint_ipf)
dsb.checkpoint_pass = args.fb

dsb.load_checkpoints(
    int(args.checkpoint_ipf) - 1,
    dsb.num_iter,
)

#print(dsb.checkpoint_it)
#print(dsb.checkpoint_pass)
dsb.train()
