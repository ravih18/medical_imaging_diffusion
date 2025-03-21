from pathlib import Path
import nibabel as nib
from nilearn.image import resample_to_img
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode


MNI_MASK = Path("/lustre/fswork/projects/rech/krk/usy14zi/diffusion_models/medical_imaging_diffusion/ressources/dilated_MNI_binary_mask.nii.gz")
CAPS_ADNI = Path("/lustre/fswork/projects/rech/krk/commun/datasets/adni/caps/caps_pet_uniform_v2025")

ref_img_path = (
    CAPS_ADNI 
    / "subjects"
    / "sub-ADNI002S0729"
    / "ses-M048"
    / "pet_linear"
    / "sub-ADNI002S0729_ses-M048_trc-18FFDG_rec-coregiso_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_suvr-cerebellumPons2_pet.nii.gz"
)


def get_mni_mask(img_size=64):
    mask_nii = nib.load(MNI_MASK)
    ref_image = nib.load(ref_img_path)
    resampled_mask_nii = resample_to_img(
        mask_nii, ref_image, interpolation='nearest', copy_header=True, force_resample=True,
    )   
    mask = resampled_mask_nii.get_fdata(dtype=np.float32)

    slice_transform = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=InterpolationMode.NEAREST),
    ])

    return torch.Tensor(mask).unsqueeze(0), slice_transform


def get_pathology_mask(pathology='AD', img_size=64):
    from scipy.ndimage import gaussian_filter
    mask_path = CAPS_ADNI / "masks" / f"mask_hypo_{pathology.lower()}_resampled.nii"
    mask_nii = nib.load(mask_path)
    mask = mask_nii.get_fdata(dtype=np.float32)
    mask = gaussian_filter(mask, sigma=2)

    slice_transform = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=InterpolationMode.NEAREST),
    ])

    return torch.Tensor(mask).unsqueeze(0), slice_transform