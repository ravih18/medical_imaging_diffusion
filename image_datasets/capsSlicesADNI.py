import json
import numpy as np
import pandas as pd
import torch
import copy
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
from typing import Callable, Optional, Tuple


CAPS_ADNI = Path("/lustre/fswork/projects/rech/krk/commun/datasets/adni/caps/caps_pet_uniform_v2025")


class CapsSliceADNI(Dataset):
    def __init__(
        self,
        caps_directory: Path,
        preprocessing_json: Path,
        subject_tsv: Path,
        image_transformations: Optional[Callable]=None,
        slice_transformations: Optional[Callable]=None,
        return_hypo: bool = False,
        label_transformations: Optional[Callable]=None,
    ):
        self.caps_directory = Path(caps_directory)
        self.df = pd.read_csv(subject_tsv, sep='\t', )

        self.modality, self.file_pattern = self.read_preprocessing_json(
            preprocessing_json
        )

        self.slice_min = 64-10
        self.slice_max = 64+10
        self.elem_per_image = self.slice_max - self.slice_min

        self.image_transformations = image_transformations
        self.slice_transformations = slice_transformations

        self.return_hypo = return_hypo
        if self.return_hypo:
            self.label_transformations = label_transformations

        self.size = self[0]['image'].size()

    def __len__(self) -> int:
        return len(self.df) * self.elem_per_image

    def __getitem__(self, idx):
        participant, session, slice_idx = self._get_meta_data(idx)

        data = {
            "participant_id": participant,
            "session_id": session,
            "slice_id": slice_idx,
        }

        image_path = (
            self.caps_directory
            / "subjects"
            / participant
            / session
            / "deeplearning_prepare_data"
            / "image_based"
            / f"{self.modality.lower()}_linear"
            / f"{participant}_{session}{self.file_pattern}"
        )
        image = torch.load(image_path)
        if self.return_hypo:
                label = copy.deepcopy(image)

        if self.image_transformations:
            image = self.image_transformations(image)
        slice_tensor = image[:,:,:,slice_idx]

        if self.slice_transformations:
            slice_tensor = self.slice_transformations(slice_tensor)
        data["image"] = slice_tensor

        if self.return_hypo:
            if self.label_transformations:
                label = self.label_transformations(label)
            slice_label = label[:,:,:,slice_idx]
            if self.slice_transformations:
                slice_label = self.slice_transformations(slice_label)
            data["label"] = slice_label

        return data

    def _get_meta_data(self, idx: int) -> Tuple[str, str, str, int, int]:
        """
        Gets all meta data necessary to compute the path with _get_image_path

        Args:
            idx (int): row number of the meta-data contained in self.df
        Returns:
            participant (str): ID of the participant.
            slice_index (int): Index of the part of the image.
        """
        image_idx = idx // self.elem_per_image
        participant = self.df.loc[image_idx, "participant_id"]
        session = self.df.loc[image_idx, "session_id"]
        #session = ''.join(list((session).pop(session.index("0"))))
        slice_idx = (idx % self.elem_per_image) + self.slice_min
        return participant, session, slice_idx

    def read_preprocessing_json(self, preprocessing_json):
        with open(preprocessing_json, 'r', encoding='utf-8') as file:
            # Charger le contenu du fichier JSON en dictionnaire
            preprocessing_dict = json.load(file)
        if preprocessing_dict["preprocessing"] == "pet-linear":
            modality = "PET"
        elif preprocessing_dict["preprocessing"] == "t1w-linear":
            modality = "T1"
        
        pattern = preprocessing_dict["file_type"]["pattern"].split("*")[1].split('.')[0] + ".pt"
        return modality, pattern

def get_ADNI_datasets(task, img_size=64):
    
    #indexes = list(range(64-10,64+10))

    pet_preprocessing_json = CAPS_ADNI / "preprocessing_json" / "extract_pet_uniform_slice.json" # v2025
    #pet_preprocessing_json = CAPS_ADNI / "tensor_extraction" / "extract_pet_uniform_image.json" # caps pet uniform
    
    image_transformations = transforms.Compose([
        transforms.Lambda(lambda t: 2*(t-t.min())/(t.max()-t.min()) - 1) # normalize images between -1 and 1
    ])
    slice_transformations = transforms.Compose([
        # transforms.Pad([0, 19], fill=-1),
        transforms.Resize((img_size, img_size)),
    ])

    train_cn_tsv = CAPS_ADNI / "splits_dsb" / "train_cn.tsv"
    val_cn_tsv = CAPS_ADNI / "splits_dsb" / "validation_cn_baseline.tsv"
    
    # if task == "ADNI_T1_PET":
    #     t1_preprocessing_json = CAPS_ADNI / "preprocessing_json" / "extract_t1w_slice.json"
    #     # transforms_t1w = transforms.Compose([
    #     #     transforms.Resize((64, 64)),
    #     #     transforms.Lambda(lambda t: 2*(t-t.min())/(t.max()-t.min()) - 1) # normalize images between -1 and 1
    #     # ])

    #     dataset_train_init = CapsSliceADNI(
    #         CAPS_ADNI,
    #         t1_preprocessing_json,
    #         train_cn_tsv,
    #         image_transformations=image_transformations,
    #         slice_transformations=slice_transformations,
    #     )
    #     dataset_train_final = CapsSliceADNI(
    #         CAPS_ADNI,
    #         pet_preprocessing_json,
    #         train_cn_tsv,
    #         image_transformations=image_transformations,
    #         slice_transformations=slice_transformations,
    #     )
    #     dataset_val_init = CapsSliceADNI(
    #         CAPS_ADNI,
    #         t1_preprocessing_json,
    #         val_cn_tsv,
    #         image_transformations=image_transformations,
    #         slice_transformations=slice_transformations,
    #     )
    #     dataset_val_final = CapsSliceADNI(
    #         CAPS_ADNI,
    #         pet_preprocessing_json,
    #         val_cn_tsv,
    #         image_transformations=image_transformations,
    #         slice_transformations=slice_transformations,
    #     )

    # elif task == "ADNI_AD_CN":
    train_ad_tsv = CAPS_ADNI / "splits_dsb" / "train_ad.tsv"
    val_ad_tsv = CAPS_ADNI / "splits_dsb" / "validation_ad_baseline.tsv"

    dataset_train_init = CapsSliceADNI(
        CAPS_ADNI,
        pet_preprocessing_json,
        #train_ad_tsv,
        train_cn_tsv,
        image_transformations=image_transformations,
        slice_transformations=slice_transformations,
    )
    dataset_train_final = CapsSliceADNI(
        CAPS_ADNI,
        pet_preprocessing_json,
        #train_cn_tsv,
        train_ad_tsv,
        image_transformations=image_transformations,
        slice_transformations=slice_transformations,
    )
    dataset_val_init = CapsSliceADNI(
        CAPS_ADNI,
        pet_preprocessing_json,
        #val_ad_tsv,
        val_cn_tsv,
        image_transformations=image_transformations,
        slice_transformations=slice_transformations,
    )
    dataset_val_final = CapsSliceADNI(
        CAPS_ADNI,
        pet_preprocessing_json,
        #val_cn_tsv,
        val_ad_tsv,
        image_transformations=image_transformations,
        slice_transformations=slice_transformations,
    )

    # mean_final = torch.zeros([1, img_size, img_size])
    # var_final = 1 * torch.ones([1, img_size, img_size])

    datasets = {
        "train_init": dataset_train_init,
        "train_final": dataset_train_final,
        "val_init": dataset_val_init,
        "val_final": dataset_val_final,
        #"mean_final": mean_final,
        #"var_final": var_final,
    }

    return datasets


def get_ADNI_validation(img_size=64):
    caps_hypo = Path("/lustre/fswork/projects/rech/krk/commun/datasets/adni/caps/hypometabolic_caps/caps_ad_30")
    val_cn_tsv = CAPS_ADNI / "splits_dsb" / "validation_cn_baseline.tsv"
    #val_ad_tsv = CAPS_ADNI / "splits_dsb" / "validation_ad_baseline.tsv"
    val_hypo_tsv = Path("/lustre/fswork/projects/rech/krk/commun/datasets/adni/splits_dsb/validation_hypo.tsv")

    pet_preprocessing_json = CAPS_ADNI / "preprocessing_json" / "extract_pet_uniform_slice.json" # v2025
    pet_hypo_json = caps_hypo / "tensor_extraction" / "extract_pet_uniform_image.json"

    image_transformations = transforms.Compose([
        transforms.Lambda(lambda t: 2*(t-t.min())/(t.max()-t.min()) - 1) # normalize images between -1 and 1
    ])
    slice_transformations = transforms.Compose([
       transforms.Resize((img_size, img_size)),
    ])

    dataset_val_init = CapsSliceADNI(
        caps_hypo,
        pet_hypo_json,
        val_hypo_tsv,
        image_transformations=image_transformations,
        slice_transformations=slice_transformations,
    )
    dataset_val_final = CapsSliceADNI(
        CAPS_ADNI,
        pet_preprocessing_json,
        val_cn_tsv,
        image_transformations=image_transformations,
        slice_transformations=slice_transformations,
    )

    return dataset_val_init, dataset_val_final


def get_ADNI_AD_validation(img_size=64):
    val_ad_tsv = CAPS_ADNI / "splits_dsb" / "validation_ad_baseline.tsv"
    pet_preprocessing_json = CAPS_ADNI / "preprocessing_json" / "extract_pet_uniform_slice.json"

    image_transformations = transforms.Compose([
        transforms.Lambda(lambda t: 2*(t-t.min())/(t.max()-t.min()) - 1) # normalize images between -1 and 1
    ])
    slice_transformations = transforms.Compose([
       transforms.Resize((img_size, img_size)),
    ])

    dataset_ad_val = CapsSliceADNI(
        CAPS_ADNI,
        pet_hypo_json,
        val_ad_tsv,
        image_transformations=image_transformations,
        slice_transformations=slice_transformations,
    )
    return dataset_ad_val

def get_ADNI_hypo_datasets(img_size=64, pathology="AD", percentage=30):

    pet_preprocessing_json = CAPS_ADNI / "preprocessing_json" / "extract_pet_uniform_slice.json" # v2025
    

    final_transformations = transforms.Compose([
        SimulateHypometabolic(pathology=pathology, percentage=percentage),
        transforms.Lambda(lambda t: 2*(t-t.min())/(t.max()-t.min()) - 1) # normalize images between -1 and 1
    ])
    image_transformations = transforms.Compose([
        transforms.Lambda(lambda t: 2*(t-t.min())/(t.max()-t.min()) - 1) # normalize images between -1 and 1
    ])
    slice_transformations = transforms.Compose([
       transforms.Resize((img_size, img_size)),
    ])

    train_cn_tsv = CAPS_ADNI / "splits_dsb" / "train_cn.tsv"
    val_cn_tsv = CAPS_ADNI / "splits_dsb" / "validation_cn_baseline.tsv"

    dataset_train_init = CapsSliceADNI(
        CAPS_ADNI,
        pet_preprocessing_json,
        train_cn_tsv,
        image_transformations=image_transformations,
        slice_transformations=slice_transformations,
    )
    dataset_train_final = CapsSliceADNI(
        CAPS_ADNI,
        pet_preprocessing_json,
        train_cn_tsv,
        image_transformations=final_transformations,
        slice_transformations=slice_transformations,
        return_hypo=True,
        label_transformations=image_transformations,
    )
    dataset_val_init = CapsSliceADNI(
        CAPS_ADNI,
        pet_preprocessing_json,
        val_cn_tsv,
        image_transformations=image_transformations,
        slice_transformations=slice_transformations,
    )
    dataset_val_final = CapsSliceADNI(
        CAPS_ADNI,
        pet_preprocessing_json,
        val_cn_tsv,
        image_transformations=final_transformations,
        slice_transformations=slice_transformations,
        return_hypo=True,
        label_transformations=image_transformations,
    )

    datasets = {
        "train_init": dataset_train_init,
        "train_final": dataset_train_final,
        "val_init": dataset_val_init,
        "val_final": dataset_val_final,
    }

    return datasets


def get_dataset_val_hypo(img_size=64, pathology="AD", percentage=30, test=False):

    if test == True:
        val_tsv = CAPS_ADNI / "splits_dsb" / "test_cn_baseline.tsv"
    else:
        val_tsv = CAPS_ADNI / "splits_dsb" / "validation_cn_baseline.tsv"

    pet_preprocessing_json = CAPS_ADNI / "preprocessing_json" / "extract_pet_uniform_slice.json"

    image_transformations = transforms.Compose([
        SimulateHypometabolic(pathology=pathology, percentage=percentage),
        transforms.Lambda(lambda t: 2*(t-t.min())/(t.max()-t.min()) - 1) # normalize images between -1 and 1
    ])
    label_transformations = transforms.Compose([
        transforms.Lambda(lambda t: 2*(t-t.min())/(t.max()-t.min()) - 1) # normalize images between -1 and 1
    ])
    slice_transformations = transforms.Compose([
       transforms.Resize((img_size, img_size)),
    ])

    dataset_val_hypo = CapsSliceADNI(
        CAPS_ADNI,
        pet_preprocessing_json,
        val_tsv,
        image_transformations=image_transformations,
        slice_transformations=slice_transformations,
        return_hypo=True,
        label_transformations=label_transformations,
    )
    return dataset_val_hypo


class SimulateHypometabolic(torch.nn.Module):
    def __init__(self, pathology: str, percentage: int, sigma: int = 2):
        import nibabel as nib

        super(SimulateHypometabolic, self).__init__()

        self.pathology = pathology
        self.percentage = percentage
        self.sigma = sigma

        mask_path = CAPS_ADNI / "masks" / f"mask_hypo_{self.pathology.lower()}_resampled.nii"
        mask_nii = nib.load(mask_path)
        self.mask = self.mask_processing(
            mask_nii.get_fdata()
        )

    def forward(self, img):
        new_img = img * self.mask
        return new_img

    def mask_processing(self, mask):
        from scipy.ndimage import gaussian_filter
        inverse_mask = 1 - mask
        inverse_mask[inverse_mask == 0] = 1 - self.percentage / 100
        gaussian_mask = gaussian_filter(inverse_mask, sigma=self.sigma)
        return np.float32(gaussian_mask)
