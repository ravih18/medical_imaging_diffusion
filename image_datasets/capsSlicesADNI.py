import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
from typing import Callable, Optional, Tuple


CAPS_ADNI = Path("/lustre/fswork/projects/rech/krk/commun/datasets/adni/caps/caps_pet_uniform_v2025")
#CAPS_ADNI = Path("/lustre/fswork/projects/rech/krk/commun/datasets/adni/caps/caps_pet_uniform")

# class CapsDatasetSlice_hr(CapsDataset):
#     """Dataset of MRI organized in a CAPS folder."""

#     def __init__(
#         self,
#         caps_directory: Path,
#         tsv_label: Path,
#         preprocessing_dict: Dict[str, Any],
#         index_slices : List,
#         train_transformations: Optional[Callable] = None,
#         label_presence: bool = True,
#         label: str = None,
#         label_code: Dict[str, int] = None,
#         all_transformations: Optional[Callable] = None,
#         transforms_slice = None
#     ):
#         """
#         Args:
#             caps_directory: Directory of all the images.
#             data_file: Path to the tsv file or DataFrame containing the subject/session list.
#             preprocessing_dict: preprocessing dict contained in the JSON file of prepare_data.
#             train_transformations: Optional transform to be applied only on training mode.
#             label_presence: If True the diagnosis will be extracted from the given DataFrame.
#             label: Name of the column in data_df containing the label.
#             label_code: label code that links the output node number to label value.
#             all_transformations: Optional transform to be applied during training and evaluation.
#             multi_cohort: If True caps_directory is the path to a TSV file linking cohort names and paths.

#         """
#         self.n_slices = len(index_slices)
#         self.index_slices = index_slices
#         self.mode = "slice"
#         self.transforms_slice = transforms_slice
#         super().__init__(
#             caps_directory,
#             tsv_label,
#             preprocessing_dict,
#             augmentation_transformations=train_transformations,
#             label_presence=label_presence,
#             label=label,
#             label_code=label_code,
#             transformations=all_transformations,
#         )
        
#         self.prepare_dl = self.preprocessing_dict["prepare_dl"]

#     @property
#     def elem_index(self):
#         return None

#     def __getitem__(self, idx):
#         participant, session, elem_idx, label, domain = self._get_meta_data(idx)

#         image_path = self._get_image_path(participant, session)
#         image = torch.load(image_path)

#         if self.transformations:
#             image = self.transformations(image)

#         if self.augmentation_transformations and not self.eval_mode:
#             image = self.augmentation_transformations(image)
        
#         slice_index = self.index_slices[elem_idx]
#         image = image[:,:,:,slice_index]

#         if self.transforms_slice is not None:
#             image = self.transforms_slice(image)

#         sample = {
#             "image": image ,
#             "label": label,
#             "participant_id": participant,
#             "session_id": session,
#             "image_id": 0,
#             "image_path": image_path.as_posix(),
#             "domain": domain,
#         }

#         return sample

#     def num_elem_per_image(self):
#         return self.n_slices

class CapsSliceADNI(Dataset):
    def __init__(
        self,
        caps_directory: Path,
        preprocessing_json: Path,
        subject_tsv: Path,
        image_transformations: Optional[Callable]=None,
        slice_transformations: Optional[Callable]=None,
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
        self.size = self[0]['image'].size()

    def __len__(self) -> int:
        return len(self.df) * self.elem_per_image

    def __getitem__(self, idx):
        participant, session, slice_idx = self._get_meta_data(idx)
        #print(participant, session)
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
        try:
            image = torch.load(image_path)
        except:
            raise ValueError(f"File {image_path} does not exist.")
        
        if self.image_transformations:
            image = self.image_transformations(image)
        slice_tensor = image[:,:,:,slice_idx]
        if self.slice_transformations:
            slice_tensor = self.slice_transformations(slice_tensor)

        data["image"] = slice_tensor
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
        elif preprocessing_dict["preprocessing"] == "t1-linear":
            modality = "T1w"
        
        pattern = preprocessing_dict["file_type"]["pattern"].split("*")[1].split('.')[0] + ".pt"
        return modality, pattern

def get_ADNI_datasets(task):
    
    #indexes = list(range(64-10,64+10))

    pet_preprocessing_json = CAPS_ADNI / "preprocessing_json" / "extract_pet_uniform_slice.json" # v2025
    #pet_preprocessing_json = CAPS_ADNI / "tensor_extraction" / "extract_pet_uniform_image.json" # caps pet uniform
    
    image_transformations = transforms.Compose([
        transforms.Lambda(lambda t: 2*(t-t.min())/(t.max()-t.min()) - 1) # normalize images between -1 and 1
    ])
    slice_transformations = transforms.Compose([
       transforms.Resize((64, 64)),
    ])

    train_cn_tsv = CAPS_ADNI / "splits_dsb" / "train_cn.tsv"
    val_cn_tsv = CAPS_ADNI / "splits_dsb" / "validation_cn_baseline.tsv"
    
    if task == "ADNI_T1_PET":
        t1_preprocessing_json = CAPS_ADNI / "preprocessing_json" / "extract_t1w_slice.json"
        # transforms_t1w = transforms.Compose([
        #     transforms.Resize((64, 64)),
        #     transforms.Lambda(lambda t: 2*(t-t.min())/(t.max()-t.min()) - 1) # normalize images between -1 and 1
        # ])

        dataset_train_init = CapsSliceADNI(
            CAPS_ADNI,
            t1_preprocessing_json,
            train_cn_tsv,
            image_transformations=image_transformations,
            slice_transformations=slice_transformations,
        )
        dataset_train_final = CapsSliceADNI(
            CAPS_ADNI,
            pet_preprocessing_json,
            train_cn_tsv,
            image_transformations=image_transformations,
            slice_transformations=slice_transformations,
        )
        dataset_val_init = CapsSliceADNI(
            CAPS_ADNI,
            t1_preprocessing_json,
            val_cn_tsv,
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

    elif task == "ADNI_AD_CN":
        train_ad_tsv = CAPS_ADNI / "splits_dsb" / "train_ad.tsv"
        val_ad_tsv = CAPS_ADNI / "splits_dsb" / "validation_ad_baseline.tsv"

        dataset_train_init = CapsSliceADNI(
            CAPS_ADNI,
            pet_preprocessing_json,
            train_ad_tsv,
            image_transformations=image_transformations,
            slice_transformations=slice_transformations,
        )
        dataset_train_final = CapsSliceADNI(
            CAPS_ADNI,
            pet_preprocessing_json,
            train_cn_tsv,
            image_transformations=image_transformations,
            slice_transformations=slice_transformations,
        )
        dataset_val_init = CapsSliceADNI(
            CAPS_ADNI,
            pet_preprocessing_json,
            val_ad_tsv,
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

    datasets = {
        "train_init": dataset_train_init,
        "train_final": dataset_train_final,
        "val_init": dataset_val_init,
        "val_final": dataset_val_final,
    }

    return datasets