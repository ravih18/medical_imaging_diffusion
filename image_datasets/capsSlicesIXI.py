import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Callable, Optional, Tuple


CAPS_IXI = Path("/lustre/fswork/projects/rech/krk/commun/datasets/IXI/caps_IXI")


class CapsSlicesIXI(Dataset):
    """"""
    def __init__(
        self,
        caps_directory: Path,
        subject_tsv: Path,
        sequence: Optional[str]=None,
        transformations: Optional[Callable]=None,
    ):
        self.caps_directory = Path(caps_directory)
        self.df = pd.read_csv(subject_tsv, sep='\t', )

        self.slice_min = 80
        self.slice_max = 110
        self.elem_per_image = self.slice_max - self.slice_min

        self.transformations = transformations

        if sequence is not None:
            if (sequence != 'T1') and (sequence != 'T2'):
                raise ValueError("Invalid value for sequence argument. Sequence must be 'T1' or 'T2'.")
            self.sequence = [sequence]
        else:
            self.sequence = ['T1', 'T2']
        self.size = self[0][self.sequence[0]].size()

    def __len__(self) -> int:
        return len(self.df) * self.elem_per_image

    def __getitem__(self, idx):
        participant, slice_idx = self._get_meta_data(idx)
        data = {
            "participant_id": participant,
            "slice_id": slice_idx,
        }

        slice_dir = (
            self.caps_directory
            / "subjects"
            / participant
            / "ses-M000"
            / "deeplearning_prepare_data"
            / "slice_based"
        )
        for sequence in self.sequence:
            seq_path = (
                slice_dir
                / f"{sequence.lower()}_linear"
                / f"{participant}_ses-m000_{sequence}w_space-MNI152NLin2009cSym_res-1x1x1_{sequence}w_slice-axial_{slice_idx}.pt"
            )
            try:
                slice_tensor = torch.load(seq_path).unsqueeze(dim=0)
            except:
                raise ValueError(f"File {seq_path} does not exist.")
            if self.transformations:
                slice_tensor = self.transformations(slice_tensor)
            data[sequence] = slice_tensor

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
        slice_idx = (idx % self.elem_per_image) + self.slice_min
        return participant, slice_idx


def get_IXI_datasets(caps_dir):
    from torchvision import transforms

    train_tsv = CAPS_IXI / "IXI_train.tsv"
    val_tsv = CAPS_IXI / "IXI_validation.tsv"

    transform=transforms.Compose([
        transforms.Pad([0, 18], fill=-1),
        transforms.Resize((64, 64)),
        #transforms.Lambda(lambda t: (t+1)/2),    # Image range between [0, 1]
    ])

    dataset_train_T1 = CapsSlicesIXI(
        CAPS_IXI,
        train_tsv,
        sequence='T1',
        transformations=transform,
    )
    dataset_train_T2 = CapsSlicesIXI(
        CAPS_IXI,
        train_tsv,
        sequence='T2',
        transformations=transform,
    )
    dataset_val_T1 = CapsSlicesIXI(
        CAPS_IXI,
        val_tsv,
        sequence='T1',
        transformations=transform
    )
    dataset_val_T2 = CapsSlicesIXI(
        CAPS_IXI,
        val_tsv,
        sequence='T2',
        transformations=transform
    )

    datasets = {
        "train_init": dataset_train_T1,
        "train_final": dataset_train_T2,
        "val_init": dataset_val_T1,
        "val_final": dataset_val_T2,
    }

    return datasets