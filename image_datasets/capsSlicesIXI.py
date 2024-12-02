import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Callable, Optional, Tuple


class CapsSlicesIXI(Dataset):
    """"""
    def __init__(
        self,
        caps_directory: Path,
        subject_tsv: Path,
        transformations: Optional[Callable]=None,
    ):
        self.caps_directory = Path(caps_directory)
        self.df = pd.read_csv(subject_tsv, sep='\t', )

        self.slice_min = 80
        self.slice_max = 110
        self.elem_per_image = self.slice_max - self.slice_min

        self.transformations = transformations

        self.size = self[0]["T1"].size()

    def __len__(self) -> int:
        return len(self.df) * self.elem_per_image

    def __getitem__(self, idx):
        participant, slice_idx = self._get_meta_data(idx)
        
        slice_dir = (
            self.caps_directory
            / "subjects"
            / participant
            / "ses-M000"
            / "deeplearning_prepare_data"
            / "slice_based"
        )
        t1w_path = (
            slice_dir
            / "t1_linear"
            / f"{participant}_ses-m000_T1w_space-MNI152NLin2009cSym_res-1x1x1_T1w_slice-axial_{slice_idx}.pt"
        )
        t2w_path = (
            slice_dir
            / "t2_linear"
            / f"{participant}_ses-m000_T2w_space-MNI152NLin2009cSym_res-1x1x1_T2w_slice-axial_{slice_idx}.pt"
        )

        try:
            slice_t1w = torch.load(t1w_path).unsqueeze(dim=0)
        except:
            raise ValueError(f"File {t1w_path} does not exist.")
        try:
            slice_t2w = torch.load(t2w_path).unsqueeze(dim=0)
        except:
            raise ValueError(f"File {t2w_path} does not exist.")

        if self.transformations:
            slice_t1w = self.transformations(slice_t1w)
            slice_t2w = self.transformations(slice_t2w)

        data = {
            "T1": slice_t1w,
            "T2": slice_t2w,
            "participant_id": participant,
            "slice_id": slice_idx,
        }
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
