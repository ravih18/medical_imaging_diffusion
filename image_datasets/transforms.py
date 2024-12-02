import torch
import numpy as np

class ClipTensor(object):
    def __call__(self, tensor):
        unique, counts = torch.unique(tensor, return_counts=True)
        clip_threshold = max(0.0, unique[np.argmax(counts)])
        return torch.clamp(tensor, min=clip_threshold)