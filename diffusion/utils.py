import torch
import numpy as np


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def batch_psnr(ref_ims, images, max_I):
    '''
    batch of ref_im and im are tensors of dims N, C, W, H
    '''
    mse = ((ref_ims - images)**2).mean(axis=(1,2,3))
    psnr_all =  10*(np.log10(max_I**2) - torch.log10(mse.clone()))
    return psnr_all.mean()
