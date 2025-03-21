import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
#import matplotlib.animation as animation
#%matplotlib inline

class MidpointNormalize(mpl.colors.Normalize):
    def __init__(self, vmin, vmax, midpoint=0, clip=False):
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        normalized_min = max(0, 1 / 2 * (1 - abs((self.midpoint - self.vmin) / (self.midpoint - self.vmax))))
        normalized_max = min(1, 1 / 2 * (1 + abs((self.vmax - self.midpoint) / (self.midpoint - self.vmin))))
        normalized_mid = 0.5
        x, y = [self.vmin, self.midpoint, self.vmax], [normalized_min, normalized_mid, normalized_max]
        return np.ma.masked_array(np.interp(value, x, y))


def make_diff_plot(x, y, gt, m=None, plot_file=None):

    imgs = [x, y, x-y, gt]

    labels = [
        "Input: AD 30%",
        "Output",
        "Input - Output",
        "Ground Truth",
    ]

    if m is not None:
        imgs.append(m)
        labels.append("Mask")

    fig, axes = plt.subplots(1, len(imgs), figsize=(14, 4))    

    for i in range(len(axes)):
        axes[i].get_xaxis().set_ticks([])
        axes[i].get_yaxis().set_ticks([])

        if i==2 or i==4:
            cmap, vmin, vmax = 'seismic', -1, 1
        else:
            cmap, vmin, vmax = 'nipy_spectral', 0, 1

        axes[i].imshow(np.rot90(imgs[i]), cmap=cmap, vmin=vmin, vmax=vmax)
        axes[i].set_xlabel(labels[i], fontsize=12)
    
    cax0,kw0 = mpl.colorbar.make_axes(
        [ax for ax in axes.flat], location='left',
        shrink=0.50,
        #aspect=80,
        pad=0.01,
    )
    plt.colorbar(
        mpl.cm.ScalarMappable(
            norm=mpl.colors.Normalize(vmin=-1, vmax=1),
            cmap='nipy_spectral'
        ),
        cax=cax0,
        **kw0
    )
    cax1,kw1 = mpl.colorbar.make_axes(
        [ax for ax in axes.flat], location='right',
        shrink=0.50,
        #aspect=80,
        pad=0.01,
    )
    plt.colorbar(
        mpl.cm.ScalarMappable(
            norm=MidpointNormalize(vmin=-1, vmax=1, midpoint=0),
            cmap='seismic'
        ),
        cax=cax1,
        **kw1)

    if plot_file:
        plt.savefig(plot_file, format="pdf", bbox_inches="tight")
    else:
        plt.show()


def make_traj_plot(input_img, samples, mask=None, plot_file=None, n = 6):

    timesteps = samples.shape[0]
    images = [input_img]
    labels = [f"Input: 0/{timesteps}"]

    for i in range(1, timesteps):
        if i%(timesteps//n+2)==0:
            if mask is not None:
                images.append(samples[i].squeeze().cpu() * mask)
            else:
                images.append(samples[i].squeeze().cpu())
            labels.append(f"T {i}/{timesteps}")
    if mask is not None:
        images.append(samples[-1].squeeze().cpu() * mask)
    else:
        images.append(samples[-1].squeeze().cpu())
    labels.append(f"Output: {timesteps}/{timesteps}")

    fig, axes = plt.subplots(1, n, figsize=(3*n, 3))

    for i in range(len(axes)):
        axes[i].get_xaxis().set_ticks([])
        axes[i].get_yaxis().set_ticks([])

        axes[i].imshow(np.rot90(images[i]), cmap='nipy_spectral', vmin=0, vmax=1)
        axes[i].set_xlabel(labels[i], fontsize=12)        
    
    cax0,kw0 = mpl.colorbar.make_axes(
        [ax for ax in axes.flat], location='right',
        shrink=0.80,
        #aspect=80,
        pad=0.01,
    )
    plt.colorbar(
        mpl.cm.ScalarMappable(
            norm=mpl.colors.Normalize(vmin=-1, vmax=1),
            cmap='nipy_spectral'
        ),
        cax=cax0,
        **kw0
    )

    if plot_file:
        plt.savefig(plot_file, format="pdf", bbox_inches="tight")
    else:
        plt.show()
    plt.close()
