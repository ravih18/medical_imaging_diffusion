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


def make_diff_plot(x, y, gt, plot_file=None):
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))

    imgs = [x, y, y-x, gt]

    for i in range(len(axes)):
        axes[i].get_xaxis().set_ticks([])
        axes[i].get_yaxis().set_ticks([])

        if i==2:
            cmap = 'seismic'
        else:
            cmap = 'nipy_spectral'

        axes[i].imshow(imgs[i], cmap=cmap, vmin=-1, vmax=1)
    
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


def make_traj_plot(input_img, samples, plot_file=None, n = 6):

    images = [input_img]
    for i in range(samples.shape[0]-1):
        if i // (n-1):
            images.append(samples[i].squeeze().cpu())
    images.append(samples[-1].squeeze().cpu())

    fig, axes = plt.subplots(1, n, figsize=(3*n, 3))

    for i in range(len(axes)):
        axes[i].get_xaxis().set_ticks([])
        axes[i].get_yaxis().set_ticks([])

        axes[i].imshow(images[i], cmap='nipy_spectral', vmin=-1, vmax=1)            
    
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