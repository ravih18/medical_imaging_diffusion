import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as animation
import torch
import torchvision.utils as vutils
from PIL import Image
#from ..data.two_dim import data_distrib
import os, sys
matplotlib.use('Agg')


DPI = 200

def save_64_images_10_columns(images, filename="image_grid.png"):
    """
    Save a plot with 64 images arranged in a grid with 10 columns.

    Args:
        images (list of numpy arrays): A list of 64 images.
                                       Each image should be a numpy array (e.g., 28x28 grayscale or 32x32 RGB).
        filename (str): The name of the file to save the plot as.
                        Default is 'image_grid.png'.
    """
    if len(images) != 64:
        raise ValueError("You must provide exactly 64 images to create the grid.")

    vmin = images.min()
    vmax = images.max()
    
    # Calculate the number of rows (ceil(64 / 10) = 7 rows)
    nrows = math.ceil(len(images) / 10)
    ncols = 10
    
    # Create a figure with the specified number of rows and columns
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols, nrows))
    
    # Turn off axes for all subplots
    for ax in axes.flatten():
        ax.axis('off')
    
    # Plot each image in the grid
    for idx, ax in enumerate(axes.flatten()):
        if idx < len(images):
            ax.imshow(images[idx][0], cmap='gray', vmin=vmin, vmax=vmax)  # You can change 'gray' to another color map for RGB images
        else:
            ax.axis('off')  # Turn off empty subplots (if any)

    # Save the figure to the specified file
    plt.subplots_adjust(wspace=0, hspace=0)  # Remove spaces between images
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()
    #print(f"Grid of 64 images with 10 columns saved as {filename}")


def make_gif(plot_paths, output_directory='./gif', gif_name='gif'):
    frames = [Image.open(fn) for fn in plot_paths]

    frames[0].save(os.path.join(output_directory, f'{gif_name}.gif'),
                   format='GIF',
                   append_images=frames[1:],
                   save_all=True,
                   duration=100,
                   loop=0)


def save_gif(x_tot_plot, fps, gif_path):
    #print(x_tot_plot.shape)
    N_frame=x_tot_plot.shape[0]
    fig, ax = plt.subplots()
    img = x_tot_plot[0][0].squeeze().cpu()
    im = ax.imshow(img, cmap='grey', vmin=-1, vmax=1)

    def update(i):
        # Load the ith image
        if i < N_frame:
            img = x_tot_plot[i][0].squeeze().cpu()
        else:
            img = x_tot_plot[-1][0].squeeze().cpu()
        im.set_array(img)
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=N_frame + 2*fps, blit=True)
    ani.save(gif_path, writer='pillow', fps=fps) 
 

def save_sequence(num_steps, x, name='', img_dir='./img', gif_dir = './gif', xlim=None, ylim=None, ipf_it=None, freq=1):
    if not os.path.isdir(img_dir):
            os.mkdir(img_dir)
    if not os.path.isdir(gif_dir):
        os.mkdir(gif_dir)

    # PARTICLES (INIT AND FINAL DISTRIB)

    plot_paths = []
    for k in range(num_steps):
        if k % freq == 0:
            filename =  name + 'particle_' + str(k) + '.png'
            filename = os.path.join(img_dir, filename)
            plt.clf()
            if (xlim is not None) and (ylim is not None):
                plt.xlim(*xlim)
                plt.ylim(*ylim)
            plt.plot(x[-1, :, 0], x[-1, :, 1], '*')
            plt.plot(x[0, :, 0], x[0, :, 1], '*')
            plt.plot(x[k, :, 0], x[k, :, 1], '*')
            if ipf_it is not None:
                str_title = 'IPFP iteration: ' + str(ipf_it)
                plt.title(str_title)
                
            #plt.axis('equal')
            plt.savefig(filename, bbox_inches = 'tight', transparent = True, dpi=DPI)
            plot_paths.append(filename)

    # TRAJECTORIES

    N_part = 10
    filename = name + 'trajectory.png'
    filename = os.path.join(img_dir, filename)
    plt.clf()
    plt.plot(x[-1, :, 0], x[-1, :, 1], '*')
    plt.plot(x[0, :, 0], x[0, :, 1], '*')
    for j in range(N_part):
        xj = x[:, j, :]
        plt.plot(xj[:, 0], xj[:, 1], 'g', linewidth=2)
        plt.plot(xj[0,0], xj[0,1], 'rx')
        plt.plot(xj[-1,0], xj[-1,1], 'rx')
    plt.savefig(filename, bbox_inches = 'tight', transparent = True, dpi=DPI)

    make_gif(plot_paths, output_directory=gif_dir, gif_name=name)

    # REGISTRATION

    colors = np.cos(0.1 * x[0, :, 0]) * np.cos(0.1 * x[0, :, 1])

    name_gif = name + 'registration'
    plot_paths_reg = []
    for k in range(num_steps):
        if k % freq == 0:
            filename =  name + 'registration_' + str(k) + '.png'
            filename = os.path.join(img_dir, filename)
            plt.clf()
            if (xlim is not None) and (ylim is not None):
                plt.xlim(*xlim)
                plt.ylim(*ylim)
            plt.plot(x[-1, :, 0], x[-1, :, 1], '*', alpha=0)
            plt.plot(x[0, :, 0], x[0, :, 1], '*', alpha=0)
            plt.scatter(x[k, :, 0], x[k, :, 1], c=colors)
            if ipf_it is not None:
                str_title = 'IPFP iteration: ' + str(ipf_it)
                plt.title(str_title)            
            plt.savefig(filename, bbox_inches = 'tight', transparent = True, dpi=DPI)
            plot_paths_reg.append(filename)

    make_gif(plot_paths_reg, output_directory=gif_dir, gif_name=name_gif)

    # DENSITY

    name_gif = name + 'density'
    plot_paths_reg = []
    npts = 100
    for k in range(num_steps):
        if k % freq == 0:
            filename =  name + 'density_' + str(k) + '.png'
            filename = os.path.join(img_dir, filename)
            plt.clf()
            if (xlim is not None) and (ylim is not None):
                plt.xlim(*xlim)
                plt.ylim(*ylim)
            else:
                xlim = [-15, 15]
                ylim = [-15, 15]
            if ipf_it is not None:
                str_title = 'IPFP iteration: ' + str(ipf_it)
                plt.title(str_title)                            
            plt.hist2d(x[k, :, 0], x[k, :, 1], range=[[xlim[0], xlim[1]], [ylim[0], ylim[1]]], bins=npts)
            plt.savefig(filename, bbox_inches = 'tight', transparent = True, dpi=DPI)
            plot_paths_reg.append(filename)

    make_gif(plot_paths_reg, output_directory=gif_dir, gif_name=name_gif)    
            

class Plotter(object):

    def __init__(self):
        pass

    def plot(self, x_tot_plot, net, i, n, forward_or_backward):
        pass

    def __call__(self, initial_sample, x_tot_plot, net, i, n, forward_or_backward):
        self.plot(initial_sample, x_tot_plot, net, i, n, forward_or_backward)


class ImPlotter(object):

    def __init__(self, img_dir = './img', gif_dir='./gif', plot_level=3):
        if not os.path.isdir(img_dir):
            os.mkdir(img_dir)
        if not os.path.isdir(gif_dir):
            os.mkdir(gif_dir)
        self.img_dir = img_dir
        self.gif_dir = gif_dir
        self.num_plots = 100
        self.num_digits = 20
        self.plot_level = plot_level

    def plot(self, initial_sample, x_tot_plot, i, n, forward_or_backward):
        if self.plot_level > 0:
            x_tot_plot = x_tot_plot[:,:self.num_plots]
            name = f"{forward_or_backward}_{n}_{i}"
            img_dir = os.path.join(self.img_dir, name)

            if not os.path.isdir(img_dir):
                os.mkdir(img_dir)         

            if self.plot_level > 0:
                plt.clf()
                filename_grid_png = os.path.join(img_dir, 'im_grid_first.png')
                #vutils.save_image(initial_sample, filename_grid_png, nrow=10)
                save_64_images_10_columns(initial_sample.detach().cpu().numpy(), filename_grid_png)
                filename_grid_png = os.path.join(img_dir, 'im_grid_final.png')
                #vutils.save_image(x_tot_plot[-1], filename_grid_png, nrow=10)
                save_64_images_10_columns(x_tot_plot[-1].detach().cpu().numpy(), filename_grid_png)
                save_gif(
                    x_tot_plot,
                    fps=4,
                    gif_path=self.gif_dir / f"traj_{forward_or_backward}_{n}_{i}_path.gif"
                )

            if self.plot_level >= 2:
                plt.clf()
                plot_paths = []
                num_steps, num_particles, channels, H, W = x_tot_plot.shape
                plot_steps = np.linspace(0,num_steps-1,self.num_plots, dtype=int) 

                for k in plot_steps:
                    # save png
                    filename_grid_png = os.path.join(img_dir, 'im_grid_{0}.png'.format(k))    
                    plot_paths.append(filename_grid_png)
                    vutils.save_image(x_tot_plot[k], filename_grid_png, nrow=10)

                make_gif(plot_paths, output_directory=self.gif_dir, gif_name=name)

    def __call__(self, initial_sample, x_tot_plot, i, n, forward_or_backward):
        self.plot(initial_sample, x_tot_plot, i, n, forward_or_backward)


class TwoDPlotter(Plotter):

    def __init__(self, num_steps, gammas, img_dir = './img', gif_dir='./gif'):

        if not os.path.isdir(img_dir):
            os.mkdir(img_dir)
        if not os.path.isdir(gif_dir):
            os.mkdir(gif_dir)

        self.img_dir = img_dir
        self.gif_dir = gif_dir

        self.num_steps = num_steps
        self.gammas = gammas

    def plot(self, initial_sample, x_tot_plot, i, n, forward_or_backward):
        fb = forward_or_backward
        ipf_it = n
        x_tot_plot = x_tot_plot.cpu().numpy()
        name = str(i) + '_' + fb +'_' + str(n) + '_'

        save_sequence(num_steps=self.num_steps, x=x_tot_plot, name=name, xlim=(-15,15),
                      ylim=(-15,15), ipf_it=ipf_it, freq=self.num_steps//min(self.num_steps,50),
                      img_dir=self.img_dir, gif_dir=self.gif_dir)


    def __call__(self, initial_sample, x_tot_plot, i, n, forward_or_backward):
        self.plot(initial_sample, x_tot_plot, i, n, forward_or_backward)


# def get_plotter(runner, args):
#     dataset_tag = getattr(args, DATASET)
#     if dataset_tag == DATASET_2D:
#         return TwoDPlotter(num_steps=runner.num_steps, gammas=runner.langevin.gammas)
#     else:
#         return ImPlotter(plot_level = args.plot_level)