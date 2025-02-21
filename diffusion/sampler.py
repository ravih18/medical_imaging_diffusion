import torch
from tqdm.notebook import tqdm
from random import choice
import matplotlib.pyplot as plt
from .utils import extract

@torch.no_grad()
def p_sample(diffusion_model, x, t, t_index, noise=None):
    betas_t = extract(diffusion_model.betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        diffusion_model.sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(diffusion_model.sqrt_recip_alphas, t, x.shape)
    
    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * diffusion_model.model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(diffusion_model.posterior_variance, t, x.shape)
        if noise is None:
            noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise 

# Algorithm 2 (including returning all images)
@torch.no_grad()
def p_sample_loop(diffusion_model, shape, noises=None):
    device = diffusion_model.device
    print(diffusion_model.timesteps)
    timesteps = diffusion_model.timesteps

    b = shape[0]
    # start from pure noise (for each example in the batch)
    if noises is None:
        img = torch.randn(shape, device=device)
    else:
        img = noises[0]
    imgs = []

    for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
        if noises is None:
            img = p_sample(diffusion_model, img, torch.full((b,), i, device=device, dtype=torch.long), i)
        else:
            img = p_sample(diffusion_model, img, torch.full((b,), i, device=device, dtype=torch.long), i, noises[i+1])
        imgs.append(img.cpu().numpy())
    return imgs

@torch.no_grad()
def sample(diffusion_model, image_size, batch_size=16, channels=3, noises=None):
    return p_sample_loop(diffusion_model, shape=(batch_size, channels, image_size, image_size), noises=noises)
    
@torch.no_grad()
def save_sample(diffusion_model, image_size, n_images, channels, experiment_dir):
    samples = sample(diffusion_model, image_size=image_size, batch_size=n_images, channels=channels)

    fig, axs = plt.subplots(1, n_images, figsize=(2*n_images, 8))
    for i in range(n_images):
        axs[i].get_xaxis().set_visible(False)
        axs[i].get_yaxis().set_visible(False)
        axs[i].set_title(f"Image {i}")
        axs[i].imshow(samples[-1][i].reshape(image_size, image_size, channels), cmap="gray")
    fig.savefig(experiment_dir / "samples.pdf", format="pdf", bbox_inches="tight") 
