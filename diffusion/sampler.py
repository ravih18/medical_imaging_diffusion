import torch

from tqdm import tqdm

from .utils import extract


@torch.no_grad()
def p_sample(diffusion_model, x, t, t_index):
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
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise 

# Algorithm 2 (including returning all images)
@torch.no_grad()
def p_sample_loop(diffusion_model, shape):
    device = diffusion_model.device
    print(diffusion_model.timesteps)
    timesteps = diffusion_model.timesteps

    b = shape[0]
    # start from pure noise (for each example in the batch)
    img = torch.randn(shape, device=device)
    imgs = []

    for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
        img = p_sample(diffusion_model, img, torch.full((b,), i, device=device, dtype=torch.long), i)
        imgs.append(img.cpu().numpy())
    return imgs

@torch.no_grad()
def sample(diffusion_model, image_size, batch_size=16, channels=3):
    return p_sample_loop(diffusion_model, shape=(batch_size, channels, image_size, image_size))
