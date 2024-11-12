import torch 
from torch import nn
import torch.nn.functional as F

from tqdm import tqdm

from .utils import extract

class DiffusionModel(nn.Module):

    def __init__(self, model, timesteps, betas, device, loss_type='l1'):
        super(DiffusionModel, self).__init__()
        self.model = model
        self.loss_type = loss_type
        self.device = device

        self.timesteps = timesteps
        self.betas = betas
        # define alphas 
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    def q_sample(self, x_start, t, noise=None):
        # forward diffusion (using the nice property)
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = self.model(x_noisy, t)

        if self.loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif self.loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        elif self.loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()

        return loss

    def train(self, epochs, optimizer, trainloader, valloader=None):
        for epoch in range(epochs):
            accumulated_losses = 0
            for step, batch in tqdm(enumerate(trainloader), f"Epoch {epoch}", total=len(trainloader)):
                optimizer.zero_grad()

                batch_size = batch["T1"].shape[0]
                batch = batch["T1"].to(torch.float).to(self.device)

                # Algorithm 1 line 3: sample t uniformally for every example in the batch
                t = torch.randint(0, self.timesteps, (batch_size,), device=self.device).long()

                loss = self.p_losses(batch, t)

                loss.backward()
                optimizer.step()

                accumulated_losses += loss.item()
            print(f"Train Loss: {accumulated_losses/len(trainloader)}", )

            if valloader is not None:
                val_loss = 0
                with torch.no_grad():
                    for _, batch in enumerate(valloader):
                        batch = batch["T1"].to(torch.float).to(self.device)
                        t = torch.randint(0, self.timesteps, (batch_size,), device=self.device).long()
                        val_loss += self.p_losses(batch, t).item()
                print(f"Val Loss: {val_loss/len(valloader)}")
