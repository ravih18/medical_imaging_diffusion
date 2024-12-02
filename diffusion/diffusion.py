import torch 
from torch import nn
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm

from .utils import extract, batch_psnr

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
        self.alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def q_sample(self, x_start, t, noise=None):
        # forward diffusion (using the nice property)
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_sample(self, x, pred_noise, t):
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)
        
        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * pred_noise / sqrt_one_minus_alphas_cumprod_t
        )
        return model_mean

    def p_losses(self, x_start, t, noise=None, loss_type='l2'):
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = self.model(x_noisy, t)

        if loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()
        return loss

    def train(self, epochs, optimizer, trainloader, valloader=None):
        
        loss_train_list, loss_val_list = [], []
        for epoch in range(epochs):
            train_loss = 0
            for step, batch in tqdm(enumerate(trainloader), f"Epoch {epoch}", total=len(trainloader)):
                optimizer.zero_grad()

                batch_size = batch["T1"].shape[0]
                batch = batch["T1"].to(torch.float).to(self.device)

                t = torch.randint(0, self.timesteps, (batch_size,), device=self.device).long()

                loss = self.p_losses(batch, t)

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
            print(f"Train Loss: {train_loss/len(trainloader)}", )

            if valloader is not None:
                val_loss = 0
                #val_psnr = 0
                with torch.no_grad():
                    for _, batch in enumerate(valloader):
                        batch = batch["T1"].to(torch.float).to(self.device)

                        t = torch.randint(0, self.timesteps, (batch_size,), device=self.device).long()
                        
                        loss = self.p_losses(batch, t ,loss_type=self.loss_type)
                        
                        val_loss += loss.item()
                print(f"Val Loss: {val_loss/len(valloader)}")
        print("End of training!")

    def compute_psnr(self, dataloader, psnr_step=10):
        psnr_out_list = []
        psnr_in_list = []
        for t in tqdm(range(0, self.timesteps, psnr_step)):
            loss_accumulated = 0
            for step, batch in enumerate(dataloader):
                batch = batch["T1"].to(torch.float).to(self.device)
                
                t_tensor = torch.full_like(batch, t, device=self.device).long()

                loss = self.p_losses(batch, t_tensor, loss_type='l2')
                loss_accumulated += loss.item() / len(dataloader.batch_size)

            alphas_cumprod_t = self.alphas_cumprod[t]
            psnr_out = 10 * (np.log10(2**2) - torch.log10((1 - alphas_cumprod_t) / alphas_cumprod_t * loss_accumulated))
            psnr_in = 10 * np.log10(2**2 * alphas_cumprod_t/(1 - alphas_cumprod_t))

            psnr_in_list.append(psnr_in.numpy())
            psnr_out_list.append(psnr_out.numpy())

        return np.array(psnr_in_list), np.array(psnr_out_list)

