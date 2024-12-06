import torch 
from torch import nn
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm
import json
import time

from .utils import extract, batch_psnr
from .time_scheduler import quadratic_beta_schedule

class DiffusionModel(nn.Module):

    def __init__(self, model, diffusion_config, device, experiment_directory=None):
        super(DiffusionModel, self).__init__()
        self.experiment_directory = experiment_directory
        
        self.model = model
        self.loss_type = diffusion_config.loss_type
        self.device = device

        self.timesteps = diffusion_config.timesteps
        self.betas = quadratic_beta_schedule(
            self.timesteps,
            beta_start=diffusion_config.beta_start,
            beta_end=diffusion_config.beta_end,
        )
        # define alphas 
        alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - alphas_cumprod_prev) / (1. - self.alphas_cumprod)

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

    def train(self, epochs, optimizer, trainloader, valloader=None, scheduler=None):
        print("\n\nStart training!")
        best_val_loss = np.inf
        for epoch in range(epochs):
            train_loss = 0
            print(f"Training epochs {epoch}/{epochs}")
            start_time = time.time()
            for step, batch in enumerate(trainloader):
                optimizer.zero_grad()

                batch_size = batch["T1"].shape[0]
                batch = batch["T1"].to(torch.float).to(self.device)

                t = torch.randint(0, self.timesteps, (batch_size,), device=self.device).long()

                loss = self.p_losses(batch, t)

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
            print(f"  Train Loss:\t{(train_loss/len(trainloader)):.6f}", )

            if valloader is not None:
                val_loss = 0
                with torch.no_grad():
                    for _, batch in enumerate(valloader):
                        batch = batch["T1"].to(torch.float).to(self.device)

                        t = torch.randint(0, self.timesteps, (batch_size,), device=self.device).long()
                        
                        loss = self.p_losses(batch, t ,loss_type=self.loss_type)
                        
                        val_loss += loss.item()
                print(f"  Val Loss:\t{(val_loss/len(valloader)):.6f}")

            print(f"  Time: {(time.time() - start_time):.3f} seconds")

            if (scheduler is not None) and epoch%10==9:
                scheduler.step()
                print(f"  Learning rate after scheduler step: {scheduler.get_last_lr()[0]}")

            if self.experiment_directory is not None:
                self._save_checkpoint(epoch)
                if valloader is not None:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        self.save_model(epoch, best_val_loss)
                
        print("End of training!")

        if self.experiment_directory is not None and valloader is None:
            self.save_model(epochs, val_loss)
            print(f"Saved model after last epoch.")

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

    def denoise(self, x_start, t, noise=None, plot=False):
        # Input (1,1,M,N)
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = self.model(x_noisy.to(self.device), t.to(self.device)).detach().cpu()
        
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)[0,0,0,0]
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)[0,0,0,0]

        denoised_x_start = (x_noisy-predicted_noise*sqrt_one_minus_alphas_cumprod_t)/sqrt_alphas_cumprod_t
        
        MSE_noisy = torch.mean((x_noisy/sqrt_alphas_cumprod_t-x_start)**2).numpy()
        MSE_denoised = torch.mean((denoised_x_start-x_start)**2).numpy()
        
        PSNR_noisy = 10*np.log10(2**2/MSE_noisy)
        PSNR_denoised = 10*np.log10(2**2/MSE_denoised)
        
        MSE_noisy_theory = (sqrt_one_minus_alphas_cumprod_t/sqrt_alphas_cumprod_t)**2
        PSNR_noisy_theory = 10 * np.log10(2**2/MSE_noisy_theory)

        if plot:
            
            print('Sigma=', (sqrt_one_minus_alphas_cumprod_t/sqrt_alphas_cumprod_t).numpy(),'. Values of vmin and vmax are set according to the original image colormap.')
            
            vmin = x_start[0,0].cpu().min()
            vmax = x_start[0,0].cpu().max()
            
            plt.figure(figsize=(12,5))
            plt.subplot(131)
            plt.imshow(x_noisy[0,0].cpu()/sqrt_alphas_cumprod_t, vmin=vmin, vmax=vmax)
            plt.title('Noisy')
            plt.subplot(132)
            plt.imshow(denoised_x_start[0,0].detach().cpu(), vmin=vmin, vmax=vmax)
            plt.title('Denoised')
            plt.subplot(133)
            plt.imshow(x_start[0,0].cpu(), vmin=vmin, vmax=vmax)
            plt.title('Original')
            plt.show()
            
            print('Original MSE/PSNR:', MSE_noisy, '/', PSNR_noisy,', theoretical values:', MSE_noisy_theory, '/', PSNR_noisy_theory)
            print('Denoised MSE/PSNR:', MSE_denoised, '/', PSNR_denoised, '.')

    def _save_checkpoint(self, epoch):
        checkpoint_dir = self.experiment_directory / "checkpoint"
        checkpoint_dir.mkdir(parents = True, exist_ok = True)
        torch.save(self.model.state_dict(), (checkpoint_dir / "denoiser_unet_checkpoint.pt"))
        with open((checkpoint_dir / "checkpoint.json"), 'w') as fp:
            json.dump({"current_epochs": epoch}, fp)

    def save_model(self, epoch, val_loss):
        model_dir = self.experiment_directory / "best_loss"
        model_dir.mkdir(parents = True, exist_ok = True)
        torch.save(self.model.state_dict(), (model_dir / "denoiser_unet.pt"))
        with open((model_dir / "best_model.json"), 'w') as fp:
            json.dump(
                {
                    "current_epochs": epoch,
                    "validation_loss": val_loss,
                },
                fp
            )
