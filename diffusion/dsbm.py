import os, sys, warnings, time
from pathlib import Path
import re
from collections import OrderedDict
from functools import partial

#import hydra
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import glob

from image_datasets.cacheloader_dsbm import DBDSB_CacheLoader
from image_datasets import repeater
from .sde import *
#from .runners import *
#from .runners.config_getters import get_model, get_optimizer, get_plotter, get_logger
from model.ema import EMAHelper
from utils.config import DsbmConfig

# from torchdyn.core import NeuralODE
from torchdiffeq import odeint


class DSBM_IMF:
    def __init__(
        self,
        experiment_directory: Path,
        dsbm_params: DsbmConfig,
        datasets: dict,
        transfer: bool,
        #init_ds,
        #final_ds,
        #mean_final,
        #var_final,
        #args,
        #accelerator=None,
        #final_cond_model=None,
        #valid_ds=None,
        #test_ds=None
    ):
        #self.accelerator = accelerator
        #self.device = self.accelerator.device  # local device for each process
        self.experiment_directory = experiment_directory
        self.transfer = transfer
        self.init_expe(experiment_directory)

        self.device = "cuda" if torch.cuda.is_available() else 'cpu'

        #self.args = args
        self.cdsb = False #self.cdsb = self.args.cdsb  # Conditional

        self.init_ds = datasets["train_init"]
        self.final_ds = datasets["train_final"]
        self.valid_ds = datasets["val_init"]
        self.test_ds = None

        self.mean_final = datasets["mean_final"]
        self.var_final = datasets["var_final"]
        self.std_final = torch.sqrt(self.var_final) if self.var_final is not None else None

        # IPF/IMF params
        self.n_ipf = dsbm_params.n_ipf
        self.use_prev_net = dsbm_params.use_prev_net
        self.first_coupling = dsbm_params.first_coupling
        self.cache_batch_size = dsbm_params.cache_batch_size
        self.cache_refresh_stride = dsbm_params.cache_refresh_stride

        # training params
        self.lr = dsbm_params.lr
        self.num_iter = dsbm_params.num_iter
        self.batch_size = dsbm_params.batch_size
        self.num_repeat_data = dsbm_params.num_repeat_data 
        assert self.batch_size % self.num_repeat_data == 0
        self.first_num_iter = dsbm_params.first_num_iter if dsbm_params.first_num_iter is not None else self.num_iter
        # assert self.first_num_iter % self.num_iter == 0
        #self.normalize_x1 = self.args.get("normalize_x1", False) # unused
        self.ema = dsbm_params.ema
        self.grad_clipping = dsbm_params.grad_clipping
        self.grad_clip = dsbm_params.grad_clip
        self.std_trick = dsbm_params.std_trick
        self.loss_scale = dsbm_params.loss_scale

        # sampling param
        self.ode_sampler = dsbm_params.ode_sampler
        self.ode_tol = dsbm_params.ode_tol
        self.ode_euler_step_size = dsbm_params.ode_euler_step_size

        # Gammas
        self.num_steps = dsbm_params.num_steps
        if dsbm_params.symmetric_gamma:
            n = self.num_steps // 2
            if dsbm_params.gamma_space == 'linspace':
                gamma_half = np.linspace(dsbm_params.gamma_min, dsbm_params.gamma_max, n)
            elif dsbm_params.gamma_space == 'geomspace':
                gamma_half = np.geomspace(dsbm_params.gamma_min, dsbm_params.gamma_max, n)
            self.gammas = np.concatenate([gamma_half, np.flip(gamma_half)])
        else:
            if dsbm_params.gamma_space == 'linspace':
                self.gammas = np.linspace(dsbm_params.gamma_min, dsbm_params.gamma_max, self.num_steps)
            elif dsbm_params.gamma_space == 'geomspace':
                self.gammas = np.geomspace(dsbm_params.gamma_min, dsbm_params.gamma_max, self.num_steps)
        self.gammas = torch.tensor(self.gammas).to(self.device).float()
        self.T = torch.sum(self.gammas)
        #print("T:", self.T.item())
        self.sigma = torch.sqrt(self.T).item()
        self.timesteps = self.gammas / self.T

        # run from checkpoint
        #self.build_checkpoints()
        
        # get models
        self.build_models(dsbm_params.unet_param)
        self.build_ema(dsbm_params.ema_rate)

        # get optims
        self.build_optimizers()

        # get data
        self.data_num_workers = dsbm_params.data_param.num_workers
        self.data_pin_memory = dsbm_params.data_param.pin_memory
        self.build_dataloaders()

        # langevin
        if dsbm_params.sde == "ve":
            self.langevin = DBDSB_VE(
                self.sigma,
                self.num_steps,
                self.timesteps,
                self.shape_x,
                self.shape_y,
                self.first_coupling
            )
        elif dsbm_params.sde == "vp":
            self.langevin = DBDSB_VP(
                self.sigma,
                self.num_steps,
                self.timesteps,
                self.shape_x,
                self.shape_y,
                self.first_coupling
            )

        self.npar = len(self.init_ds)
        self.cache_npar = dsbm_params.cache_npar if dsbm_params.cache_npar is not None else self.batch_size * self.cache_refresh_stride // self.num_repeat_data
        self.cache_epochs = (self.batch_size * self.cache_refresh_stride) / self.cache_npar             # How many passes of the cached dataset does each outer iteration perform
        self.data_epochs = (self.num_iter * self.cache_npar) / (self.npar * self.cache_refresh_stride)  # How many repeats the training dataset has in the cache
        print("Cache npar:", self.cache_npar)
        print("Num repeat data:", self.num_repeat_data)
        print("Cache epochs:", self.cache_epochs)
        print("Data epochs:", self.data_epochs)

        self.cache_num_steps = dsbm_params.cache_num_steps if dsbm_params.cache_num_steps is not None else self.num_steps
        self.test_num_steps = dsbm_params.test_num_steps if dsbm_params.test_num_steps is not None else self.num_steps

        # get loggers
        self.logger = self.get_logger('train_logs')
        self.save_logger = self.get_logger('test_logs')
        self.plotter = self.get_plotter()
        self.stride = 5000
        self.stride_log = 100

        self.checkpoint_it = 1
        self.checkpoint_pass = 'b'
        # self.first_pass = True
        # self.step = 1

    def init_expe(self, experiment_directory: Path):
        # Make logs, img/gifs, checkpoints directory
        (experiment_directory / "logs").mkdir(parents = True, exist_ok = True)
        (experiment_directory / "imgs").mkdir(parents = True, exist_ok = True)
        (experiment_directory / "gifs").mkdir(parents = True, exist_ok = True)
        (experiment_directory / "checkpoints").mkdir(parents = True, exist_ok = True)

    def get_logger(self, name='logs'):
        from utils.logger import CSVLogger

        logger = CSVLogger(
            directory=self.experiment_directory / 'logs',
            name=name
        )
        return logger

    def get_plotter(self):
        from utils.plotters import ImPlotter

        plotter = ImPlotter(
            plot_level = 1,
            img_dir=self.experiment_directory / 'imgs',
            gif_dir=self.experiment_directory / 'gifs',
        )
        return plotter

    # def build_checkpoints(self):
    #     self.first_pass = True  # Load and use checkpointed networks during first pass
    #     self.ckpt_dir = './checkpoints/'
    #     self.ckpt_prefixes = ["net_b", "sample_net_b", "optimizer_b", "net_f", "sample_net_f", "optimizer_f"]
    #     self.cache_dir='./cache/'
    #     if self.accelerator.is_main_process:
    #         os.makedirs(self.ckpt_dir, exist_ok=True)
    #         os.makedirs(self.cache_dir, exist_ok=True)

    #     if self.args.get('checkpoint_run', False):
    #         self.resume, self.checkpoint_it, self.checkpoint_pass, self.step = \
    #             True, self.args.checkpoint_it, self.args.checkpoint_pass, self.args.checkpoint_iter
    #         print(f"Resuming training at iter {self.checkpoint_it} {self.checkpoint_pass} step {self.step}")

    #         self.checkpoint_b = hydra.utils.to_absolute_path(self.args.checkpoint_b)
    #         self.sample_checkpoint_b = hydra.utils.to_absolute_path(self.args.sample_checkpoint_b)
    #         self.optimizer_checkpoint_b = hydra.utils.to_absolute_path(self.args.optimizer_checkpoint_b)

    #         self.resume_f = False
    #         if self.args.checkpoint_f is not None:
    #             self.resume_f = True
    #             self.checkpoint_f = hydra.utils.to_absolute_path(self.args.checkpoint_f)
    #             self.sample_checkpoint_f = hydra.utils.to_absolute_path(self.args.sample_checkpoint_f)
    #             self.optimizer_checkpoint_f = hydra.utils.to_absolute_path(self.args.optimizer_checkpoint_f)
            
    #     else:
    #         self.ckpt_dir_load = os.path.abspath(self.ckpt_dir)
    #         ckpt_dir_load_list = os.path.normpath(self.ckpt_dir_load).split(os.sep)
    #         if 'test' in ckpt_dir_load_list:
    #             self.ckpt_dir_load = os.path.join(os.sep, *ckpt_dir_load_list[:ckpt_dir_load_list.index('test')], "checkpoints/")
    #         self.resume, self.checkpoint_it, self.checkpoint_pass, self.step, ckpt_b_suffix, ckpt_f_suffix = self.find_last_ckpt()

    #         self.resume_f = False
    #         if self.resume:
    #             if not self.args.autostart_next_it and self.step == 1 and not (self.checkpoint_it == 1 and self.checkpoint_pass == 'b'): 
    #                 self.checkpoint_pass, self.checkpoint_it = self.compute_prev_it(self.checkpoint_pass, self.checkpoint_it)
    #                 self.step = self.compute_max_iter(self.checkpoint_pass, self.checkpoint_it) + 1

    #             print(f"Resuming training at iter {self.checkpoint_it} {self.checkpoint_pass} step {self.step}")
    #             self.checkpoint_b, self.sample_checkpoint_b, self.optimizer_checkpoint_b = [os.path.join(self.ckpt_dir_load, f"{ckpt_prefix}_{ckpt_b_suffix}.ckpt") for ckpt_prefix in self.ckpt_prefixes[:3]]
    #             if ckpt_f_suffix is not None:
    #                 self.resume_f = True
    #                 self.checkpoint_f, self.sample_checkpoint_f, self.optimizer_checkpoint_f = [os.path.join(self.ckpt_dir_load, f"{ckpt_prefix}_{ckpt_f_suffix}.ckpt") for ckpt_prefix in self.ckpt_prefixes[3:]]

    def build_models(self, unet_config, forward_or_backward=None):
        # running network
        #from model.unet import Unet

        #net_f, net_b = Unet(**unet_config.model_dump()), Unet(**unet_config.model_dump())

        from model.unet_dsbm import UNetModel

        kwargs = {
            "in_channels": 1,
            "model_channels": 128,
            "out_channels": 1,
            "num_res_blocks": 2,
            "attention_resolutions": (4, 8), # tuple(img_size / 16, img_size / 8)
            "dropout": 0.1,
            "channel_mult": (1, 2, 2, 2),
            "num_classes": None,
            "use_checkpoint": False,
            "num_heads": 4,
            "use_scale_shift_norm": True,
            "resblock_updown": False,
            "temb_scale": 1000,
        }

        net_f, net_b = UNetModel(**kwargs), UNetModel(**kwargs)

        if forward_or_backward is None:
            net_f = net_f.to(self.device)
            net_b = net_b.to(self.device)
            self.net = torch.nn.ModuleDict({'f': net_f, 'b': net_b})
        if forward_or_backward == 'f':
            net_f = net_f.to(self.device)
            self.net.update({'f': net_f})
        if forward_or_backward == 'b':
            net_b = net_b.to(self.device)
            self.net.update({'b': net_b})

    # def accelerate(self, forward_or_backward):
    #     (self.net[forward_or_backward], self.optimizer[forward_or_backward]) = self.accelerator.prepare(
    #         self.net[forward_or_backward], self.optimizer[forward_or_backward])

    def update_ema(self, forward_or_backward):
        if self.ema:
            self.ema_helpers[forward_or_backward] = EMAHelper(
                mu=self.ema_rate,
                device=self.device
            )
            self.ema_helpers[forward_or_backward].register(
                self.net[forward_or_backward]
            )

    def build_ema(self, ema_rate):
        if self.ema:
            self.ema_rate = ema_rate
            self.ema_helpers = {}
            self.update_ema('f')
            self.update_ema('b')

    def build_optimizers(self, forward_or_backward=None):
        from torch.optim import Adam

        optimizer_b = Adam(self.net['b'].parameters(), lr=self.lr)
        optimizer_f = Adam(self.net['f'].parameters(), lr=self.lr)
        self.optimizer = {'f': optimizer_f, 'b': optimizer_b}

    def build_dataloader(self, ds, batch_size, shuffle=True, drop_last=True, repeat=True):
        # def worker_init_fn(worker_id):
        #     np.random.seed(np.random.get_state()[1][0] + worker_id + self.accelerator.process_index * self.args.num_workers)
        dl_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": self.data_pin_memory,
        #     "worker_init_fn": worker_init_fn
        }

        dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, **dl_kwargs)
        # dl = self.accelerator.prepare(dl)
        if repeat:
            dl = repeater(dl)
        return dl

    def build_dataloaders(self):
        #self.plot_npar = min(self.args.plot_npar, len(self.init_ds)) # seems to be unused
        #self.test_npar = min(self.args.test_npar, len(self.init_ds))

        #self.cache_batch_size = min(self.args.cache_batch_size * self.accelerator.num_processes, len(self.init_ds))  # Adjust automatically to num_processes
        #self.test_batch_size = min(self.args.test_batch_size * self.accelerator.num_processes, len(self.init_ds))    # Adjust automatically to num_processes

        self.init_dl = self.build_dataloader(
            self.init_ds,
            batch_size=self.batch_size // self.num_repeat_data
        )
        self.cache_init_dl = self.build_dataloader(
            self.init_ds,
            batch_size=self.cache_batch_size,
        )

        self.save_init_dl = self.build_dataloader(
            self.init_ds,
            batch_size=self.batch_size, #batch_size=self.test_batch_size,
            shuffle=False,
            repeat=False
        )
        self.save_dls_dict = {"train": self.save_init_dl}

        #if self.final_ds is not None:
        self.final_dl = self.build_dataloader(
            self.final_ds,
            batch_size=self.batch_size // self.num_repeat_data
        )
        self.cache_final_dl = self.build_dataloader(
            self.final_ds,
            batch_size=self.cache_batch_size
        )

        self.save_final_dl_repeat = self.build_dataloader(
            self.final_ds,
            batch_size=self.batch_size, #batch_size=self.test_batch_size
        )
        self.save_final_dl = self.build_dataloader(
            self.final_ds,
            batch_size=self.batch_size, #batch_size=self.test_batch_size,
            shuffle=False,
            repeat=False
        )
        # else:
        #     self.final_dl = None
        #     self.cache_final_dl = None

        #     self.save_final_dl = None
        #     self.save_final_dl_repeat = None

        if self.valid_ds is not None:
            self.save_valid_dl = self.build_dataloader(
                self.valid_ds,
                batch_size=self.batch_size, #batch_size=self.test_batch_size,
                shuffle=False,
                repeat=False
            )
            self.save_dls_dict["valid"] = self.save_valid_dl

        if self.test_ds is not None:
            self.save_test_dl = self.build_dataloader(
                self.test_ds,
                batch_size=self.batch_size, #batch_size=self.test_batch_size,
                shuffle=False,
                repeat=False
                )
            self.save_dls_dict["test"] = self.save_test_dl

        batch = next(self.cache_init_dl)["image"]
        self.shape_x = batch[0][0].shape
        self.shape_y = batch[1][0].shape

    def get_sample_net(self, fb):
        if self.ema:
            #sample_net = self.ema_helpers[fb].ema_copy(self.accelerator.unwrap_model(self.net[fb]))
            sample_net = self.ema_helpers[fb].ema_copy(self.net[fb])
        else:
            sample_net = self.net[fb]
        sample_net = sample_net.to(self.device)
        sample_net.eval()
        return sample_net

    def new_cacheloader(self, sample_direction, n, build_dataloader=True, refresh_idx=0):
        sample_net = self.get_sample_net(sample_direction)
        sample_fn = partial(self.apply_net, net=sample_net, fb=sample_direction)

        if (n == 1) and (sample_direction == 'b'):  # For training 1f
            refresh_tot = int(np.ceil(self.first_num_iter/self.cache_refresh_stride))
        else:
            refresh_tot = int(np.ceil(self.num_iter/self.cache_refresh_stride))

        assert refresh_idx < refresh_tot

        cache_npar = self.cache_npar
        assert cache_npar % self.cache_batch_size == 0
        num_batches = cache_npar // self.cache_batch_size
        
        new_ds = DBDSB_CacheLoader(sample_direction,
                                   sample_fn,
                                   self.cache_init_dl,
                                   self.cache_final_dl,
                                   num_batches,
                                   self.langevin, self, n, 
                                   refresh_idx=refresh_idx, refresh_tot=refresh_tot)

        if build_dataloader:
            new_dl = self.build_dataloader(new_ds, batch_size=self.batch_size // self.num_repeat_data)
            return new_dl

    def train(self):
        for n in range(self.checkpoint_it, self.n_ipf + 1):

            print('IPF iteration: ' + str(n) + '/' + str(self.n_ipf))
            # BACKWARD OPTIMISATION
            if (self.checkpoint_pass == 'f') and (n == self.checkpoint_it):
                self.ipf_iter('f', n)
            else:
                self.ipf_iter('b', n)
                self.ipf_iter('f', n)

    def sample_batch(self, init_dl, final_dl):
        mean_final = self.mean_final
        std_final = self.std_final

        init_batch = next(init_dl)["image"]
        init_batch_x = init_batch[0]
        init_batch_y = init_batch[1]

        if self.final_ds is not None:
            final_batch = next(final_dl)["image"]
            final_batch_x = final_batch[0]

        else:
            mean_final = mean_final.to(init_batch_x.device)
            std_final = std_final.to(init_batch_x.device)
            final_batch_x = mean_final + std_final * torch.randn_like(init_batch_x)

        mean_final = mean_final.to(init_batch_x.device)
        std_final = std_final.to(init_batch_x.device)
        var_final = std_final ** 2

        if not self.cdsb:
            init_batch_y = None

        return init_batch_x, init_batch_y, final_batch_x, mean_final, var_final

    def backward_sample(self, final_batch_x, y_c, permute=True, num_steps=None):
        sample_net = self.get_sample_net('b')
        sample_net.eval()
        sample_fn = partial(self.apply_net, net=sample_net, fb="b")

        with torch.no_grad():
            # self.set_seed(seed=0 + self.accelerator.process_index)
            final_batch_x = final_batch_x.to(self.device)
            if self.cdsb:
                y_c = y_c.expand(final_batch_x.shape[0], *self.shape_y).clone().to(self.device)
            x_tot_c, _, _, _ = self.langevin.record_langevin_seq(sample_fn, final_batch_x, y_c, 'b', sample=True, num_steps=num_steps)

            if permute:
                x_tot_c = x_tot_c.permute(1, 0, *list(range(2, len(x_tot_c.shape))))  # (num_steps, num_samples, *shape_x)

        return x_tot_c, self.num_steps if num_steps is None else num_steps

    def backward_sample_ode(self, final_batch_x, y_c, permute=True):
        sample_net_b = self.get_sample_net('b')
        sample_net_b.eval()
        sample_fn_b = partial(self.apply_net, net=sample_net_b, fb="b")
        try:
            sample_net_f = self.get_sample_net('f')
            sample_net_f.eval()
            sample_fn_f = partial(self.apply_net, net=sample_net_f, fb="f")
        except KeyError:
            sample_fn_f = None

        if self.cdsb:
            y_c = y_c.expand(final_batch_x.shape[0], *self.shape_y).clone().to(self.device)
        node_drift = self.langevin.probability_flow_ode(net_f=sample_fn_f, net_b=sample_fn_b, y=y_c)

        with torch.no_grad():
            final_batch_x = final_batch_x.to(self.device)
            node_drift.nfe = 0
            rtol = atol = self.ode_tol
            x_tot_c = odeint(node_drift, final_batch_x, t=torch.linspace(self.langevin.T, 0, self.num_steps+1).to(self.device),
                             method=self.ode_sampler, rtol=rtol, atol=atol, options={'step_size': self.ode_euler_step_size})[1:]

            if not permute:
                x_tot_c = x_tot_c.permute(1, 0, *list(range(2, len(x_tot_c.shape))))  # (num_samples, num_steps, *shape_x)

        return x_tot_c, node_drift.nfe

    def forward_sample(self, init_batch_x, init_batch_y, permute=True, num_steps=None):
        sample_net = self.get_sample_net('f')
        sample_net.eval()
        sample_fn = partial(self.apply_net, net=sample_net, fb="f")

        with torch.no_grad():
            # self.set_seed(seed=0 + self.accelerator.process_index)
            init_batch_x = init_batch_x.to(self.device)
            # init_batch_y = init_batch_y.to(self.device)
            # assert not self.cond_final
            mean_final = self.mean_final.to(self.device)
            var_final = self.var_final.to(self.device)
            # if n == 0:

            #     x_tot, _, _, _ = self.langevin.record_init_langevin(init_batch_x, init_batch_y,
            #                                                         mean_final=mean_final, var_final=var_final)
            # else:
            x_tot, _, _, _ = self.langevin.record_langevin_seq(sample_fn, init_batch_x, init_batch_y, 'f', sample=self.transfer,
                                                               var_final=var_final, num_steps=num_steps)

        if permute:
            x_tot = x_tot.permute(1, 0, *list(range(2, len(x_tot.shape))))  # (num_steps, num_samples, *shape_x)

        return x_tot, self.num_steps if num_steps is None else num_steps
    
    # def forward_sample_ode(self, init_batch_x, init_batch_y, permute=True):
    #     sample_net_b = self.get_sample_net('b')
    #     sample_net_b.eval()
    #     sample_fn_b = partial(self.apply_net, net=sample_net_b, fb="b")
    #     try:
    #         sample_net_f = self.get_sample_net('f')
    #         sample_net_f.eval()
    #         sample_fn_f = partial(self.apply_net, net=sample_net_f, fb="f")
    #     except KeyError:
    #         sample_fn_f = None

    #     node_drift = self.langevin.probability_flow_ode(net_f=sample_fn_f, net_b=sample_net_b)

    #     with torch.no_grad():
    #         init_batch_x = init_batch_x.to(self.device)
    #         node_drift.nfe = 0
    #         rtol = atol = self.args.ode_tol
    #         x_tot = odeint(node_drift, init_batch_x, t=torch.linspace(self.langevin.T, 0, self.num_steps+1).to(self.device),
    #                          method=self.args.ode_sampler, rtol=rtol, atol=atol, options={'step_size': self.args.ode_euler_step_size})[1:]

    #         if not permute:
    #             x_tot = x_tot.permute(1, 0, *list(range(2, len(x_tot.shape))))  # (num_samples, num_steps, *shape_x)

    #     return x_tot, node_drift.nfe

    def plot_and_test_step(self, i, n, fb, sampler=None):
        #self.set_seed(seed=0 + self.accelerator.process_index)
        test_metrics = self.plotter(i, n, fb, sampler='sde' if sampler is None else sampler)

        #if self.accelerator.is_main_process:
        self.save_logger.log_metrics(test_metrics, step=self.compute_current_step(i, n))
        return test_metrics

    def set_seed(self, seed=0):
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)

    def clear(self):
        #self.accelerator.free_memory()
        torch.cuda.empty_cache()

    def save_step(self, i, n, fb):
        num_iter = self.compute_max_iter(fb, n)

        if i % self.stride == 0 or i == num_iter:
            self.save_ckpt(i, n, fb)
            self.plot_and_test_step(i, n, fb)
    
    def save_ckpt(self, i, n, fb):
        #if self.accelerator.is_main_process:
        name_net = f'net_{fb}_{n:03}_{i:07}.ckpt'
        name_net_ckpt = os.path.join(self.ckpt_dir, name_net)
        #torch.save(self.accelerator.unwrap_model(self.net[fb]).state_dict(), name_net_ckpt)
        torch.save(self.net[fb].state_dict(), name_net_ckpt)
        name_opt = f'optimizer_{fb}_{n:03}_{i:07}.ckpt'
        name_opt_ckpt = os.path.join(self.ckpt_dir, name_opt)
        torch.save(self.optimizer[fb].optimizer.state_dict(), name_opt_ckpt)

        if self.ema:
            #sample_net = self.ema_helpers[fb].ema_copy(self.accelerator.unwrap_model(self.net[fb]))
            sample_net = self.ema_helpers[fb].ema_copy(self.net[fb])
            name_net = f'sample_net_{fb}_{n:03}_{i:07}.ckpt'
            name_net_ckpt = os.path.join(self.ckpt_dir, name_net)
            torch.save(sample_net.state_dict(), name_net_ckpt)
            
            # for ckpt_prefix in self.ckpt_prefixes:
            #     existing_ckpts = sorted(glob.glob(os.path.join(self.ckpt_dir, f"{ckpt_prefix}_**.ckpt")))
            #     for ckpt_i in range(max(len(existing_ckpts)-5, 0)):
            #         os.remove(existing_ckpts[ckpt_i])

    def save_current_ckpt(self):
        self.save_ckpt(self.i, self.n, self.fb)
    
    def find_last_ckpt(self):
        existing_ckpts_dict = {}
        for ckpt_prefix in self.ckpt_prefixes:
            existing_ckpts = sorted(glob.glob(os.path.join(self.ckpt_dir_load, f"{ckpt_prefix}_**.ckpt")))
            existing_ckpts_dict[ckpt_prefix] = set([os.path.basename(existing_ckpt)[len(ckpt_prefix)+1:-5] for existing_ckpt in existing_ckpts])
        
        existing_ckpts_b = sorted(list(existing_ckpts_dict["net_b"].intersection(existing_ckpts_dict["sample_net_b"], existing_ckpts_dict["optimizer_b"])), reverse=True)
        existing_ckpts_f = sorted(list(existing_ckpts_dict["net_f"].intersection(existing_ckpts_dict["sample_net_f"], existing_ckpts_dict["optimizer_f"])), reverse=True)

        if len(existing_ckpts_b) == 0:
            return False, 1, 'b', 1, None, None
        
        def return_valid_ckpt_combi(b_i, b_n, f_i=None, f_n=None):
            # Return is_valid, checkpoint_it, checkpoint_pass, checkpoint_step
            if f_i is None:  # f_n = 0, b_n = 1
                if b_n == 1:
                    num_iter = self.first_num_iter
                    if b_i == num_iter:
                        return True, 1, 'f', 1
                    else:
                        return True, 1, 'b', b_i + 1
                else:
                    return False, None, None, None
            if b_n < f_n or b_n > f_n + 1:
                return False, None, None, None
            
            num_iter = self.compute_max_iter('f', f_n)

            if (b_n == 1 and b_i != self.first_num_iter) or (b_n > 1 and b_i != self.num_iter):  # during b pass
                if f_i == num_iter and f_n == b_n - 1:
                    return True, b_n, 'b', b_i + 1
                return False, None, None, None
            else:  # before or during or end of f pass
                if f_i != num_iter:  # during f pass
                    assert f_i < num_iter
                    if f_n == b_n:
                        return True, f_n, 'f', f_i + 1
                    return False, None, None, None
                else:
                    if f_n == b_n:
                        return True, b_n + 1, 'b', 1
                    else:
                        return True, b_n, 'f', 1

        existing_ckpt_f = None
        for existing_ckpt_b in existing_ckpts_b:
            ckpt_b_n, ckpt_b_i = existing_ckpt_b.split("_")
            ckpt_b_n, ckpt_b_i = int(ckpt_b_n), int(ckpt_b_i)
            if len(existing_ckpts_f) == 0:
                is_valid, checkpoint_it, checkpoint_pass, checkpoint_step = return_valid_ckpt_combi(ckpt_b_i, ckpt_b_n)
                if is_valid:
                    break
            else:
                for existing_ckpt_f in existing_ckpts_f:
                    ckpt_f_n, ckpt_f_i = existing_ckpt_f.split("_")
                    ckpt_f_n, ckpt_f_i = int(ckpt_f_n), int(ckpt_f_i)
                    is_valid, checkpoint_it, checkpoint_pass, checkpoint_step = return_valid_ckpt_combi(ckpt_b_i, ckpt_b_n, ckpt_f_i, ckpt_f_n)
                    if is_valid:
                        break
                if is_valid:
                    break

        if not is_valid:
            return False, 1, 'b', 1, None, None
        else:
            return True, checkpoint_it, checkpoint_pass, checkpoint_step, existing_ckpt_b, existing_ckpt_f

    def ipf_iter(self, forward_or_backward, n):
        # if self.first_pass:
        #     step = self.step
        # else:
        #     step = 1
        step = 1
        
        self.set_seed(seed=self.compute_current_step(step - 1, n))
        self.i, self.n, self.fb = step - 1, n, forward_or_backward

        # if (not self.first_pass) and (not self.use_prev_net):
        #     self.build_models(forward_or_backward)
        #     self.build_optimizers(forward_or_backward)

        #self.accelerate(forward_or_backward)

        # if (forward_or_backward not in self.ema_helpers.keys()) or ((not self.first_pass) and (not self.use_prev_net)):
        #     self.update_ema(forward_or_backward)
        
        num_iter = self.compute_max_iter(forward_or_backward, n)
        
        def first_it_fn(forward_or_backward, n):
            if self.first_coupling == 'ref':
                first_it = ((n == 1) and (forward_or_backward == 'b'))
            elif self.first_coupling == 'ind':
                first_it = (n == 1)
            return first_it
        first_it = first_it_fn(forward_or_backward, n)

        for i in tqdm(range(step, num_iter + 1), mininterval=30):
            
            if (i == step) or ((i-1) % self.cache_refresh_stride == 0):
                new_dl = None
                torch.cuda.empty_cache()
                if not first_it:
                    new_dl = self.new_cacheloader(*self.compute_prev_it(forward_or_backward, n), refresh_idx=(i-1) // self.cache_refresh_stride)

            self.net[forward_or_backward].train()

            self.set_seed(seed=self.compute_current_step(i, n))

            y = None
            if first_it:
                x0, y, x1, _, _ = self.sample_batch(self.init_dl, self.final_dl)
            else:
                if self.cdsb:
                    x0, x1, y = next(new_dl)
                else:
                    x0, x1 = next(new_dl)

            x0, x1 = x0.to(self.device), x1.to(self.device)
            x0, x1 = x0.repeat_interleave(self.num_repeat_data, dim=0), x1.repeat_interleave(self.num_repeat_data, dim=0)
            x, t, out = self.langevin.get_train_tuple(x0, x1, fb=forward_or_backward, first_it=first_it)

            if self.cdsb:
                y = y.to(self.device)
                y = y.repeat_interleave(self.num_repeat_data, dim=0)

            pred, std = self.apply_net(x, y, t, net=self.net[forward_or_backward], fb=forward_or_backward, return_scale=True)

            if self.loss_scale:
                loss_scale = std
            else:
                loss_scale = 1.

            loss = F.mse_loss(loss_scale*pred, loss_scale*out)

            #self.accelerator.backward(loss)
            loss.backward()

            if self.grad_clipping:
                clipping_param = self.grad_clip
                #total_norm = self.accelerator.clip_grad_norm_(self.net[forward_or_backward].parameters(), clipping_param)
                total_norm = torch.nn.utils.clip_grad_norm_(
                    self.net[forward_or_backward].parameters(), clipping_param
                )
            else:
                total_norm = 0.

            if i == 1 or i % self.stride_log == 0 or i == num_iter:
                self.logger.log_metrics({'fb': forward_or_backward,
                                         'ipf': n,
                                         'loss': loss,
                                         'grad_norm': total_norm,
                                         "cache_epochs": self.cache_epochs,
                                         "num_repeat_data": self.num_repeat_data,
                                         "data_epochs": self.data_epochs}, step=self.compute_current_step(i, n))

            self.optimizer[forward_or_backward].step()
            self.optimizer[forward_or_backward].zero_grad(set_to_none=True)
            if self.ema:
                #self.ema_helpers[forward_or_backward].update(self.accelerator.unwrap_model(self.net[forward_or_backward]))
                self.ema_helpers[forward_or_backward].update(self.net[forward_or_backward])

            self.i, self.n, self.fb = i, n, forward_or_backward

            if i != num_iter:
                self.save_step(i, n, forward_or_backward)

        # Pre-cache current iter at end of training
        new_dl = None
        self.save_ckpt(num_iter, n, forward_or_backward)
        if not first_it_fn(*self.compute_next_it(forward_or_backward, n)):
            self.new_cacheloader(forward_or_backward, n, build_dataloader=False)

        self.save_step(num_iter, n, forward_or_backward)

        #self.net[forward_or_backward] = self.accelerator.unwrap_model(self.net[forward_or_backward])
        self.clear()
        # self.first_pass = False

    def apply_net(self, x, y, t, net, fb, return_scale=False):
        out = net.forward(x, y, t)
        if (not self.loss_scale) and (not self.std_trick):
            std = 1.
        else:
            std = self.langevin.marginal_prob(None, t, 'b' if fb=='f' else 'f')[1]
            std = std.view([t.shape[0]] + [1]*(len(x.shape)-1))
            if self.std_trick:
                out = out / std

        if return_scale:
            return out, std
        else:
            return out

    def compute_current_step(self, i, n):
        return i + self.num_iter*max(n-2, 0) + self.first_num_iter * (1 if n > 1 else 0)
    
    def compute_max_iter(self, forward_or_backward, n):
        if n == 1:
            num_iter = self.first_num_iter
        else:
            num_iter = self.num_iter
        return num_iter

    def compute_prev_it(self, forward_or_backward, n):
        if forward_or_backward == 'b':
            prev_direction = 'f'
            prev_n = n-1
        else:
            prev_direction = 'b'
            prev_n = n
        return prev_direction, prev_n

    def compute_next_it(self, forward_or_backward, n):
        if forward_or_backward == 'b':
            next_direction = 'f'
            next_n = n
        else:
            next_direction = 'b'
            next_n = n+1
        return next_direction, next_n