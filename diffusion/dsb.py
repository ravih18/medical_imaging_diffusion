from pathlib import Path
#import time
#import datetime
from tqdm import tqdm

import random
import numpy as np

import torch 
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.profiler import profile, tensorboard_trace_handler, ProfilerActivity, schedule
import gc

from .langevin import Langevin
from model.ema import EMAHelper
from image_datasets.cacheloader import CacheLoader
from image_datasets import repeater
from utils.config import DsbConfig
from utils import print_memory_usage, print_param_diff


class DiffusionSchrodingerBridge(nn.Module):
    def __init__(
        self,
        #caps_directory: Path,
        experiment_directory: Path,
        dsb_params: DsbConfig,
        datasets: dict,
        transfer: bool,
        evaluation: bool = False,
    ):
        super(DiffusionSchrodingerBridge, self).__init__()

        self.experiment_directory = experiment_directory
        self.transfer = transfer
        self.init_expe(experiment_directory)

        #self.accelerator = Accelerator(mixed_precision="fp16", cpu=(not torch.cuda.is_available()))
        #self.device = self.accelerator.device
        self.device = "cuda" if torch.cuda.is_available() else 'cpu'

        # IPF parameters
        self.n_ipf = dsb_params.n_ipf
        self.use_prev_net = dsb_params.use_prev_net
        self.num_cache_batches = dsb_params.num_cache_batches
        self.cache_batch_size = dsb_params.cache_batch_size
        self.cache_refresh_stride = dsb_params.cache_refresh_stride
        # Intialise gamma
        self.num_steps = dsb_params.num_steps
        gammas = self.build_gammas(dsb_params)
        self.T = torch.sum(gammas)

        # Training parameters
        self.lr = dsb_params.lr
        self.batch_size = dsb_params.batch_size
        self.num_iter = dsb_params.num_iter
        self.ema = dsb_params.ema
        self.grad_clipping = dsb_params.grad_clipping
        self.grad_clip = dsb_params.grad_clip

        # get models
        self.build_models(dsb_params.unet_param)
        self.build_ema(dsb_params.ema_rate)

        # get optims
        self.build_optimizers()

        # get dataloaders
        #self.build_dataloaders(dsb_params.data_param)
        self.build_dataloaders(datasets, dsb_params.data_param)
    
        # get loggers
        self.logger = self.get_logger()
        self.save_logger = self.get_logger('plot_logs')
        self.plotter = self.get_plotter()
        self.stride = 5000
        self.stride_log = 10

        # langevin
        # if dsb_params.weight_distrib:
        #     alpha = dsb_params.weight_distrib_alpha
        #     prob_vec = (1 + alpha) * torch.sum(gammas) - \
        #         torch.cumsum(gammas, 0)
        # else:
        #     prob_vec = gammas * 0 + 1
        prob_vec = gammas * 0 + 1 # no weight distrib
        time_sampler = torch.distributions.categorical.Categorical(prob_vec)

        shape = next(self.save_init_dl)['image'][0].shape
        self.shape = shape
        self.langevin = Langevin(
            self.num_steps,
            shape,
            gammas,
            time_sampler,
            device=self.device,
            mean_final=self.mean_final,
            var_final=self.var_final
        )

        # checkpoint
        # date = str(datetime.datetime.now())[0:10]
        # self.name_all = date

        self.checkpoint_it = 1
        self.checkpoint_pass = 'b'

        self.dataparallel = False

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

    #def accelerate(self, forward_or_backward):
    #    self.net[forward_or_backward], self.optimizer[forward_or_backward] = self.accelerator.prepare(
    #        self.net[forward_or_backward],
    #        self.optimizer[forward_or_backward]
    #    )

    def build_gammas(self, dsb_params):
        n = self.num_steps//2
        if dsb_params.gamma_space == 'linspace':
            gamma_half = np.linspace(dsb_params.gamma_min, dsb_params.gamma_max, n)
        elif dsb_params.gamma_space == 'geomspace':
            gamma_half = np.geomspace(
                dsb_params.gamma_min, dsb_params.gamma_max, n
            )
        gammas = np.concatenate([gamma_half, np.flip(gamma_half)])
        return torch.tensor(gammas).to(self.device)

    def build_models(self, unet_config, forward_or_backward=None):
        from model.unet import Unet

        net_f, net_b = Unet(**unet_config.model_dump()), Unet(**unet_config.model_dump())
        #from model.unet_0.unet import UNetModel

        #attention_ds = []
        #for res in [1, 6, 8]:
        #    attention_ds.append(64 // res)

        # kwargs = {
        #     "in_channels": 1,
        #     "model_channels": 64,
        #     "out_channels": 1,
        #     "num_res_blocks": 2,
        #     "attention_resolutions": tuple(attention_ds),
        #     "dropout": 0.0,
        #     "channel_mult": (1, 2, 3, 4),
        #     "num_classes": None,
        #     "use_checkpoint": False,
        #     "num_heads": 4,
        #     "num_heads_upsample": -1,
        #     "use_scale_shift_norm": True
        # }
        # net_f, net_b = UNetModel(**kwargs), UNetModel(**kwargs)

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

    def update_ema(self, forward_or_backward):
        if self.ema:
            self.ema_helpers[forward_or_backward] = EMAHelper(
                mu=self.ema_rate, device=self.device)
            self.ema_helpers[forward_or_backward].register(
                self.net[forward_or_backward])

    def build_ema(self, ema_rate):
        if self.ema:
            self.ema_rate = ema_rate
            self.ema_helpers = {}
            self.update_ema('f')
            self.update_ema('b')

    def build_optimizers(self):
        from torch.optim import Adam

        optimizer_b = Adam(self.net['b'].parameters(), lr=self.lr)
        optimizer_f = Adam(self.net['f'].parameters(), lr=self.lr)
        self.optimizer = {'f': optimizer_f, 'b': optimizer_b}

    def build_dataloaders(self, datasets, data_param):

        self.mean_final = torch.tensor(0.).to(self.device) # à vérifier
        self.var_final = torch.tensor(1.).to(self.device) # à vérifier
        self.std_final = torch.sqrt(self.var_final).to(self.device)

        save_init_dl = DataLoader(
            datasets["train_init"],
            #worker_init_fn = worker_init_fn,
            **data_param.model_dump()
        )
        save_final_dl = DataLoader(
            datasets["train_final"],
            #worker_init_fn = worker_init_fn,
            **data_param.model_dump()
        )
        cache_init_dl = DataLoader(
            datasets["train_init"],
            batch_size = self.cache_batch_size,
            **data_param.model_dump(exclude={'batch_size'})
        )
        cache_final_dl = DataLoader(
            datasets["train_final"],
            batch_size = self.cache_batch_size,
            **data_param.model_dump(exclude={'batch_size'})
        )

        val_init_dl = DataLoader(
            datasets["val_init"],
            batch_size=64,
            shuffle=True,
            num_workers=8,
            drop_last=True,
        )
        val_final_dl = DataLoader(
            datasets["val_final"],
            batch_size=64,
            shuffle=True,
            num_workers=8,
            drop_last=True,
        )

        #save_init_dl, save_final_dl, cache_init_dl, cache_final_dl = self.accelerator.prepare(
        #    save_init_dl,
        #    save_final_dl,
        #    cache_init_dl,
        #    cache_final_dl
        #)

        self.save_init_dl = repeater(save_init_dl)
        self.save_final_dl = repeater(save_final_dl)
        self.cache_init_dl = repeater(cache_init_dl)
        self.cache_final_dl = repeater(cache_final_dl)
        self.val_init_dl = repeater(val_init_dl)
        self.val_final_dl = repeater(val_final_dl)

    def new_cacheloader(self, forward_or_backward, n, use_ema=True):
        sample_direction = 'f' if forward_or_backward == 'b' else 'b'
        if use_ema:
            sample_net = self.ema_helpers[sample_direction].ema_copy(
                self.net[sample_direction]
            )
        else:
            sample_net = self.net[sample_direction]

        if forward_or_backward == 'b':
            #sample_net = self.accelerator.prepare(sample_net)
            new_dl = CacheLoader(
                'b',
                sample_net,
                self.cache_init_dl,
                self.num_cache_batches,
                self.langevin, n,
                mean=None,
                std=None,
                batch_size=self.cache_batch_size,
                device=self.device,
                dataloader_f=self.cache_final_dl,
                transfer=self.transfer,
            )
        else:  # forward
            #sample_net = self.accelerator.prepare(sample_net)
            new_dl = CacheLoader(
                'f',
                sample_net,
                None,
                self.num_cache_batches,
                self.langevin,
                n,
                mean=self.mean_final,
                std=self.std_final,
                batch_size=self.cache_batch_size,
                device=self.device,
                dataloader_f=self.cache_final_dl,
                transfer=self.transfer,
            )

        new_dl = DataLoader(new_dl, batch_size=self.batch_size)

        #new_dl = self.accelerator.prepare(new_dl) # memory issue
        new_dl = repeater(new_dl)

        return new_dl

    def load_checkpoints(self, i, n, fb=None):
        if (fb=='f') or (not fb):
            self.ema_helpers["f"].load_state_dict(torch.load(
                self.experiment_directory / "checkpoints" / f"sample_net_f_{i}_{n}.ckpt"
            ))
        if (fb=='b') or (not fb):
            self.ema_helpers["b"].load_state_dict(torch.load(
                self.experiment_directory / "checkpoints" / f"sample_net_b_{i}_{n}.ckpt"
            ))

    def sample_batch(self, batch, fb):
        with torch.no_grad():
            x_tot, _, _ = self.langevin.record_langevin_seq(
                self.ema_helpers[fb].ema_copy(self.net[fb]),
                #dsb.net[fb],
                batch.to(self.device),
                sample=True
            )
        return x_tot

    def save_step(self, i, n, fb):
        #if self.accelerator.is_local_main_process:
        if ((i % self.stride == 0) or (i % self.stride == 1)) and (i > 0):

            if self.ema:
                sample_net = self.ema_helpers[fb].ema_copy(self.net[fb])
            else:
                sample_net = self.net[fb]

            name_net_ckpt = self.experiment_directory \
                            / "checkpoints" \
                            / f"net_{fb}_{n}_{i}.ckpt"

            if self.dataparallel:
                torch.save(self.net[fb].module.state_dict(), name_net_ckpt)
            else:
                torch.save(self.net[fb].state_dict(), name_net_ckpt)

            if self.ema:
                name_net_ckpt = self.experiment_directory \
                                / "checkpoints" \
                                / f"sample_net_{fb}_{n}_{i}.ckpt"
                if self.dataparallel:
                    torch.save(
                        sample_net.module.state_dict(),
                        name_net_ckpt
                    )
                else:
                    torch.save(sample_net.state_dict(), name_net_ckpt)

            with torch.no_grad():
                #self.set_seed(seed=0 + self.accelerator.process_index)
                if fb == 'f':
                    batch = next(self.val_init_dl)['image']
                    batch = batch.to(self.device)
                elif self.transfer:
                    batch = next(self.val_final_dl)['image']
                    batch = batch.to(self.device)
                else:
                    batch = self.mean_final + self.std_final * \
                        torch.randn(
                            (self.batch_size, *self.shape),
                            device=self.device,
                        )

                x_tot, out, steps_expanded = self.langevin.record_langevin_seq(
                    sample_net,
                    batch,
                    ipf_it=n,
                    sample=True
                )
                shape_len = len(x_tot.shape)
                x_tot = x_tot.permute(1, 0, *list(range(2, shape_len)))
                x_tot_plot = x_tot.detach()  # .cpu().numpy()

            init_x = batch.detach().cpu().numpy()
            final_x = x_tot_plot[-1].detach().cpu().numpy()
            std_final = np.std(final_x)
            std_init = np.std(init_x)
            mean_final = np.mean(final_x)
            mean_init = np.mean(init_x)

            print('Initial variance: ' + str(std_init ** 2))
            print('Final variance: ' + str(std_final ** 2))

            self.save_logger.log_metrics(
                {
                    'FB': fb,
                    'init_var': std_init**2,
                    'final_var': std_final**2,
                    'mean_init': mean_init,
                    'mean_final': mean_final,
                    'T': self.T
                }
            )
            self.x_tot_plot = x_tot_plot
            self.plotter(batch, x_tot_plot, i, n, fb)

            gc.collect()

    def set_seed(self, seed=0):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)

    def clear_gpu_memory(self):
        torch.cuda.empty_cache()
        variables = gc.collect()
        del variables
        torch.cuda.empty_cache()
        # then collect the garbage
        gc.collect()

    def clear(self):
        torch.cuda.empty_cache()


class IPF(DiffusionSchrodingerBridge):

    def __init__(
        self,
        experiment_directory: Path,
        dsb_params: DsbConfig,
        datasets: dict,
        transfer: bool,
    ):
        super(IPF, self).__init__(
            experiment_directory,
            dsb_params,
            datasets,
            transfer,
        )

    def ipf_step(self, forward_or_backward, n):
        new_dl = None # why ?
        torch.cuda.empty_cache()
        
        new_dl = self.new_cacheloader(forward_or_backward, n, self.ema)

        #if not self.use_prev_net: # Technique 5
        #    self.build_models(forward_or_backward)
        #    self.update_ema(forward_or_backward)

        prev_params = {name: param.data.clone() for name, param in self.net[forward_or_backward].named_parameters()}

        self.build_optimizers()
        #self.accelerate(forward_or_backward)

        # with profile(
        #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        #     schedule=schedule(wait=1, warmup=1, active=12, repeat=1),
        #     on_trace_ready=tensorboard_trace_handler(self.experiment_directory / 'profiler'),
        #     profile_memory=True
        # ) as prof:

        for i in tqdm(range(self.num_iter+1)):
            self.optimizer[forward_or_backward].zero_grad()
            self.set_seed(seed=n * self.num_iter+i)

            x, out, steps_expanded = next(new_dl)
            x = x.to(self.device)
            out = out.to(self.device)
            steps_expanded = steps_expanded.to(self.device)
            # eval_steps = self.num_steps - 1 - steps_expanded
            eval_steps = self.T - steps_expanded
            pred = self.net[forward_or_backward](x, eval_steps)

            loss = F.mse_loss(pred, out)

            loss.backward()
            #self.accelerator.backward(loss)

            if self.grad_clipping:
                clipping_param = self.grad_clip
                total_norm = torch.nn.utils.clip_grad_norm_(
                    self.net[forward_or_backward].parameters(), clipping_param
                )
            else:
                total_norm = 0.

            if (i % self.stride_log == 0) and (i > 0):
                self.logger.log_metrics(
                    {
                        'forward_or_backward': forward_or_backward,
                        'loss': loss,
                        'grad_norm': total_norm
                    },
                    step=i + self.num_iter * n
                )

            self.optimizer[forward_or_backward].step()
            self.optimizer[forward_or_backward].zero_grad()

            if self.ema:
                self.ema_helpers[forward_or_backward].update(
                    self.net[forward_or_backward])

            self.save_step(i, n, forward_or_backward)

            #if (i % 100 == 0):
            #    print(f"After Iteration {i + 1}:")
            #    print_param_diff(self.net[forward_or_backward], prev_params)
            #    prev_params = {name: param.data.clone() for name, param in self.net[forward_or_backward].named_parameters()}

            if (i % self.cache_refresh_stride == 0) and (i > 0):
                new_dl = None
                torch.cuda.empty_cache()
                new_dl = self.new_cacheloader(
                    forward_or_backward,
                    n,
                    self.ema
                )

            # Profiler
            #prof.step()

        # print_memory_usage("Before deleting cacheloader")
        new_dl = None # Why ? probably memory clear
        # del new_dl
        torch.cuda.empty_cache()
        # gc.collect()
        # torch.cuda.empty_cache()
        # print_memory_usage("After deleting cacheloader")

    def initial_fwd_pass(self):
        """See what does this do
        It seems like it is mainly to make a plot in the original implementation
        """
        #if self.accelerator.is_local_main_process:
        init_sample = next(self.save_init_dl)['image']
        init_sample = init_sample.to(self.device)
        x_tot, _, _ = self.langevin.record_init_langevin(init_sample)
        shape_len = len(x_tot.shape)
        x_tot = x_tot.permute(1, 0, *list(range(2, shape_len)))
        x_tot_plot = x_tot.detach()  # .cpu().numpy()

        self.plotter(init_sample, x_tot_plot, 0, 0, 'f')
        x_tot_plot = None
        x_tot = None
        torch.cuda.empty_cache()

    def train(self):
        
        print("Training initial forward pass")
        self.initial_fwd_pass()

        for n in range(self.checkpoint_it, self.n_ipf+1):
            print(f"IPF iteration: {n}/{self.n_ipf}")
            # BACKWARD OPTIMISATION
            if (self.checkpoint_pass == 'f') and (n == self.checkpoint_it):
                self.ipf_step('f', n)
            else:
                self.ipf_step('b', n)
                self.ipf_step('f', n)