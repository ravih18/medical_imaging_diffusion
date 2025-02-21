from pydantic import BaseModel, model_serializer
from typing import Union, List, Tuple, Optional, Dict
import toml

class DataConfig(BaseModel):
    batch_size: int
    shuffle: bool
    num_workers: int
    prefetch_factor: int
    drop_last: bool
    pin_memory: bool

class TrainerConfig(BaseModel):
    n_epochs: int
    current_epoch: int
    
class UnetConfig(BaseModel):
    dim: int
    channels: int
    dim_mults: List[int]
    resnet_block_groups: int
    n_block_klass: int
    self_condition: bool = False
    init_dim: Optional[int] = None
    out_dim: Optional[int] = None
    
class DiffusionConfig(BaseModel):
    timesteps: int
    beta_start: float
    beta_end: float
    loss_type: str

class DsbConfig(BaseModel):
    n_ipf: int
    use_prev_net: bool
    num_steps: int
    gamma_space: str
    gamma_min: float
    gamma_max: float
    num_cache_batches: int
    cache_batch_size: int
    cache_refresh_stride: int
    batch_size: int
    lr: float
    num_iter: int
    ema: bool
    ema_rate: float
    grad_clipping: bool
    grad_clip: float
    data_param: DataConfig
    unet_param: UnetConfig

class DsbmConfig(BaseModel):
    n_ipf: int
    use_prev_net: bool
    num_steps: int
    symmetric_gamma: bool
    gamma_space: str
    gamma_min: float
    gamma_max: float
    # num_cache_batches: int
    cache_batch_size: int
    cache_refresh_stride: int
    batch_size: int
    lr: float
    num_iter: int
    ema: bool
    ema_rate: float
    grad_clipping: bool
    grad_clip: float
    data_param: DataConfig
    unet_param: UnetConfig
    sde: str
    first_coupling: str
    loss_scale: bool
    ode_sampler: str #dopri5
    ode_tol: float #1e-5
    ode_euler_step_size: float #1e-2
    cache_npar: Optional[int] = None
    num_repeat_data: Optional[int] = 1
    first_num_iter: Optional[int] = None
    std_trick: Optional[bool] = False
    cache_num_steps: Optional[int] = None
    test_num_steps: Optional[int] = None

class EMAConfig(BaseModel):
    mu: float


def get_configs_from_toml(config_toml):
    with open(config_toml, 'r') as f:
        config = toml.load(f)

    data_config = DataConfig.parse_obj(config["Data"])
    unet_config = UnetConfig.parse_obj(config["Unet"])
    diffusion_config = DiffusionConfig.parse_obj(config["Diffusion"])
    return data_config, unet_config, diffusion_config


def dsb_config_from_toml(config_toml):
    with open(config_toml, 'r') as f:
        config = toml.load(f)

    data_config = DataConfig.parse_obj(config["Data"])
    unet_config = UnetConfig.parse_obj(config["Unet"])
    param_dict = config["Ipf"]
    param_dict.update(config["Training"])
    param_dict["data_param"] = data_config
    param_dict["unet_param"] = unet_config
    dsb_config = DsbConfig.parse_obj(param_dict)
    return dsb_config

def dsbm_config_from_toml(config_toml):
    with open(config_toml, 'r') as f:
        config = toml.load(f)

    data_config = DataConfig.parse_obj(config["Data"])
    unet_config = UnetConfig.parse_obj(config["Unet"])
    param_dict = config["Ipf"]
    param_dict.update(config["Training"])
    param_dict.update(config["Sampling"])
    param_dict["data_param"] = data_config
    param_dict["unet_param"] = unet_config
    dsb_config = DsbmConfig.parse_obj(param_dict)
    return dsb_config
