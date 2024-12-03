from pydantic import BaseModel, model_serializer
from typing import Union, List, Tuple, Optional


class DataConfig(BaseModel):
    batch_size: int
    shuffle: bool
    num_workers: int
    prefetch_factor: int
    drop_last: bool

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

class EMAConfig(BaseModel):
    mu: float
